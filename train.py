import torch
from torch.nn import functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import wandb.wandb_run
from deep_vgae import DeepVGAE
import random
import numpy as np
import wandb
from wandb.sync import SyncManager
from wandb.sdk.wandb_run import Run
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.data import Data, Batch
import os
import json
import time
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from argparse import Namespace


default_config = {
    'dataset': 'MUTAG',
    'batch_size': 1024,
    'hidden': 16,
    'epochs': 500,
    'classify': True,
    'labeler': True,
    'lr': 0.001,
    'dropout': 0,
    'mixtures': 4,
    'temp_mixture': 10,
    'dynamic_temp': False,
    'folds': 10,
    'fold': 0,
    'log_graphs': False,
    'export_model': True,
}


def load_folds(dataset: str, count: int, data: Data):
    file_path = f"./folds/{dataset}_{count}.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    skf = StratifiedKFold(n_splits=count, shuffle=True, random_state=0)
    folds = list(skf.split(np.arange(data.y.shape[0]), data.y))
    folds_split = []
    for fold in range(count):
        train_i_split, val_i_split = train_test_split(
            [int(i) for i in folds[fold][0]],
            stratify=[n for n in np.asarray(data.y)[folds[fold][0]]],
            test_size=int(len(list(folds[fold][0])) * 0.1),
            random_state=0)
        test_i_split = [int(i) for i in folds[fold][1]]
        folds_split.append([train_i_split, val_i_split, test_i_split])

    if not os.path.exists("./folds"):
        os.mkdir("./folds")
    with open(file_path, "w") as f:
        json.dump(folds_split, f)
    return folds_split


def permute_edges(tensor: torch.Tensor, torch_prev_state: torch.Tensor, seed: int = 0):
    prev_size = tensor.size()
    torch.random.set_rng_state(torch_prev_state)
    random.seed()
    r = torch.randperm(tensor.size()[1])
    torch.manual_seed(seed)
    random.seed(seed)
    tensor = torch.cat([tensor[0, r], tensor[1, r]])
    tensor.resize_(*prev_size)
    return tensor


def complete_node_features(data: Data, device: torch.device):
    if data.x is None:
        data.x = torch.zeros((data.num_nodes, 1))
    return data.to(device)


def load_dataset(dataset_name: str, fold: int, folds_count: int, device: torch.device = torch.device('cpu')):
    dataset = TUDataset('data', dataset_name, pre_transform=lambda data: complete_node_features(data, device))
    folds = load_folds(dataset_name, folds_count, dataset.data)
    train_i_split, val_i_split, test_i_split = folds[fold]
    return dataset, dataset[train_i_split], dataset[val_i_split], dataset[test_i_split]


def train_batch(model: DeepVGAE, batch: Batch, optimizer: torch.optim.Adam):
    optimizer.zero_grad()
    e_z = model(batch.x, batch.edge_index, batch.batch)
    loss = model.loss(batch.x, batch.edge_index, batch.batch, batch.y)
    train_loss = loss * len(batch)
    loss.backward()
    optimizer.step()
    return e_z, train_loss, loss


def test_step(model: DeepVGAE, test_loader: DataLoader, log_graphs=False):
    model.eval()
    test_loss = 0
    auc, ap = 0, 0
    total_length = 0
    table = wandb.Table(columns=["Original Graph", "Reconstructed Graph"])
    b = 0
    for batch in test_loader:
        b += 1
        print(f'''\rTest: {b: 4d}/{len(test_loader)}  {b/len(test_loader)*100:.2f}%''', end='')
        e_z = model(batch.x, batch.edge_index, batch.batch)
        test_loss += model.loss(batch.x, batch.edge_index, batch.batch, batch.y, False) * len(batch)
        auc_b, ap_b, pred_edges = model.test(
            e_z, batch.edge_index, model.get_neg_edge_indices(e_z, batch.edge_index), log_graphs)
        auc += auc_b * len(batch)
        ap += ap_b * len(batch)
        total_length += len(batch)
        if (log_graphs):
            for i in range(batch.num_graphs):
                x = batch.x[batch.batch == i]
                edge_index = torch.clone(batch.edge_index[:, batch.batch[batch.edge_index[0]] == i])
                edge_index -= edge_index.min()
                pred_edge_index = torch.clone(
                    pred_edges[:, (batch.batch[pred_edges[0]] == i) & (batch.batch[pred_edges[1]] == i)])
                pred_edge_index -= pred_edge_index.min()
                extract_graphs(table, i, x, edge_index, pred_edge_index)
    if (log_graphs):
        wandb.log({"Graphs": table})
    test_loss /= total_length
    auc /= total_length
    ap /= total_length
    return test_loss, auc, ap


def get_wandb_run_name(config: dict):
    wandb_name = [
        config["dataset"],
        f'c{int(config["classify"])}',
        f'm{config["mixtures"]}',
        f'd{int(config["dynamic_temp"])}',
    ]
    if not config['dynamic_temp']:
        wandb_name.append(f't{config["temp_mixture"]}')
    return '_'.join(wandb_name)


def sync_wandb(run: Run):
    try:
        folder = [d for d in os.listdir('wandb') if run.id in d][0]
        folder = os.path.abspath(f'''./wandb/{folder}''')
        sm = SyncManager()
        sm.add(folder)
        sm.start()
        while not sm.is_done():
            time.sleep(1)
    except:
        print('Folder not found')


def train(configs={}, project: str = 'lvae_test', entity: str = 'ripper346phd'):
    config = {**default_config, **configs}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('partial'):
        os.makedirs('partial')

    prev_state = torch.random.get_rng_state()
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    partial_model_file: str = f'''partial/model_checkpoint_id-{config.get('run_id', '')}.pth'''
    resumer: dict = {}
    if config.get('run_id') is not None and os.path.exists(partial_model_file):
        run = wandb.init(resume=config['run_id'])
        resumer = torch.load(partial_model_file)
        config = {**configs, **resumer['config']}

    dataset, train_dataset, val_dataset, test_dataset = load_dataset(
        config['dataset'], configs.get('fold', 0), configs.get('folds', 10), device)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

    config['num_node_features'] = dataset.num_node_features
    wandb_run_name = get_wandb_run_name(config)

    model = DeepVGAE(
        config['num_node_features'],
        config['hidden'],
        dataset.num_classes,
        config['dropout'],
        config['classify'],
        config['labeler'],
        None,
        config['mixtures'],
        config['temp_mixture'],
        device=device,
    )
    model = model.to(device)
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=config['lr'])

    run: wandb.wandb_run.Run = None
    if resumer:
        model.load_state_dict(resumer['model'])
        optimizer.load_state_dict(resumer['optimizer'])
        run = wandb.init(resume=config['run_id'])
    else:
        run = wandb.init(config=config, project=project, entity=entity, name=wandb_run_name,
                         mode='offline' if not config['log_graphs'] else 'online')
        partial_model_file = f'partial/model_checkpoint_id-{run.id}.pth'
    model.wandb_run = run

    loss = 0
    best_epoch = -1
    best_train_loss = None
    best_val_loss = np.Inf
    last_temp_increment = -1
    best_model: dict[str, any] = None
    # z = None
    print(f'Start train, {len(train_loader)} batches')
    for epoch in range(resumer.get('epoch', 0), config['epochs']):
        train_loss = 0
        model.train()
        b = 0
        for batch in train_loader:
            b += 1
            print(f'''\rTrain: {b: 4d}/{len(train_loader)}  {b/len(train_loader)*100: 3.2f}%''', end='')
            _, batch_loss, loss = train_batch(model, batch, optimizer)
            run.log({'epoch': epoch, 'loss': loss})
            train_loss += batch_loss
        train_loss /= len(train_dataset)
        print()

        with torch.no_grad():
            val_loss, auc, ap = test_step(model, val_loader)

        print(
            f'\rEpoch: {epoch:04d}, loss: {loss:.6f}, val loss: {val_loss:.6f}.' +
            f'AUC val: {auc:.4f}, AP val: {ap:.4f}.' +
            (' Best val loss' if val_loss < best_val_loss else '')
        )
        curr_temp = model.temp_mixture
        if val_loss < best_val_loss:
            best_train_loss = train_loss
            best_val_loss = val_loss
            best_epoch = epoch
            best_model = model.state_dict()
        elif config['dynamic_temp'] and epoch - best_epoch > 30 and epoch - last_temp_increment > 20 and model.temp_mixture > 1e-20:
            # increment temperature every 10 epochs after 30 epochs of no improvement
            last_temp_increment = epoch
            model.temp_mixture /= 10

        run.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auc': auc,
            'val_ap': ap,
            'temperature': curr_temp,
            'mixture_probs_x': model.encoder.classifier.mix_prob_x if model.classify else None,
            'mixture_scores_x': model.encoder.classifier.mixtures_x if model.classify else None,
        })
        if epoch % 10 == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                },
                partial_model_file)
        # model.gamma = model.gamma.detach()

    model.load_state_dict(best_model)
    if config['export_model']:
        torch.save(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
            },
            f'models/model_{run.id}_{wandb_run_name}.pth')
    if os.path.exists(partial_model_file):
        os.remove(partial_model_file)

    with torch.no_grad():
        test_loss, auc, ap = test_step(model, test_loader, config['log_graphs'])

    run.log({'best_epoch': best_epoch, 'best_val_loss': best_val_loss,
            'test_auc': auc, 'test_ap': ap, 'test_loss': test_loss})
    wandb.finish(0)
    sync_wandb(run)
    print(f'Best Epoch: {best_epoch:04d}, loss: {best_train_loss:.6f}, val loss: {best_val_loss:.6f}\nAUC test: {auc:.4f}, AP test: {ap:.4f}')


def extract_graphs(table, idx, batch_x, batch_edge_index, pred_edges):
    original_graph = to_networkx(Data(x=batch_x, edge_index=batch_edge_index), to_undirected=True)
    reconstructed_graph = to_networkx(Data(x=batch_x, edge_index=pred_edges), to_undirected=True)

    pos = nx.spring_layout(original_graph, seed=0)

    original_img = plt.figure(figsize=(4, 4))
    nx.draw(original_graph, pos, with_labels=True)
    plt.savefig(f"/tmp/original_{idx}.svg", format="svg")
    plt.close(original_img)

    reconstructed_img = plt.figure(figsize=(4, 4))
    nx.draw(reconstructed_graph, pos, with_labels=True)
    plt.savefig(f"/tmp/reconstructed_{idx}.svg", format="svg")
    plt.close(reconstructed_img)

    table.add_data(wandb.Html(f"/tmp/original_{idx}.svg"), wandb.Html(f"/tmp/reconstructed_{idx}.svg"))


if __name__ == '__main__':
    train({'mixtures': 4, 'dynamic_temp': True, 'log_graphs': True})
