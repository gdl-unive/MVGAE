
import fcntl
import itertools
import json
import pickle
import random
from datetime import datetime
from pathlib import Path
from time import sleep
import wandb
from train import train, default_config


def get_grid(file=None, idx=0):
    print(f'check {file.name}, idx: {idx}')
    config = {'group': None, 'project': None, 'grid': None, 'settings': None}
    with open(file) as f:
        config = json.load(f)
    is_list_file = isinstance(config, list)
    if is_list_file:
        if idx >= len(config):
            idx = -1
            config = {'group': None, 'project': None, 'grid': None, 'settings': None}
        else:
            config = config[idx]
    return config['group'], config['project'], config.get('grid'), config.get('settings'), is_list_file, idx


def is_param_list_in_wandb(params, runs_conf, keys_to_check=None):
    if keys_to_check is None:
        keys_to_check = params.keys()
    for r in runs_conf:
        same = True
        for k in keys_to_check:
            if r.get(k) is None or params.get(k) is None:
                continue
            if isinstance(params[k], list):
                for i in range(len(params[k])):
                    same = same and params[k][i] == r[k][i]
            else:
                same = same and params[k] == r[k]
        if same:
            return r
    return False


def wandb_checker(all_confs, group, project, runs_count=0):
    test = True

    try:
        api = wandb.Api(timeout=60)
        runs = api.runs(f"{group}/{project}")
        runs_conf = [crun.config for crun in runs if crun.state not in ['crashed', 'failed']
                     or (crun.state == 'finished' and crun.summary.get('test_acc') is not None)]
        [crun.delete() for crun in runs if crun.state in ['crashed', 'failed'] or (
            crun.state == 'finished' and crun.summary.get('test_acc') is None)]
        print(f"FOUND {len(runs_conf)} RUNS ON WANDB")
        print(f"Checking {len(all_confs)} CONFIGURATIONS")
        grid_keys = get_grid().keys()
        params_list = [p for p in all_confs if not is_param_list_in_wandb(p, runs_conf, grid_keys)]
        if len(params_list) == 0:
            raise NameError()
    except Exception as ex:
        print(ex)
        if runs_count < 10:
            sleep(10)
            return wandb_checker(all_confs, group, project, runs_count + 1)
        test = False
    return params_list if test else []


def local_wandb_checker(all_confs, group, project, grid):
    data = None
    runs_conf = None
    conf = None
    file = f'./configs_{project}.pickle'
    try:
        f = open(file, 'rb+')
        data = pickle.load(f)
    except FileNotFoundError as ex:
        f = open(file, 'wb')

    fcntl.flock(f, fcntl.LOCK_EX)
    print('Unlocked')
    try:
        api = wandb.Api(timeout=60)
        if data is None:
            print('New data')
            runs = api.runs(f"{group}/{project}", {"$and": [{"state": {"$nin": ["crashed", "failed"]}}]})
            print(f'Found {len(runs)} clean runs')
            runs_conf = [{'id': run.id, **run.config} for run in runs if run.config.get('dataset') is not None]
        else:
            print('Read data')
            runs_conf = data
            ids = [run['id'] for run in data if run.get('id')]
            print(f'Found {len(ids)} saved runs')
            runs = api.runs(f"{group}/{project}",
                            {"$and": [{"state": {"$nin": ["crashed", "failed"]}}, {"name": {"$nin": ids}}]})
            filter_in_runs = [{'id': run.id, **run.config} for run in runs if run.config.get('dataset') is not None]
            print(f'Found {len(filter_in_runs)} clean runs')
            runs = api.runs(f"{group}/{project}", {"$and": [{"state": {"$in": ["crashed", "failed"]}}]})
            filter_out_runs = [run.id for run in runs]
            print(f'Found {len(filter_out_runs)} wasted runs')
            runs_conf = [run for run in runs_conf if run.get('in_progress') or (run.get('id') is not None and
                         run['id'] not in filter_out_runs and run.get('dataset') is not None)] + filter_in_runs

        print(f'Found {len(runs_conf)} runs on wandb')
        print(f'Checking {len(all_confs)} configurations')
        for c in all_confs:
            if not is_param_list_in_wandb(c, runs_conf, grid.keys()):
                c['fake_id'] = random.randint(0, 10 ** 15)
                c['in_progress'] = 1
                conf = c
                runs_conf.append(conf)
                print(f'''Found new configuration. Id: {c['fake_id']}''')
                break
        f.seek(0)
        pickle.dump(runs_conf, f)
    except Exception as ex:
        raise ex
    finally:
        f.close()

    return conf


def search_grid_config_file():
    n = 0
    conf = None
    group = ''
    project = ''
    new_group = ''
    new_project = ''
    idx_config = None
    files = iter(sorted(list(Path('.').glob('configs_*.json'))[::-1], reverse=True))
    file = next(files, None)
    while conf is None and file is not None:
        if idx_config is None:
            new_group, new_project, grid, settings, is_list_file, idx_config = get_grid(file)
        else:
            new_group, new_project, grid, settings, is_list_file, idx_config = get_grid(file, idx_config)
        entered = True
        while conf is None:
            if entered:
                entered = False
            else:
                new_group, new_project, grid, settings, is_list_file, idx_config = get_grid(file, idx_config)
            if idx_config < 0:
                break
            group, project = new_group, new_project
            print(group, project)
            if grid is not None:
                all_confs = [{k: val for k, val in zip(grid, v)} for v in itertools.product(*[grid[k] for k in grid])]
                random.shuffle(all_confs)
                conf = local_wandb_checker(all_confs, group, project, grid)
            elif settings is not None:
                conf = local_wandb_checker(settings, group, project, settings[0])
            else:
                break
            if not is_list_file:
                break
            idx_config += 1
        file = next(files, None)
        idx_config = None
    return conf, group, project


def run_config(conf, group, project):
    print(f'Dataset: {conf["dataset"]}')
    default = default_config
    for k in conf.keys():
        default[k] = conf[k]
    default['date'] = f'{datetime.utcnow():%Y-%m-%d %H:%MZ}'
    try:
        train(default, project, group)
    except Exception as ex:
        raise ex
    finally:
        if conf.get('fake_id') is not None:
            file = f'./configs_{project}.pickle'
            with open(file, 'rb+') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                data = pickle.load(f)
                f.seek(0)
                pickle.dump([run for run in data if run.get('fake_id') != conf['fake_id']], f)


if __name__ == "__main__":
    print('hello', datetime.now())
    conf, group, project = search_grid_config_file()
    if conf is not None:
        run_config(conf, group, project)
