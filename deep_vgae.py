import torch
from torch import Tensor
from typing import Optional, Tuple
from torch_geometric.nn import VGAE, InnerProductDecoder
from torch.nn.functional import one_hot
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from encoder import GCNEncoder
from torch_geometric.data.data import Data
from torch_geometric.data import Batch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
import numpy as np


class DeepVGAE(VGAE):
    encoder: GCNEncoder
    decoder: InnerProductDecoder

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        num_classes: Optional[int] = None,
        dropout: float = 0,
        classify: bool = True,
        labeler: bool = True,
        wandb_run=None,
        mixtures: Optional[int] = None,
        temp_mixture: float = 1.0,
        device: str = 'cpu',
    ):
        super(DeepVGAE, self).__init__(
            encoder=GCNEncoder(
                in_channels=in_channels,
                hidden1=2 * hidden,
                hidden2=hidden,
                hidden_classifier=hidden,
                mixtures=mixtures,
                use_classify=classify,
                dropout=dropout),
            decoder=InnerProductDecoder(),
        )
        self.eps = 1e-15

        self.wandb_run = wandb_run
        self.dropout: float = dropout
        self.classify: bool = classify
        self.labeler: bool = labeler
        self.num_classes: int = num_classes
        self.mixtures: int = mixtures
        self.temp_mixture: float = temp_mixture

        self.__mu__: Tensor = None
        self.__logstd__: Tensor = None
        self.__scores__: Tensor = None
        self.__scores_graph__: Tensor = None
        self.pi: Tensor = None
        self.gamma: Tensor = torch.nn.Parameter(torch.randn(mixtures, num_classes, requires_grad=True,
                                                            device=device).clamp(min=self.eps)) if self.labeler else None
        self.z: Tensor = None
        self.MAX_LOGSTD: int = 10

    def reparametrize(self, x: Tensor, edge_index: Tensor, mu: Tensor, scores: Tensor, training: bool = True) -> Tensor:
        if training:
            # Sampling to compute the data likelihood

            # First we sample a class from the class probs
            self.pi = None
            if self.classify:
                self.pi = torch.nn.functional.gumbel_softmax(scores, self.temp_mixture, dim=1)

            # Then predict mu, sigma from the given class
            mu, log_std = self.encoder.predict_params(x, edge_index, self.pi)
            return mu + torch.randn_like(log_std) * torch.exp(log_std)
        else:
            # In test mode, make a hard decision about the class
            probs = None
            if self.classify:
                classes = scores.argmax(dim=1)
                probs = torch.nn.functional.one_hot(classes, num_classes=self.mixtures)

            # Find mu matching this class
            mu, log_std = self.encoder.predict_params(x, edge_index, probs)
            return mu

    def encode(self, x: Tensor, edge_index: Tensor, batch: Batch, training: bool = True, encode_res: dict = None) -> Tensor:
        # check if setting scores is ok even when validating and testing
        mu, logstd, scores, scores_graph = self.encoder.forward(x, edge_index, batch, training)
        logstd = logstd.clamp(max=self.MAX_LOGSTD)
        if training:
            self.__mu__ = mu
            self.__logstd__ = logstd
            self.__scores__ = scores
            self.__scores_graph__ = scores_graph
        if encode_res is not None:
            encode_res['__mu__'] = mu
            encode_res['__logstd__'] = logstd
            encode_res['__scores__'] = scores
            encode_res['__scores_graph__'] = scores_graph

        z = self.reparametrize(x, edge_index, mu, scores, training=training)
        return z

    def kl_loss(self, training: bool = True, gamma=None) -> Tensor:
        # KL-loss modified with extra term for class probabilities
        mu = self.__mu__
        logstd = self.__logstd__
        probs_loss = 0
        if self.classify:
            label_loss = 0
            class_loss = self.encoder.classifier.loss()
            probs_loss += class_loss

            if self.labeler:
                probs_entropy = -torch.sum(gamma * torch.log(gamma + self.eps))
                label_loss = torch.sum(torch.tensordot(
                    self.encoder.classifier.mix_prob_x.transpose(1, 0), probs_entropy, dims=0))

                probs_loss += label_loss
            if self.wandb_run is not None and training:
                self.wandb_run.log({'label_kl_loss': label_loss, 'class_loss': class_loss})

        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1) + probs_loss)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Batch) -> Tensor:
        z = self.encode(x, edge_index, batch)
        self.z = z
        # adj_pred = self.decoder.forward_all(z)
        return z

    def loss(self, x: Tensor, edge_index: Tensor, batch: Batch, y: Tensor, training: bool = True) -> Tuple[Tensor, Tensor]:
        encode_res: dict = {}
        z = self.encode(x, edge_index, batch, training, encode_res) if not training else self.z

        pos_loss = -torch.log(self.decoder(z, edge_index, sigmoid=True) + self.eps).mean()
        neg_edge_index = self.get_neg_edge_indices(z, edge_index)
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + self.eps).mean()

        gamma = (self.gamma / self.gamma.sum(dim=1, keepdim=True)).clamp(min=self.eps)
        kl_loss = self.kl_loss(training, gamma) / x.size(0)

        label_loss = 0
        if self.labeler:
            label = (self.__scores_graph__ if training else encode_res['__scores_graph__']) @ gamma
            y_hot = one_hot(y, self.num_classes).float()
            label_loss = -torch.log(torch.sum(y_hot * label, dim=1) + self.eps).mean()
            if training:
                self.wandb_run.log({'label_loss': label_loss})

        return pos_loss + neg_loss + kl_loss + label_loss

    def get_neg_edge_indices(self, z: Tensor, edge_index: Tensor) -> Tensor:
        full_edge_index, _ = remove_self_loops(edge_index)
        full_edge_index, _ = add_self_loops(full_edge_index)
        return negative_sampling(full_edge_index, z.size(0), edge_index.size(1))

    def test(self, z: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor, log_graphs: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Base method copied from torch_geometric.nn.models.autoencoder.GAE

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        pred_edge_indices = None
        if log_graphs:
            fpr, tpr, thresholds = roc_curve(y, pred)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            pred_edges = (pred >= optimal_threshold).astype(int)
            # Filter edges based on the optimal threshold
            all_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            pred_edge_indices = all_edge_index[:, pred_edges == 1]

        return roc_auc_score(y, pred), average_precision_score(y, pred), pred_edge_indices
