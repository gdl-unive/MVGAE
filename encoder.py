import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GCN
from torch_geometric.nn.aggr import SortAggregation
from classifier import Classifier


class GCNEncoder(torch.nn.Module):
    def __init__(
        self,
            in_channels: int,
            hidden1: int,
            hidden2: int,
            hidden_classifier: int = None,
            mixtures: int = None,
            use_classify: bool = True,
            dropout: int = 0
    ):
        super(GCNEncoder, self).__init__()
        self.classifier: Classifier = None
        conv1_in = in_channels
        if mixtures is not None and use_classify:
            print("Using classifier")
            self.classifier = Classifier(conv1_in, hidden_classifier or hidden1, mixtures)
            conv1_in += mixtures
        else:
            print("Not using classifier")

        self.conv1 = GCNConv(conv1_in, hidden1)  # cached only for transductive learning
        self.conv_mu = GCNConv(hidden1, hidden2)
        self.conv_logstd = GCNConv(hidden1, hidden2)
        self.dropout = dropout
        self.sorted_idx = torch.randn(hidden2)

    def predict_params(self, x: Tensor, edge_index: Tensor, probs_mixture: Tensor = None) -> tuple[Tensor, Tensor]:
        # returns mu, sigma for the latent space distributions of the given graph, class and label probabilities probabilities
        if probs_mixture is not None:
            x = torch.cat((x, probs_mixture), 1)
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor, training: bool = True) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.dropout:
            x = torch.nn.functional.dropout(x, self.dropout)

        mixtures_x, mix_prob_x, mix_prob_g = self.classifier.predict_class(x, edge_index, batch, training) \
            if self.classifier is not None else (None, None, None)

        mu, logstd = self.predict_params(x, edge_index, mix_prob_x)
        return mu, logstd, mixtures_x, mix_prob_g
