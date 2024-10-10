import torch
from torch import Tensor
from torch_geometric.nn.models import GCN
from torch_geometric.nn.aggr import SortAggregation
from torch_geometric.data import Batch


class Classifier(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int, classes: int):
        # The classifier to generate pi
        # This is a GCN with sort aggregation and a linear layer to predict the classes
        # Similar to DGCNN An End-to-End Deep Learning Architecture for Graph Classification Muhan Zhang, Zhicheng Cui, Marion Neumann, Yixin Chen
        super(Classifier, self).__init__()
        self.classnet = GCN(in_channels, hidden, 3)
        self.agg = SortAggregation(10)
        self.classfc = torch.nn.Linear(10 * hidden, classes)

        self.mixtures_x = None
        self.mix_prob_x = None

    def predict_class(self, x: Tensor, edge_index: Tensor, batch: Batch, training: bool = False) -> tuple[Tensor, Tensor]:
        # returns a tensor for the (soft) one-hot encoding of the classes of this graph
        c = self.classnet(x, edge_index).relu()
        c = self.agg(c, batch)
        c = self.classfc(c)
        mixtures_x = c[batch]
        mix_prob_g = torch.nn.functional.softmax(c, dim=1)
        mix_prob_x = mix_prob_g[batch]
        if training:
            self.mixtures_x = mixtures_x
            self.mix_prob_x = mix_prob_x
        return mixtures_x, mix_prob_x, mix_prob_g

    def loss(self) -> Tensor:
        probs_loss = -torch.sum(self.mix_prob_x * torch.log(self.mix_prob_x + 1e-15))
        return probs_loss
