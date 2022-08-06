import torch
import torch_geometric
from torch_geometric.nn import GCNConv, GAE

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_features, latent_features):
        super().__init__()
        self.conv1 = GCNConv(in_features, 2 * latent_features)
        self.conv2 = GCNConv(2 * latent_features, latent_features)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

