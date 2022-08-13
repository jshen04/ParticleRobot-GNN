import torch
import torch_geometric
from torch_geometric.nn import GCNConv

class ImitationNet(torch.nn.Module):
    def __init__(self, features):
        super(ImitationNet, self).__init__()
        self.conv1 = GCNConv(features, 128)
        self.conv2 = GCNConv(128, 12)

        self.linear1 = torch.nn.Linear(12, 512)
        self.linear2 = torch.nn.Linear(512, 512)
        self.linear3 = torch.nn.Linear(512, 2)

        self.dropout = torch.nn.Dropout()

    def get_latent(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return torch.relu(x)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.linear3(x)
        return x
