import torch
import torch_geometric
from torch_geometric.nn import GCNConv

class Net(torch.nn.Module):
    def __init__(self, in_features, latent_features, out_features):
        super(Net, self).__init__()
        self.conv1 = GCNConv(in_features, 128)
        self.conv2 = GCNConv(128, latent_features)

        self.linear1 = torch.nn.Linear(latent_features, 512)
        self.linear2 = torch.nn.Linear(512, 512)
        self.linear3 = torch.nn.Linear(512, out_features)

        self.dropout = torch.nn.Dropout()


    def get_latent_state(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        return x

    def predict_from_latent(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.linear3(x)
        return x

    def forward(self, data):
        x = self.get_latent_state(data)
        x = self.predict_from_latent(x)
        return x
