import torch

class ImitationNet(torch.nn.Module):
    def __init__(self, features):
        super(ImitationNet, self).__init__()

        self.linear1 = torch.nn.Linear(features, 512)
        self.linear2 = torch.nn.Linear(512, 512)
        self.linear3 = torch.nn.Linear(512, 2)

        self.dropout = torch.nn.Dropout()


    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.linear3(x)
        return x
