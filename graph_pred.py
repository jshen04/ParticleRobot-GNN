import torch
from torch_geometric.nn import GAE
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from gae import GCNEncoder
from sim import GraphSimulator

writer = SummaryWriter('run/more_predictive_gae')

numbots = 9

simulator = GraphSimulator(num=numbots, start=(300, 300), goal=(0, 0), visualizer=None)
net = GAE(GCNEncoder(in_features=6, latent_features=4))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

optim = torch.optim.Adam(net.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

net.to(device)
net.train()

episodesteps = 2500
for epoch in range(2000):
    episode_loss = 0

    simulator.setup()
    for step in range(episodesteps):
        action = np.random.randint(2, size=numbots)
        data = simulator.step(action)
        data.to(device)

        out = net.encode(data.x, data.edge_index)
        loss = net.recon_loss(out, data.edge_index) + loss_fn(out, data.y[:, [0, 1, 2, 3]])
        loss.backward()
        optim.step()
        optim.zero_grad()
        episode_loss = episode_loss + loss.item()
        writer.add_scalar("Loss/train", loss, epoch)

    print("Episode: {}; Episode Loss: {:.4e}".format(epoch + 1, episode_loss))

torch.save(net.state_dict(), "./models/morepredgae_v0.pt")
writer.flush()
writer.close()
