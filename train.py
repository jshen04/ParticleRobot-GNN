import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from sim import GraphSimulator
from graph_nn import Net

writer = SummaryWriter('runs/vanilla_gnn')

numbots = 9

simulator = GraphSimulator(num=numbots, start=(300, 300), goal=(0, 0), visualizer=None)
net = Net(in_features=6, latent_features=12, out_features=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

optim = torch.optim.SGD(net.parameters(), lr=0.001)
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

        out = net(data)
        loss = loss_fn(out, data.y)
        loss.backward()
        optim.step()
        optim.zero_grad()
        episode_loss = episode_loss + loss.item()
        writer.add_scalar("Loss/train", loss, epoch)

    print("Episode: {}; Episode Loss: {:.4e}".format(epoch + 1, episode_loss))

torch.save(net.state_dict(), "./models/gnn_v0.pt")
writer.flush()
writer.close()

# episodesteps = 2500
# for epoch in range(10):
#     episodeloss = 0
#     simulator.setup()
#     for step in range(episodesteps):
#         totalSteps, actions = simulator.wavePolicy()
#         for j in range(totalSteps):
#             data = simulator.step(actions[j])
#             out = net(data)
#
#         out = net(data)
#
#         loss = loss_fn(out, data.y)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#
#         episodeloss = episodeloss + loss.item()
#
#     print("Episode: {}; Mean Step Loss: {:.4e}".format(epoch, episodeloss / episodesteps))
#
# torch.save(net.state_dict(), "./models/gnn_test.pt")
