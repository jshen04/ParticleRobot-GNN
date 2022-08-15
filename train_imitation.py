import torch
import numpy as np
# from torch.utils.tensorboard import SummaryWriter

from sim import GraphSimulator
from imitation_learning import ImitationNet

# writer = SummaryWriter('runs/vanilla_gnn')

numbots = 9

simulator = GraphSimulator(num=numbots, start=(300, 300), goal=(0, 0), visualizer=None)
net = ImitationNet(features=6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

optim = torch.optim.SGD(net.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

net.to(device)
net.train()

episodesteps = 2500
for epoch in range(100):
    episodeloss = 0
    simulator.setup()
    for step in range(episodesteps):
        totalSteps, actions = simulator.wavePolicy()
        for j in range(totalSteps):
            data = simulator.step(actions[j])
            out = net(data)

            target = []
            for a in actions[j]:
                if a == 0:
                    target.append([1, 0])
                else:
                    target.append([0, 1])

            loss = loss_fn(out, torch.tensor(target, dtype=torch.float))
            loss.backward()
            optim.step()
            optim.zero_grad()

            episodeloss = episodeloss + loss.item()

            for _ in range(9):
                simulator.step(actions[j])

    print("Episode: {}; Mean Train Step Loss: {:.4e}".format(epoch, episodeloss / episodesteps))

    if epoch % 10 == 0:
        correct = 0
        total = 0
        for step in range(episodesteps):
            totalSteps, actions = simulator.wavePolicy()
            for j in range(totalSteps):
                data = simulator.step(actions[j])
                out = net(data)

                out = torch.argmax(out, dim=1)

                for i in range(len(actions[j])):
                    total = total + 1
                    if actions[j][i] == out[i].item():
                        correct = correct + 1

                for _ in range(9):
                    simulator.step(actions[j])

        print("Episode: {}; Test Accuracy: {:.4e}".format(epoch, correct / total))

torch.save(net.state_dict(), "saved_models/imitation_gnn_v0.pt")