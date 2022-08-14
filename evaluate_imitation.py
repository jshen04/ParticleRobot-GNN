import torch
import numpy as np

from sim import Simulator
from vanilla_imitation import ImitationNet

numbots = 9
simulator = Simulator(num=numbots, start=(300, 300), goal=(0, 0), visualizer=None)

net = ImitationNet(features=6)
net.load_state_dict(torch.load("./models/imitation_v0.pt"))

episodesteps = 2500
for step in range(episodesteps):
    totalSteps, actions = simulator.wavePolicy()
    for j in range(totalSteps):
        data = simulator.step(actions[j])
        out = net(torch.tensor(data, dtype=torch.float))

        out = torch.argmax(out, dim=1)
        print("Truth: ", actions[j])
        print("Prediction: ", out)

        for _ in range(9):
            simulator.step(actions[j])
