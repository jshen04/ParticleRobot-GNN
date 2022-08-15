import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import os.path as osp

from sim import GraphSimulator

class ParticleRobotDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.episodes = 2000
        self.episodelength = 5000
        super(ParticleRobotDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return "not_implemented.pt"

    @property
    def processed_file_names(self):
        return f'data_{self.episodes*self.episodelength-1}.pt'

    def download(self):
        pass

    def process(self):
        numbots = 9
        simulator = GraphSimulator(num=numbots, start=(300, 300), goal=(0, 0), visualizer=None)

        idx = 0
        for epoch in range(self.episodes):
            simulator.setup()
            for step in range(self.episodelength):
                action = np.random.randint(2, size=numbots)
                data = simulator.step(action, append_action=True)

                torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
                idx += 1
                print(idx)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
