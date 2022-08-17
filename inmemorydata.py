import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Dataset, Data
from torch_geometric.loader import DataLoader
import os.path as osp

from sim import GraphSimulator

class ParticleRobotDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, start_idx=0):
        self.episodes = 100
        self.episodelength = 5000
        self.start_idx = start_idx
        super(ParticleRobotDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "not_implemented.pt"

    @property
    def processed_file_names(self):
        return 'data.pt'
        # return f'episode_{self.episodes-1+self.start_idx}.pt'

    def download(self):
        pass

    def process(self):
        numbots = 9
        simulator = GraphSimulator(num=numbots, start=(300, 300), goal=(0, 0), visualizer=None)
        data_list = []

        for epoch in range(self.episodes):
            simulator.setup()
            for step in range(self.episodelength):
                action = np.random.randint(2, size=numbots)
                data = simulator.step(action, append_action=True)
                data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), osp.join(self.processed_dir, 'data.pt'))

dataset = ParticleRobotDataset(root="in_memory_data/")