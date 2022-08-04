import pymunk
import pygame
import numpy as np
import math

import torch
import torch_geometric

import DPR_ParticleRobot
from DPR_ParticleRobot import ParticleRobot
from DPR_SuperAgent import SuperCircularBot
from DPR_World import World
from viz import Visualizer


class Simulator:
    def __init__(self, num, start, goal, visualizer=None):
        self.world = World(visualizer=visualizer)
        self.num = num
        self.start = start
        self.goal = goal
        self.superagent = SuperCircularBot(num, start, goal)
        self.steps = 0

    def setup(self, num=None, start=None, goal=None):
        self.world.removeAll()
        self.steps = 0
        if num is not None:
            self.num = num
        if start is not None:
            self.start = start
        if goal is not None:
            self.goal = goal
        self.superagent = SuperCircularBot(self.num, self.start, self.goal)
        self.world.addSuperAgent(self.superagent)

    def step(self, action):
        '''
        Take one step
        1. get observations
        2. simulates magnetic attraction
        3. takes actions
        4. run in pymunk
        5. reset magnets

        :param action: particle robot action
        :return: world observations
        '''
        self.steps += 1

        # 1. get observation
        obs = self.superagent.observeSelf()

        # 2. particles react/simulate magnetic attraction
        self.world.addMagnets()

        # 3. take actions
        self.superagent.actionAll(action)

        # 4. simulate
        for i in range(self.world.pymunk_steps_per_frame):
            self.world.space.step(self.world._dt)

        # 5. clear current magnets
        self.world.removeMagnets()

    def run_wave(self, timesteps):
        self.setup()
        for i in range(timesteps):
            totalSteps, actions = self.wavePolicy()
            for j in range(totalSteps):
                for k in range(10):
                    action = actions[j]
                    self.step(action)
                    if self.world.visualizer is not None and self.world.visualizer.viz(i, self.world) == False:
                            break
                # After action
                state = self.superagent.observeSelf()


    def wavePolicy(self):
        '''
        Hand-crafted baseline algorithm. Particle robots receive an index based to distance to the goal
        (closer distance, earlier index), and the index determines the particle robot's location in the
        expansion cycle.

        :return: Array of sequential actions
        '''
        buffer = 60
        cycleIx = []
        dists = []

        def lineThrough2Points(p, q):
            a = p[1] - q[1]
            b = q[0] - p[0]
            c = p[1] * q[0] - p[0] * q[1]
            return (a, b, -c)

        def distPoint2Line(p, line):
            if line[0] == 0 and line[1] == 0:
                return np.linalg.norm(p)
            return abs((line[0] * p[0] + line[1] * p[1] + line[2])) / (math.sqrt(line[0] * line[0] + line[1] * line[1]))

        cx, cy = self.superagent.getCOM()
        gx, gy = self.superagent.goal
        px = (((gy - cy)**2)/(gx-cx)) + gx
        py = cy
        p2 = pymunk.vec2d.Vec2d(px, py)
        perpLine = lineThrough2Points(self.superagent.goal, p2)
        for bot in self.superagent.particles:
            dists.append(distPoint2Line(bot.body.position, perpLine))

        lower = min(dists)
        for dist in dists:
            ix = ((dist - lower) // buffer) + 1
            cycleIx.append(int(ix))
        lastAction = max(cycleIx)
        actionArray = np.zeros((lastAction, self.superagent.numBots))
        for botIx, ix in enumerate(cycleIx):
            actionArray[ix - 1][botIx] = 1
        return lastAction, actionArray


class GraphSimulator(Simulator):
    def __init__(self, num, start, goal, visualizer=Visualizer()):
        super().__init__(num, start, goal, visualizer)

    def get_label(self):
        '''
        :return: 2 x n tensor s.t. [0, 1] means system is getting closer, [1, 0] means getting further
        '''
        labels = []
        for particle in self.superagent.particles:
            labels.append([particle.body.position[0] / 1000, particle.body.position[1] / 1000, particle.body.velocity[0] / 10, particle.body.velocity[1] / 10])
        return torch.tensor(labels, dtype=torch.float)

    def step(self, action):
        '''
        Take one step
        1. get observations
        2. simulates magnetic attraction
        3. takes actions
        4. run in pymunk
        5. reset magnets
        6. get observations

        :param action: particle robot action
        :return: world observations
        '''
        self.steps += 1

        # 1. get observation
        first_obs = self.superagent.observeSelf()

        # creating edge index
        edge_index = [[], []]
        for pair in self.world.generatePairs():
            edge_index[0].append(pair[0].botId)
            edge_index[1].append(pair[1].botId)

            edge_index[0].append(pair[1].botId)
            edge_index[1].append(pair[0].botId)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # 2. particles react/simulate magnetic attraction
        self.world.addMagnets()

        # 3. take actions
        self.superagent.actionAll(action)

        # combine action with current state
        node_features = torch.tensor(np.hstack((first_obs, np.array(action).reshape(-1, 1))), dtype=torch.float)

        # 4. simulate
        for i in range(self.world.pymunk_steps_per_frame):
            self.world.space.step(self.world._dt)

        # 5. clear current magnets
        self.world.removeMagnets()

        # 6. get observation
        # final_obs = torch.tensor(self.superagent.observeSelf(), dtype=torch.float)
        label = self.get_label()

        if self.world.visualizer is not None and self.world.visualizer.viz(self.steps, self.world) == False:
            return

        data = torch_geometric.data.Data(x=node_features, edge_index=edge_index, y=label)

        return data

