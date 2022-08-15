import numpy as np
import pymunk
import math
from statistics import mean

import DPR_ParticleRobot as DPR_ParticleRobot
from DPR_ParticleRobot import ParticleRobot


class SuperCircularBot:
    '''A collection of particle robots that can be observed and controleld in a centralized manner'''
    def __init__(self, numBots, pos, goal, botType=ParticleRobot,
                 continuousAction=False, deadIxs=[]):
        dims = math.ceil(math.sqrt(numBots))

        def posFunc(centerPos, ix):
            BOT_DIAMETER = 2 * DPR_ParticleRobot.BOT_RADIUS + DPR_ParticleRobot.PADDLE_WIDTH + DPR_ParticleRobot.PADDLE_LENGTH
            xc, yc = centerPos
            x = ((ix % dims) - (dims / 2) + 0.5) * BOT_DIAMETER + xc
            y = ((ix // dims) - (dims / 2) + 0.5) * BOT_DIAMETER + yc
            return (x, y)

        particleRobots = []
        for i in range(numBots):
            isDead = (i in deadIxs)
            bot = botType(posFunc(pos, i), goal, i, continuousAction=continuousAction, dead=isDead)
            particleRobots.append(bot)

        self.numBots = numBots
        self.particles = particleRobots

        self.prevCOM = pos
        self.currCOM = pos

        self.start = pos
        self.goal = goal

    def observeSelf(self):
        '''
        Superagent observes positions and velocities of individual particles
        bot position transformed to goal reference frame

        :return: np array containing x y components of position and velocity and angle for each particle
        '''
        self.updateCOM()
        obs = []
        for bot in self.particles:
            obs.append(bot.observeSelf())
        return np.array(obs)


    def actionAll(self, action):
        '''
        Superagent controls each particle robot - single agent system with multi-binary action space

        :param action: self.numBots sized array corresponding to action for each particle (binary)
        :return: list with the returns of each action from CircularBot action
        '''
        results = []
        for i in range(self.numBots):
            result = self.particles[i].act(action[i])
            results.append(result)
        return results

    def getCOM(self):
        '''
        Calculates center of mass of the system of particle robots - assumes every particle robot weighs the same

        :return: vector coordinates of center of mass
        '''
        xs = []
        ys = []
        for bot in self.particles:
            x, y = bot.shape.body.position
            xs.append(x)
            ys.append(y)
        return pymunk.vec2d.Vec2d(mean(xs), mean(ys))

    def updateCOM(self):
        '''
        Updates superagent center of mass

        :return:
        '''
        self.prevCOM = self.currCOM
        self.currCOM = self.getCOM()
