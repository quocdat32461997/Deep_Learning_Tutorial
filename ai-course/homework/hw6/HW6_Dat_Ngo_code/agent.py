# agent.py

import numpy as np


class Agent(object):
    def __init__(self, space, reward, discount, prob):

        # initialize Q-table
        self.q_table = np.ones([space, space, 4])* -1
        self.q_table[0, 0] = 0
        self.q_table[-1, 0] = 0

        # set additional parameters
        self.reward = reward
        self.discount = discount
        self.prob = prob

    def move(self):
        pass
