# models.py
import numpy as np
import torch

class ValueIter(object):
    DIRECTIONS = {
        0: [-1, 0], # up
        1: [0, 1], # right
        2: [1, 0], # bottom
        3: [0, -1] # lleft
    }

    def __init__(self, space, reward, discount, prob):
        # initialize Q-table
        self.q_table = np.ones([space, space, 4]) * -1
        self.q_table[0, 0], self.q_table[-1, -1] = 0, 0

        # initialize converged table store converged states or goals
        self.converged = self.q_table.copy()

        # set additional parameters
        self.reward = reward
        self.discount = discount
        self.prob = prob

    def forward(self, num_iter, delta=0.1):
        for iter in range(num_iter):
            # each state
            _q_table = self.q_table.copy()
            for x in range(self.q_table.shape[0]): # vertical
                for y in range(self.q_table.shape[1]): # horizontal
                    for m in range(4): # each action/direction
                        if self.converged[x, y, m] != 0: # either converged or goal state
                            # compute the max cumulative reward
                            _reward = self.max_reward(x, y, m)

                            # update reward
                            _q_table[x, y, m] = _reward

            # check converged
            if abs((self.q_table - _q_table).sum()) <= delta:
                return None # stop once converged

            # update q_table if not converged
            self.q_table = _q_table.copy()

        return None

    def max_reward(self, x, y, m):
        # get next state
        _x = x + ValueIter.DIRECTIONS[m][0]
        _y = y + ValueIter.DIRECTIONS[m][-1]

        def _max_reward(i, j):
            # Function to get max expected value of the next state
            return max(self.q_table[i, j])

        reward = 0
        # vertical: up then bottom
        if 0 <= _x < self.q_table.shape[0] and 0 <= _y < self.q_table.shape[1]:
            reward = (self.reward + self.discount * _max_reward(_x, _y)) / 25

        return reward


class DeepQNetwork(torch.nn.Module):
    def __init__(self, input_feature, num_action, loss_fn):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_feature, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, num_action) # 4 actions
        )
        self.loss_fn = loss_fn

    def forward(self, inputs):
        # extract inputs and labels
        inputs, labels = inputs

        # predict
        outputs = self.model(inputs)

        # loss
        loss = self.loss_fn(outputs, labels)

        return loss
