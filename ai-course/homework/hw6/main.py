# main.py

import argparse
import numpy as np
from models import *


def main(args):
    # world params
    space, reward, discount, prob = 5, -5, 0.8, 1/25
    grid = np.zeros([space, space])
    grid[0,0], grid[-1,-1] = 1, 1 # mark goals

    # hyperparameters
    num_iter, lr = 5, 0.01

    # initialize ValueIteration algo
    model = ValueIter(space=space, reward=reward,
                  discount=discount, prob=prob)
    q_table = model.forward(num_iter=num_iter)

    # Deep Q Learning
    model = DeepQNetwork(num_feature=1, num_action=4,
                         loss_fn=torch.nn.MSELoss())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for iter in range(num_iter):


if __name__ == '__main__':
    # initialize parser
    parser = argparse.ArgumentParser()

    main(parser.parse_args())