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
    num_iter, lr = 20, 1e-3
    max_step = 100

    # initialize ValueIteration algo
    model = ValueIter(space=space, reward=reward,
                  discount=discount, prob=prob)

    print('Initial Q_value', model.q_table.reshape([25, 4]))

    # train by value-iter
    model.forward(num_iter=num_iter)

    # flatten q table
    q_table = model.q_table.reshape([25, 4])
    print('Q_table by Vallue-Itderation algo', q_table)


    # Deep Q Learning
    model = DeepQNetwork(input_feature=2, num_action=4,
                         loss_fn=torch.nn.MSELoss(),
                         space=space, reward=reward,
                         discount=discount, prob=prob)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                    gamma=0.9,
                                                    verbose=True)

    for iter in range(num_iter):
        # each state
        epoch_loss = 0
        for x in range(model.q_table.shape[0]):
            for y in range(model.q_table.shape[1]):
                if model.converged[x, y] != 0:
                    for t in range(max_step):
                        # zero gradient
                        optimizer.zero_grad()

                        # predict action
                        loss = model([x,y])

                        # backward
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.sum()
        scheduler.step()
        print('loss', epoch_loss)
    q_table = model.q_table.copy()
    model.eval()
    for x in range(model.q_table.shape[0]):
        for y in range(model.q_table.shape[1]):
            if model.converged[x, y] != 0:
                q_table[x,y] = model.evaluate([x,y]).detach().cpu().numpy()

    # flatten q_table
    q_table = q_table.reshape([25, 4])
    print('Q_table by DeepQNetwork', q_table)
    pass


if __name__ == '__main__':
    # initialize parser
    parser = argparse.ArgumentParser()

    main(parser.parse_args())