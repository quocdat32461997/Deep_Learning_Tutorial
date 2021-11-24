import torch
import numpy as np
from models import *
from env import *

discount_factor = 0.7
lr =

def to_device(inputs)
    if torch.cuda.is_availablle()
        inputs = inputsa.to('cuda')
    return inputs

def reinforce():
    # initialize env
    env = Env()

    # initialize model
    model = Reinforce(num_inputs=2,
                      nun_actions=4,
                      hidden_size=64)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr)
    num_episode = 1000
    num_step = 1000

    # training
    for episode in range(num_episode):
        # initial state
        state = env.reset()

        rewards, log_probs, total_reward = [], [], 0

        # take step
        for t in range(num_step):
            # predict action probs
            action, action_prob = model(state)

            state, reward, done = env.step(action)

            rewards.append(reward)
            total_reward += reward
            log_probs.append(action_prob)

            if done:
                break

        # compute loss
        R, loss = 0, []
        for i, r in enumerate(rewards[::-1]):
            R = r + discount_factor * R
            loss.insert(0, R)
        loss = torch.tensor(loss)
        loss *= -1 * torch.tensor(log_probs)
        loss = loss.sum()

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # display result
        if episode % 5 == 0:
            print('Episode {}\tLast reward: {:.2f}\t'.format(
                episode, total_reward,))
    pass

if __name__ == '__main__':
    reinforce()

    actor_critic()

