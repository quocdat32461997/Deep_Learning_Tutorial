import torch
import numpy as np
from models import *
from env import *

discount_factor = 0.7
lr = 0.01
num_episode, num_step = 1000, 1000

torch.manual_seed(1000)

def to_device(inputs):
    if torch.cuda.is_availablle():
        inputs = inputsa.to('cuda')
    return inputs


def reinforce():
    # initialize env
    env = Env()

    # initialize model
    model = Reinforce(num_inputs=2,
                      nun_actions=4,
                      hidden_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training
    for episode in range(num_episode):
        # initial state
        state = env.reset()

        rewards, log_probs, total_reward = [], [], 0

        # take step
        for t in range(num_step):
            # predict action probs
            action, action_prob = model.get_action(state)
            state, reward, done = env.step(action)

            # update rewards
            model.rewards.append(reward)
            total_reward += reward
            model.log_probs.append(action_prob)

            if done:
                break

        # compute loss
        R, loss, returns = 0, [], []
        for i, r in enumerate(rewards[::-1]):
            R = r + discount_factor * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        for log_prob, R in zip(log_probs, returns):
            loss.append(-log_prob * R)

        # backprop
        optimizer.zero_grad()
        loss = torch.stack(loss).sum()
        loss.backward()
        optimizer.step()

        # display result
        if episode % 5 == 0:
            print('Episode {}\tLast reward: {:.2f}\t'.format(
                episode, total_reward))

        del model.rewards[:]
        del model.log_probs[:]


def actor_critic():
    # initialize env
    env = Env()

    # initialize model
    model = ActorCritic(num_inputs=2,
                      nun_actions=4,
                      hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training
    for episode in range(num_episode):
        # initial state
        state = env.reset()
        state_values, total_reward = [], 0

        # take step
        for t in range(num_step):
            # take action
            action, action_prob, state_value = model.get_action(state)

            # get next_state and reward
            state, reward, done = env.step(action)

            # update rewards
            model.rewards.append(reward)
            model.log_probs.append(action_prob)
            total_reward += reward
            state_values.append(state_value)

            if done:
                break

        # compute loss
        R, policy_loss, value_loss, returns = 0, [], [], []
        for i, r in enumerate(model.rewards[::-1]):
            R = r + discount_factor * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        for log_prob, value, R in zip(model.log_probs, state_values, returns):
            # calculate actor (policy) loss
            policy_loss.append(-log_prob * R)

            # calculate critic (value) loss using MSELoss
            value_loss.append(torch.nn.MSELoss()(value, torch.tensor([R])))

        # backprop
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + \
               torch.stack(value_loss).sum()
        loss.backward()
        optimizer.step()

        # display result
        #if episode % 5 == 0:
        print('Episode {}\tLast reward: {:.2f}\tLoss: {:.2f}\tSteps: {}'.format(episode, total_reward, loss, t))

        del model.rewards[:]
        del model.log_probs[:]
    pass


if __name__ == '__main__':
    #reinforce()

    actor_critic()
