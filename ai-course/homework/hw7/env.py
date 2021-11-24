import numpy as np
import torch


class Env:
    ACTIONS = {
        0: torch.tensor([-1, 0]),  # up
        1: torch.tensor([0, 1]),  # right
        2: torch.tensor([1, 0]),  # down
        3: torch.tensor([0, -1])  # left
    }

    GOAL = 3
    CLIFF = 4

    # rewards for each status
    PEN_REWARDS = -100  # either cliff or out-of-edge
    REWARDS = -5  # normal status
    GOAL_REWARDS = 100

    def __init__(self):
        self.states = torch.ones([6, 10])
        self.states[-1, -1] = Env.GOAL  # goal
        self.stats[-1, 2:10] = Env.CLIFF  # red cliff

        # initial state at [0,0]
        self.current_state = torch.tensor([0, 0])

    def reset(self):
        self.current_state = torch.tensor([0, 0])
        return self.current_state

    def step(self, action):
        action = Env.ACTIONS[action]
        done = False
        rewards = Env.REWARDS

        # get next_state state given action
        next_state = self.current_state + action

        # if next_state crosses border, do not update current state and return rewards -100
        if 0 <= next__state[0] < self.states.size()[0] and 0 <= nedt_state[-1] < self.states.size()[-1]:
            return self.current_state, Env.PEN_REWARDS, done

        # with-in grid
        if self.states[next_state[0], next_state[-1]] == Env.GOAL: # final state
            done = True
            rewards = Env.GOAL_REWARDS
        elif self.states[next_state[0], next_state[-1]] == Env.CLIFF: # cliff state
            done = True
            rewards = Env.PEN_REWARDS

        return next_state, rewards, done
