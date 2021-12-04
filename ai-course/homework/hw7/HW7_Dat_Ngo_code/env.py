import numpy as np
import torch


class Env:
    ACTIONS = {
        0: torch.tensor([-1, 0]),  # up
        1: torch.tensor([0, 1]),  # right
        2: torch.tensor([1, 0]),  # down
        3: torch.tensor([0, -1])  # left
    }

    GOAL = 4
    CLIFF = 5

    # rewards for each status
    REWARDS = -5  # normal status

    def __init__(self):
        self.states = torch.ones([6, 10]) * -1
        self.states[-1, -1] = Env.GOAL  # goal
        self.states[-1, 1:9] = Env.CLIFF  # red cliff

        # initial state at [0,0]
        self.current_state = torch.tensor([5, 0])

        # policy table
        self.policy_table = self.states.clone()

    def reset(self):
        self.current_state = torch.tensor([5, 0])
        return self.current_state

    def step(self, action):
        _action = action # copy action id
        action = Env.ACTIONS[action]
        done = False

        # get next_state state given action
        next_state = self.current_state + action
        #print(self.current_state, next_state, self.states.size())

        # if next_state crosses border, do not update current state and return rewards -100
        if next_state[0] < 0 or next_state[0] >= self.states.size()[0] or next_state[-1] \
                < 0 or next_state[-1] >= self.states.size()[-1]:
            return self.current_state, Env.REWARDS, done

        # with-in grid
        if self.states[next_state[0], next_state[-1]] in [Env.GOAL, Env.CLIFF]: # final state
            done = True

        # update policy table w/ action
        #self.policy_table[self.current_state[0], self.current_state[-1]] = _action

        # update current state
        self.current_state = next_state

        return next_state, Env.REWARDS, done
