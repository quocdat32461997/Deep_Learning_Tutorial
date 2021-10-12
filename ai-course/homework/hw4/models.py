# schedulers.py

import numpy as np


class Scheduler(object):
    def __init__(self, learning_rate):
        self.initial_learning_rate = learning_rate
        pass

    def get_lr(self, **kwargs):
        return self.initial_learning_rate


class AdaGradScheduler(Scheduler):
    def __init__(self, learning_rate):
        super(AdaGrad, self).__init__(learning_rate=learning_rate)
        self.r = 1e-8
        self.accumulated_gradients = {'weight': 0, 'bias': 0}
        pass

    def update_lr(self, gradients, **kwargs):
        # square and accumulate gradients
        for k, v in gradients.items():
            self.accumulated_gradients[k] += np.square(v)

        return {k: self.initial_learning_rate / np.sqrt(v + self.r) for k, v in self.accumulated_gradients.items()}

    def get_lr(self, gradients, **kwargs):
        return self.update_lr(gradients)


class AdamScheduler(Scheduler):
    def __init__(self, learning_rate):
        super(AdamScheduler, self).__init__(learning_rate=learning_rate)
        self.r = 1e-8

    def udpate_lr(self, gradients, **kwargs):
        return None

    def get_lr(self, gradients, **kwargs):
        return None