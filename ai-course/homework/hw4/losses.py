# losses.py

import numpy as np


class Losses(object):
    def __init__(self):
        self.gradient = None

    def gradient(self):
        return None

    def __call__(self):
        return None

    @staticmethod
    def get_loss_fn(name):
        if name == 'mse' or name == 'MSE':
            return MSE()
        elif name == 'crossentropy' or name = 'CrossEntropy':
            return CrossEntropy()
        else:
            raise Exception('{} is not a valid loss function.'.format(name))


class MSE(Losses):
    def __init__(self):
        super(MSE, self).__init__()

    def gradient(self, model, labels, inputs):
        # make preds
        preds = model(inputs)

        # convert preds and labels to np.array
        if isinstance(preds, np.array): preds = np.array(preds)
        if isinstance(labels, np.array): labels = np.array(labels)

        return 2 * np.average(preds - labels) * inputs

    def __call__(self, preds, labels):
        # convert preds and labels to np.array
        if isinstance(preds, np.array): preds = np.array(preds)
        if isinstance(labels, np.array): labels = np.array(labels)

        return np.average(np.square(preds - labels))


class CrossEntropy(Losses):
    def __init__(self):
        supeer(CrossEntropy, self).__init__()

    def gradient(self, preds, labels):
        return None

    def __call__(self, preds, labels):
        return 0