# losses.py

import numpy as np


def to_numpy(inputs):
    if isinstance(inputs, np.array): inputs = np.array(inputs)
    return inputs

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
        preds = model.predict(inputs)

        # convert preds and labels to np.array
        preds, labels = to_numpy(preds), to_numpy(labels)

        return 2 * np.average(preds - labels) * inputs

    def __call__(self, preds, labels):
        # convert preds and labels to np.array
        preds, labels = to_numpy(preds), to_numpy(labels)

        return np.average(np.square(preds - labels))


class CrossEntropy(Losses):
    def __init__(self):
        supeer(CrossEntropy, self).__init__()

    def gradient(self, model, labels, inputs):
        # make preds
        preds = model(inputs)

        # convert preds and labels to np.array
        preds, labels = to_numpy(preds), to_numpy(labels)

        # gradient of cross-entropy

        # gradient of softmax

        return None

    def __call__(self, preds, labels):
        # Args:
        #   - preds: np.array
        #       Prediction matrix of shape [batch, n_class]
        #   - labels: np.array
        #       One-hot encoding label matrix of shape [batch, n_class]

        # convert preds and labels to np.array
        preds, labels = to_numpy(preds), to_numpy(labels)

        return -1 * np.sum(np.log(preds) * preds)