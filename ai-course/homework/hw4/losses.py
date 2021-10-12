# losses.py

import numpy as np


def to_numpy(inputs):
    if isinstance(inputs, np.ndarray): inputs = np.array(inputs)
    return inputs


class Losses(object):
    def __init__(self):
        pass

    def gradient(self):
        return None

    def __call__(self):
        return None

    @staticmethod
    def get_loss_fn(name):
        if name == 'mse' or name == 'MSE':
            return MSE()
        elif name == 'crossentropy' or name == 'CrossEntropy':
            return CrossEntropy()
        else:
            raise Exception('{} is not a valid loss function.'.format(name))


class MSE(Losses):
    def __init__(self):
        super(MSE, self).__init__()

    def gradient(self, model, inputs, labels):
        # make predictions
        preds = model.predict(inputs)

        # convert preds and labels to np.array
        preds, labels = to_numpy(preds), to_numpy(labels)

        # compute gradients
        gradients = preds - labels
        N = gradients.shape[0]

        gradients = {
            'weight': np.dot(inputs.T, gradients) * 2 / N,
            'bias': np.expand_dims(np.sum(gradients) * 2 / N, axis=0)
        }
        return np.concatenate(list(gradients.values()), axis=0)

    def __call__(self, model, inputs, labels):
        # make predictions
        preds = model.predict(inputs)

        # convert preds and labels to np.array
        preds, labels = to_numpy(preds), to_numpy(labels)

        return np.average(np.square(preds - labels))


class CrossEntropy(Losses):
    def __init__(self):
        self.epsilon = 1e-7 # to avoid zero to log
        super(CrossEntropy, self).__init__()

    def gradient(self, model, inputs, labels):
        # make predictions
        preds = model.binary_predict(inputs)

        # convert preds and labels to np.array
        preds, labels = to_numpy(preds), to_numpy(labels)

        # compute gradients
        gradients = preds - labels
        N = labels.shape[0]

        gradients = {
            'weight': np.dot(inputs.T, gradients) * 2 / N,
            'bias': np.expand_dims(np.sum(gradients) * 2 / N, axis=0)
        }
        return np.concatenate(list(gradients.values()), axis=0)

    def __call__(self, model, inputs, labels):
        # Args:
        #   - preds: np.array
        #       Prediction matrix of shape [batch, n_class]
        #   - labels: np.array
        #       One-hot encoding label matrix of shape [batch, n_class]

        # make predictions
        preds = model.predict(inputs)

        # convert preds and labels to np.array
        preds, labels = to_numpy(preds), to_numpy(labels)
        N = labels.shape[0]

        # cross entropy
        cross_entropy = np.dot(np.log(self.epsilon + preds).T, labels) + \
                        np.dot(np.log(1 - preds + self.epsilon).T, 1 - labels)
        return -1 * cross_entropy / N
