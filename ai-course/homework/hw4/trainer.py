# trainer.py

import numpy as np
from tqdm import tqdm
from optimizers import GradientDescent


class Trainer(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def _stochastic_train(self, data, iter):
        epoch_loss = []  # epoch loss
        for inputs, labels in tqdm(zip(*data)):
            # reshape to rank > 2
            if len(inputs.shape) < 2:
                inputs = np.reshape(inputs, [1] + list(inputs.shape))
                labels = np.reshape(labels, [1] + list(labels.shape))

            # make predictions
            preds = self.optimizer.model.predict(inputs)

            # update weights by optimizer
            self.optimizer.backward(preds, inputs, labels, iter=iter)

            # update losses
            epoch_loss.append(self.optimizer.loss)

        # log loss every 5 epochs
        # if iter % 5 == 0:
        epoch_loss = np.mean(epoch_loss)
        print('Iteration {}: loss = {}'.format(iter, epoch_loss))

    def _train(self, data, iter, **kwargs):
        # extract inputs and labels
        inputs, labels = data

        # make predictions
        preds = self.optimizer.model.predict(inputs)

        # update weights by optimizer
        self.optimizer.backward(preds, inputs, labels, iter=iter)

        print('Iteration {}: loss = {}'.format(iter, self.optimizer.loss))

    def train(self, data, num_iter):
        # Args
        #   - data: tuple(inputs, labels)
        #   - num_iter: ints
        #       Number of iterations

        # figure training pipeline
        if type(self.optimizer) == GradientDescent:
            train_pipeline = self._train
        else:
            train_pipeline = self._stochastic_train

        # training pipeline
        for iter in range(num_iter):
            train_pipeline(data=data, iter=iter)

    def get_model(self):
        return self.optimizer.model