# optimizers.py

import copy
import numpy as np

from losses import *


class Optimizers(object):
    def __init__(self):
        self.loss = 0
        pass

    def compute_gradient(self):
        pass

    def backward(self):
        pass


class GradientDescent(Optimizers):
    def __init__(self,
                 model,
                 loss_fn,
                 learning_rate=0.01,
                 momentum=0.0,
                 nesterov=False):
        # Args:
        #   - learning_rate: float
        #   - momentum: float
        #       Momentum for SGD
        #   - nesterov: bool
        #       Flag to usee Nesterov momentum. If True, momentum must not be 0.0
        super(GradientDescent, self).__init__()

        self.loss_fn = Losses.get_loss_fn(loss_fn) if isinstance(loss_fn, str) else loss_fn()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov

        self.model = copy.copy(model)
        self.model.coef_ = np.zeros_like(self.model.coef_)
        self.model.intercept_ = np.expand_dims(np.zeros_like(self.model.intercept_), axis=0)

        # initialize features
        self.features = np.concatenate((self.model.coef_, self.model.intercept_), axis=-1)
        self.gradients = self.features.copy()

    def compute_gradient(self, preds, inputs, labels, **kwargs):
        # compute momentum-enhanced gradients in previous step
        self.gradients *= self.momentum

        # if nesterov momentum, append prev-step gradients to model weights
        if self.nesterov:
            self.model.coef_ += self.gradients[..., :-1]  # ['weight']
            self.model.intercept_ += self.gradients[..., -1]  # ['bias']

        # compute gradients at current step
        gradients = self.loss_fn.gradient(self.model, inputs, labels)
        self.gradients -= self.learning_rate * gradients

        # update weights and bias
        self.model.coef_ = self.features[..., :-1] = self.features[..., :-1] + self.gradients[..., :-1]
        self.model.intercept_ = self.features[..., -1] = self.features[..., -1] + self.gradients[..., -1]

    def backward(self, preds, inputs, labels, **kwargs):
        # compute loss
        self.loss = self.loss_fn(self.model, inputs, labels)

        # compute gradients
        self.compute_gradient(preds, inputs, labels)

class SGD(GradientDescent):
    def __init__(self,
                 model,
                 loss_fn,
                 learning_rate=0.01,
                 momentum=0.0,
                 nesterov=False):
        super(SGD, self).__init__(model=model,
                                  loss_fn=loss_fn,
                                  learning_rate=learning_rate,
                                  momentum=momentum,
                                  nesterov=nesterov)
        pass


class AdaGrad(Optimizers):
    def __init__(self,
                 model,
                 loss_fn,
                 learning_rate=0.01):
        # Args:
        #   - model: scikit-learn model
        #   - loss_fn: loss function
        #   - learning_rate: float
        super(AdaGrad, self).__init__()
        self.loss_fn = Losses.get_loss_fn(loss_fn) if isinstance(loss_fn, str) else loss_fn()
        self.learning_rate = learning_rate

        self.model = copy.copy(model)
        self.model.coef_ = np.zeros_like(self.model.coef_)
        self.model.intercept_ = np.expand_dims(np.zeros_like(self.model.intercept_), axis=0)

        # initialize gradients
        self.gradients = np.concatenate((self.model.coef_, self.model.intercept_), axis=-1)
        self.accumulated_gradients = self.gradients.copy()
        self.r = 1e-8

    def _get_lr(self):
        # Function to produce learning-rates corresponding to past gradients
        # square and accumulate gradients
        self.accumulated_gradients += np.square(self.gradients)

        return self.learning_rate / np.sqrt(self.accumulated_gradients + self.r)

    def compute_gradient(self, preds, inputs, labels, **kwargs):
        # compute gradients at current step
        self.gradients = self.loss_fn.gradient(self.model, inputs, labels)

        # update lr
        lr = self._get_lr()

        # update weights and bias
        self.model.coef_ -= lr[..., :-1] * self.gradients[..., :-1]
        self.model.intercept_ -= lr[..., -1] * self.gradients[..., -2:-1]

    def backward(self, preds, inputs, labels, **kwargs):
        # compute loss
        self.loss = self.loss_fn(self.model, inputs, labels)

        # compute gradients
        self.compute_gradient(preds, inputs, labels)


class Adam(Optimizers):
    def __init__(self,
                 model,
                 loss_fn,
                 learning_rate=0.01,
                 beta_1=0.9,
                 beta_2=0.999):
        super(Adam, self).__init__()

        self.loss_fn = Losses.get_loss_fn(loss_fn) if isinstance(loss_fn, str) else loss_fn()
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.model = copy.copy(model)
        self.model.coef_ = np.zeros_like(self.model.coef_)
        self.model.intercept_ = np.expand_dims(np.zeros_like(self.model.intercept_), axis=0)

        # initialize features
        self.gradients = np.concatenate((self.model.coef_, self.model.intercept_), axis=-1)
        self.beta_1_step, self.beta_2_step = 0, 0
        self.r = 1e-8

    def _get_lr(self, beta_2_bias):
        return self.learning_rate / np.sqrt(beta_2_bias + self.r)

    def compute_gradient(self, preds, inputs, labels, iter, **kwargs):
        # compute gradients at current step
        self.gradients = self.loss_fn.gradient(self.model, inputs, labels)

        # compute beta_1_step and beta_2_step
        self.beta_1_step = self.beta_1 * self.beta_1_step + (1 - self.beta_1) * self.gradients
        self.beta_2_step = self.beta_2 * self.beta_2_step + (1 - self.beta_2) * np.square(self.gradients)

        # compute bias ccrrection
        bias_1 = self.beta_1_step / (1 - pow(self.beta_1, iter + 1))
        bias_2 = self.beta_2_step / (1 - pow(self.beta_2, iter + 1))

        # get learning rate
        lr = self._get_lr(bias_2)

        # update weights and bias
        self.model.coef_ -= lr[..., :-1] * bias_1[..., :-1]
        self.model.intercept_ -= lr[..., -1] * bias_1[..., -1]

    def backward(self, preds, inputs, labels, **kwargs):
        # compute loss
        self.loss = self.loss_fn(self.model, inputs, labels)

        # compute gradients
        self.compute_gradient(preds, inputs, labels, iter=kwargs['iter'])
