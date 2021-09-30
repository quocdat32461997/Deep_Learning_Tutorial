# optimizers.py

from losses import *


class Optimizers(object):
    def __init__(self):
        pass

    def compute_gradient(self, gradients):
        pass

    def backward(self, preds, labels):
        pass


class SGD(Optimizers):
    def __init__(self, model, loss_fn, learning_rate=0.01, momentum=0.0, nesterov=False):
        # Args:
        #   - learning_rate: float
        #   - momentum: float
        #       Momentum for SGD
        #   - nesterov: bool
        #       Flag to usee Nesterov momentum. If True, momentum must not be 0.0
        super(SGD, self).__init__()

        self.loss_fn = Losses.get_loss_fn(name) if isinstance(loss_fn, str) else loss_fn
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov

        self.loss = None
        self.model = model
        self.gradients = 0

    def compute_gradient(self, labels, inputs):
        # compute momentum-enhanced gradients in previous step
        self.gradients *= self.momentum

        # if nesterov momentum, append prev-step gradients to model weights
        if self.nesterov:
            self.model.weights += self.gradients

        # compute gradients at current step
        self.gradients -= self.learning_rate * self.loss_fn.gradient(self.model, labels, inputs)

        # update weights
        self.model.weights += self.gradients

    def backward(self, labels, inputs):
        # compute loss
        self.loss = self.loss_fn(self.model(inputs), labels)

        # compute gradients
        self.compute_gradient(preds, labels, inputs)