# model.py
import numpy as np


class LogisticRegressor:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def sigmoid(self, inputs):
        return 1 / (1 + np.exp(-inputs))

    def fit(self, inputs, labels):
        # initializes weights and bias
        self.coef_ = np.zeros(shape=(1, inputs.shape[-1]))
        self.intercept_ = np.zeros(shape=(1))

    def predict(self, inputs):
        # linear output
        outputs = np.dot(inputs, self.coef_.T) + self.intercept_

        # sigmoid
        return self.sigmoid(outputs)[..., -1]

    def binary_predict(self, inputs):
        binary_fn = lambda x: 1 if x > 0.5 else 0
        preds = self.predict(inputs)
        return np.array([binary_fn(x) for x in preds])