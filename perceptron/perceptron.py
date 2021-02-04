import numpy as np


class Perceptron(object):
    """Implements a single layer Perceptron """

    def __init__(self, input_size, epochs=100, lr=1):
        # initialize the weights vector parameter, adding one for bias
        self.W = np.zeros(input_size + 1)

        # Each time through the training data applying the learning rule is
        # called an Epoch
        self.epochs = epochs

        # Learning rate --this is a hyperparameter, it is not learned via the
        # training set but is supplied.
        self.lr = lr

    @staticmethod
    def sigma(x):
        """This is the activation function. """
        return 1 if x >= 0 else 0

    def train(self, X, d):
        """Iterate through the training data (X) and score the perceptrons
        performance via the labels (d) and adjust weights accordingly."""
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                y = self.predict(X[i])
                e = d[i] - y
                self.W = self.W + self.lr * e * np.insert(X[i], 0, 1)

    def predict(self, x):
        """y = sigma(z) """
        # add the bias term
        x = np.insert(x, 0, 1)
        # W^T * x (inner product)
        z = self.W.T.dot(x)
        # apply activation function
        a = self.sigma(z)
        # return predicted label
        return a

    def score(self, X, y):
        misclassified_count = 0
        for x_i, target in zip(X, y):
            output = self.predict(x_i)
            if target != output:
                misclassified_count += 1
        total_data_count = len(X)
        return (total_data_count - misclassified_count) / total_data_count


