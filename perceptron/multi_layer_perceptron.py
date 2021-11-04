import numpy as np


class MultiLayerPerceptron:
    """
    Multilayer Perceptron with one hidden layer
    swiped from https://pabloinsente.github.io/the-multilayer-perceptron
    """

    @staticmethod
    def init_parameters(n_features, n_neurons, n_output):
        """generate initial parameters sampled from an uniform distribution
            Args:
                n_features (int): number of feature vectors
                n_neurons (int): number of neurons in hidden layer
                n_output (int): number of output neurons

            Returns:
                parameters dictionary:
                    w1: weight matrix, shape = [n_features, n_neurons]
                    b1: bias vector, shape = [1, n_neurons]
                    W2: weight matrix, shape = [n_neurons, n_output]
                    b2: bias vector, shape = [1, n_output]
        """

        # Backpropagation is very sensitive to initialization values
        np.random.seed(100)  # for reproducibility

        # weights & bias for input layer
        w1 = np.random.uniform(size=(n_features, n_neurons))
        b1 = np.random.uniform(size=(1, n_neurons))
        # weights & bias for hidden layer
        w2 = np.random.uniform(size=(n_neurons, n_output))
        b2 = np.random.uniform(size=(1, n_output))

        parameters = {"W1": w1,
                      "b1": b1,
                      "W2": w2,
                      "b2": b2}
        return parameters

    @staticmethod
    def linear(W, X, b):
        """computes net input as dot product
            Args:
                W: weight matrix
                X: matrix of features
                b: vector of biases

            Returns: weighted sum of features (Z)
        """
        # @ operator does matrix multiplication
        return (X @ W) + b

    @staticmethod
    def sigmoid(Z):
        """computes sigmoid activation element wise

        Args:
            Z: weighted sum of features

        Returns:
            A: neuron activation
        """
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def cost(A, y):
        """computes squared error

        Args:
            A: neuron activation
            y: vector of expected values

        Returns:
            E: total squared error"""
        return (np.mean(np.power(A - y, 2))) / 2

    @staticmethod
    def threshold(a):
        """
        Output function that actually does the labelling / classification
        """
        return np.where(a >= 0.5, 1, 0)

    def predict(self, X, w1, w2, b1, b2):
        """predicts label from learned parameters

        This is basically forward propagation but stops after getting
        a label.

        Args:
            X: matrix of features vectors
            w1: weight matrix for the first layer
            w2: weight matrix for the second layer
            b1: bias vector for the first layer
            b2: bias vector for the second layer

        Returns:
            d: vector of predicted values
        """
        # input layer
        z1 = self.linear(w1, X, b1)
        a1 = self.sigmoid(z1)
        # hidden layer
        z2 = self.linear(w2, a1, b2)
        a2 = self.sigmoid(z2)

        return self.threshold(a2)

    def fit(self, X, y, n_features=2, n_neurons=3, n_output=1, epochs=10,
            lr=0.001):
        """Multi-layer perceptron trained with backpropagation

        Args:
            X: matrix of features
            y: vector of expected values
            n_features (int): number of feature vectors
            n_neurons (int): number of neurons in hidden layer
            n_output (int): number of output neurons
            epochs (int): number of iterations over the training set
            lr (float): learning rate

        Returns:
            errors (list): list of errors over iterations
            param (dic): dictionary of learned parameters
        """

        # Initialize parameters
        param = self.init_parameters(n_features=n_features,
                                     n_neurons=n_neurons,
                                     n_output=n_output)

        # storage errors after each iteration
        errors = []

        for _ in range(epochs):
            # Let's do  Forward propagation again!
            # input layer
            z1 = self.linear(param['W1'], X, param['b1'])
            a1 = self.sigmoid(z1)

            # And again for the hidden layer
            # hidden layer
            z2 = self.linear(param['W2'], a1, param['b2'])
            a2 = self.sigmoid(z2)

            # Error computation
            error = self.cost(a2, y)
            errors.append(error)

            # Back propagation
            # update output weights
            delta2 = (a2 - y) * a2 * (1 - a2)

            # matrix math is saving us
            w2_gradients = a1.T @ delta2
            param['W2'] = param['W2'] - w2_gradients * lr

            # update output bias
            param['b2'] = param['b2'] - np.sum(delta2, axis=0, keepdims=True) * lr

            # update hidden weights
            delta1 = (delta2 @ param['W2'].T) * a1 * (1 - a1)
            w1_gradients = X.T @ delta1
            param['W1'] = param['W1'] - w1_gradients * lr

            # update hidden bias
            param['b1'] = param['b1'] - np.sum(delta1, axis=0, keepdims=True) * lr

        return errors, param
