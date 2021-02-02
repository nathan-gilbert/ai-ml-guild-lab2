import numpy as np


class MultiLayerPerceptron:
    @staticmethod
    def init_parameters(n_features, n_neurons, n_output):
        """generate initial parameters sampled from an uniform distribution
            Args:
                n_features (int): number of feature vectors
                n_neurons (int): number of neurons in hidden layer
                n_output (int): number of output neurons

            Returns:
                parameters dictionary:
                    W1: weight matrix, shape = [n_features, n_neurons]
                    b1: bias vector, shape = [1, n_neurons]
                    W2: weight matrix, shape = [n_neurons, n_output]
                    b2: bias vector, shape = [1, n_output]
        """
        np.random.seed(100)  # for reproducibility
        W1 = np.random.uniform(size=(n_features, n_neurons))
        b1 = np.random.uniform(size=(1, n_neurons))
        W2 = np.random.uniform(size=(n_neurons, n_output))
        b2 = np.random.uniform(size=(1, n_output))

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
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
            S: neuron activation
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

    def predict(self, X, W1, W2, b1, b2):
        """computes predictions with learned parameters

        Args:
            X: matrix of features
            W1: weight matrix for the first layer
            W2: weight matrix for the second layer
            b1: bias vector for the first layer
            b2: bias vector for the second layer

        Returns:
            d: vector of predicted values
        """
        Z1 = self.linear(W1, X, b1)
        S1 = self.sigmoid(Z1)
        Z2 = self.linear(W2, S1, b2)
        S2 = self.sigmoid(Z2)

        return np.where(S2 >= 0.5, 1, 0)

    def fit(self, X, y, n_features=2, n_neurons=3, n_output=1, iterations=10,
            eta=0.001):
        """Multi-layer perceptron trained with backpropagation

        Args:
            X: matrix of features
            y: vector of expected values
            n_features (int): number of feature vectors
            n_neurons (int): number of neurons in hidden layer
            n_output (int): number of output neurons
            iterations (int): number of iterations over the training set
            eta (float): learning rate

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

        for _ in range(iterations):
            # Forward-propagation
            Z1 = self.linear(param['W1'], X, param['b1'])
            S1 = self.sigmoid(Z1)
            Z2 = self.linear(param['W2'], S1, param['b2'])
            S2 = self.sigmoid(Z2)

            # Error computation
            error = self.cost(S2, y)
            errors.append(error)

            # Back propagation

            # update output weights
            delta2 = (S2 - y) * S2 * (1 - S2)
            W2_gradients = S1.T @ delta2
            param["W2"] = param["W2"] - W2_gradients * eta

            # update output bias
            param["b2"] = param["b2"] - np.sum(delta2, axis=0, keepdims=True) * eta

            # update hidden weights
            delta1 = (delta2 @ param["W2"].T) * S1 * (1 - S1)
            W1_gradients = X.T @ delta1
            param["W1"] = param["W1"] - W1_gradients * eta

            # update hidden bias
            param["b1"] = param["b1"] - np.sum(delta1, axis=0, keepdims=True) * eta

        return errors, param