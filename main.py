import numpy as np
from perceptron.perceptron import Perceptron


if __name__ == '__main__':
    # AND Gate
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    d = np.array([0, 0, 0, 1])

    perceptron = Perceptron(input_size=2)
    perceptron.train(X, d)
    print(perceptron.W)
