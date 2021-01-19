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
    and_gate_labels = np.array([0, 0, 0, 1])
    #or_gate_labels = np.array([0, 1, 1, 1])

    perceptron = Perceptron(input_size=2)
    perceptron.train(X, and_gate_labels)
    print("Weight vector: ", perceptron.W)
