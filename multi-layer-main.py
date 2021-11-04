import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from perceptron.multi_layer_perceptron import MultiLayerPerceptron


def score(X, Y, y):
    misclassified_count = 0
    for y_i, target in zip(Y, y):
        if y_i != target:
            misclassified_count += 1
    total_data_count = len(X)
    return (total_data_count - misclassified_count) / total_data_count


if __name__ == '__main__':
    # Binary Ops
    binary_ops = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    xor_labels = np.array([[0], [1], [1], [0]])

    perceptron = MultiLayerPerceptron()
    _, params = perceptron.fit(binary_ops,
                               xor_labels,
                               epochs=100,
                               lr=0.50)

    y_pred = perceptron.predict(binary_ops,
                                params["W1"],
                                params["W2"],
                                params["b1"],
                                params["b2"])
    print('Multi-layer perceptron accuracy: %.2f%%' % score(binary_ops,
                                                            xor_labels,
                                                            y_pred))
