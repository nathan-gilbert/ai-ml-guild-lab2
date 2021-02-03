import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

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
                               epochs=5000,
                               lr=0.1)

    y_pred = perceptron.predict(binary_ops,
                                params["W1"],
                                params["W2"],
                                params["b1"],
                                params["b2"])
    print('Multi-layer perceptron accuracy: %.2f%%' % score(binary_ops,
                                                            xor_labels,
                                                            y_pred))

    # Now try some real data
    # Load the data set
    bc = datasets.load_breast_cancer()
    L = bc.data
    y = bc.target

    # Create training and test split
    X_train, X_test, y_train, y_test = train_test_split(L,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42,
                                                        stratify=y)

    _, params = perceptron.fit(X_train,
                               y_train,
                               n_features=30,
                               n_neurons=10,
                               n_output=1,
                               epochs=1000,
                               lr=0.25)
    y_train_pred = perceptron.predict(X_train, params["W1"], params["W2"], params["b1"], params["b2"])
    y_test_pred = perceptron.predict(X_test, params["W1"], params["W2"], params["b1"], params["b2"])

    print("Training set accuracy: ", score(X_train, y_train, y_train_pred[0]))
    print("Test set accuracy: ", score(X_test, y_test, y_test_pred[0]))
