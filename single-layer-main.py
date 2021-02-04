import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from perceptron.perceptron import Perceptron


def show_results(ops, weights):
    for op in ops:
        final_result = Perceptron.sigma(weights[0] + op[0]*weights[1] + op[1]*weights[2])
        print(f"{weights[0]} + {op[0]}*{weights[1]} + {op[1]}*{weights[2]} = {final_result}")


if __name__ == '__main__':
    # Binary Op
    binary_ops = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    perceptron = Perceptron(input_size=2)
    and_labels = np.array([[0], [0], [0], [1]])
    perceptron.train(binary_ops, and_labels)
    print("1st Weight vector: ", perceptron.W)

    show_results(binary_ops, perceptron.W)
    or_labels = np.array([[0], [1], [1], [1]])
    perceptron.train(binary_ops, or_labels)
    print("2nd Weight vector: ", perceptron.W)
    show_results(binary_ops, perceptron.W)

    xor_labels = np.array([[0], [1], [1], [0]])
    perceptron.train(binary_ops, xor_labels)
    print("3rd Weight vector: ", perceptron.W)
    show_results(binary_ops, perceptron.W)

    # Now try some real data
    # Load the data set
    bc = datasets.load_breast_cancer()
    L = bc.data
    y = bc.target

    # Create training and test split
    X_train, X_test, y_train, y_test = train_test_split(L, y, test_size=0.3, random_state=42, stratify=y)

    perceptron = Perceptron(input_size=X_train.shape[1], epochs=100, lr=1)
    perceptron.train(X_train, y_train)

    print("Training set accuracy: ", perceptron.score(X_train, y_train))
    print("Test set accuracy: ", perceptron.score(X_test, y_test))
