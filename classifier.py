import numpy as np
from perceptron import Perceptron


class TicTacToeClassifier:
    '''sketch approach'''

    def train(self, X_train, y_train):
        self.perceptron = Perceptron(len(X_train[0]))
        self.perceptron.train(X_train, y_train)

    def test(self, X_test):
        y = np.array([])
        for x in X_test:
            y.append(self.perceptron.predict(x))

        return y
