import numpy as np
from perceptron import Perceptron


class TicTacToeClassifier:
    '''
    a tool to train and test on my data (it will be opinionated about the data)
    '''

    def train(self, X_train, y_train):
        self.perceptron = Perceptron(9)  # len(X_train[0])
        self.perceptron.train(X_train, y_train)

    def test(self, X_test):
        y_hat = np.array([])
        for x in X_test:
            prediction = self.perceptron.predict(x)
            y_hat = np.append(y_hat, prediction)

        return y_hat
