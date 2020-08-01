import numpy as np
from perceptron import Perceptron

from sklearn.preprocessing import OneHotEncoder


class TicTacToeClassifier:
    '''
    a tool to train and test on my data (it will be opinionated about the data)
    '''

    def __init__(self):
        self.encoder_board = OneHotEncoder()

    def train(self, X_train, y_train):
        X_train = self.encoder_board.fit_transform(X_train).toarray()

        self.perceptron = Perceptron(len(X_train[0]))
        self.perceptron.train(X_train, y_train)

    def test(self, X_test):
        X_test = self.encoder_board.fit_transform(X_test).toarray()

        y_hat = np.array([])

        for x in X_test:
            prediction = self.perceptron.predict(x)
            y_hat = np.append(y_hat, prediction)

        return y_hat
