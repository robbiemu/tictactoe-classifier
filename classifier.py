import numpy as np
#from perceptron import Perceptron
from svm import KernelPerceptron
from perceptron import Perceptron

from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict

classifiers = defaultdict(lambda: 'perceptron')
classifiers['perceptron'] = 'perceptron'
classifiers['svm'] = 'svm'
classifiers['dkp'] = 'dkp'


class TicTacToeClassifier:
    '''
    a tool to train and test on my data (it will be opinionated about the data)
    '''

    def __init__(self, classifier):
        self.encoder_board = OneHotEncoder()
        self.type = classifier

    def train(self, X_train, y_train):
        X_train = self.encoder_board.fit_transform(X_train).toarray()

        if self.type == 'perceptron':
            self.perceptron = Perceptron(len(X_train[0]))
        if self.type == 'svm':
            self.perceptron = KernelPerceptron(gamma=[(-1, 0), (1, 1)])
        if self.type == 'dkp':
            print('not implemented')
            exit(0)

        self.perceptron.train(X_train, y_train)

    def test(self, X_test):
        X_test = self.encoder_board.fit_transform(X_test).toarray()

        y_hat = np.array([])
        for x in X_test:
            prediction = self.perceptron.predict(x)
            y_hat = np.append(y_hat, prediction)

        return y_hat
