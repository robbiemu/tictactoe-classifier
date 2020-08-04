import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras import callbacks


class FCNN(object):
    def __init__(self, categories, with_earlystopping=False):
        self.with_earlystopping = with_earlystopping
        self.categories = categories

        # here begins our neural network
        self.model = Sequential()

        # we have an input layer of 27
        # we have 3 hidden layers of 14 (27 + 1 / 2)
        self.model.add(Dense(units=14, kernel_initializer='uniform',
                             activation='relu', input_dim=27))
        self.model.add(Dense(units=14, kernel_initializer='uniform',
                             activation='relu'))
        self.model.add(Dense(units=14, kernel_initializer='uniform',
                             activation='relu'))
        # we have an output layer of 1: this is a binary classifier: its either a vlaid tic tac toe board, or it is not.
        self.model.add(
            Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

        '''
        we want to train our network by modifying the internal weights based on some loss: binary crossentropy is perfectly fit for a binary classifier. for determining how we use our loss, we use adam (this is the go-to method in practice generally).
        '''
        self.model.compile(optimizer='adam', loss='binary_crossentropy',
                           metrics=['accuracy'])

    def train(self, X_train, y_train):
        X_train = X_train.tolist()

        if self.with_earlystopping:
            # let's use validation data to support early stopping
            offset = -(len(X_train)//5)
            X_val = X_train[offset:]
            y_val = y_train[offset:]
            X_train = X_train[:offset]
            y_train = y_train[:offset]

        '''
        max epochs of training - the number of epochs might need to be much lower, or could need to be higher as well. I am using the complexity of the data.
        at likely best fit, wait 1% of processing time to be sure before stopping early
        '''
        sigma = 1.5
        complexity_of_data = sigma * 9 * \
            len(self.categories) * len(np.unique(y_train))
        scaling = math.log(len(y_train), 10) + 1
        # by hand, went from 100 to 500
        epochs = int(complexity_of_data * scaling)
        print('epochs', epochs)

        if self.with_earlystopping:
            earlystopping = callbacks.EarlyStopping(monitor="val_accuracy",
                                                    mode="max", patience=epochs//100,
                                                    restore_best_weights=True)
        '''
        it is suggested to decrease batch size as much as possible for parallelism in gpu training, but lower batch sizes won't globally optimize as well.
        in our case, we know some things about the sample data. I generated it with of-each, generating three different classes of data: fully random, sparse random, and random but valid. so we will need at least a batch_size large enough to allow each batch a chance of reflecting 3 states. that size grows with the log of the training data length
        '''
        batch_size = 3 * (int(math.log(len(X_train))) + 1)
        print('batch_size', batch_size)

        # train the data, yeay!
        if self.with_earlystopping:
            self.model.fit(X_train, y_train, batch_size=batch_size,
                           epochs=epochs, callbacks=[earlystopping], validation_data=(X_val, y_val))
        else:
            self.model.fit(X_train, y_train, batch_size=batch_size,
                           epochs=epochs)

    def predict(self, X_test):
        y_pred = (self.model.predict(X_test) > 0.5).astype("int32")
        return y_pred
