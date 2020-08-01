import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras import callbacks

'''
kinda like a jupyter notebook, but without the notebook
'''

'''
get features and labels values from a csv
a puzzle is usually a 3 by 3 matrix (3,3), but it has been written to a csv as a single row (9,)
'''
dataset = pd.read_csv('samples-tictactoe.csv')
X = dataset.iloc[:, 0:9].values
y = dataset.iloc[:, 9:10].values

'''
our features are categorial, so encode them.. 
this transforms the data from (9,) to (27,)
'''
encoder_board = OneHotEncoder()
X = encoder_board.fit_transform(X).toarray()

# we need to separate some data to train with, and reserve some to test on
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)  # when we set our hyperparameters or share results, we need to remove the variance, ie set a random_state. in practice though, we want to run a few times with different random_state, to remove bias

# let's use validation data to support early stopping
offset = -(len(X_train)//5)
X_val = X_train[offset:]
y_val = y_train[offset:]
X_train = X_train[:offset]
y_train = y_train[:offset]

# here begins our neural network
model = Sequential()

# we have an input layer of 27
# we have 3 hidden layers of 14
model.add(Dense(units=14, kernel_initializer='uniform',
                activation='relu', input_dim=27))
model.add(Dense(units=14, kernel_initializer='uniform',
                activation='relu'))
model.add(Dense(units=14, kernel_initializer='uniform',
                activation='relu'))
# we have an output layer of 1: this is a binary classifier: its either a vlaid tic tac toe board, or it is not.
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

'''
we want to train our network by modifying the internal weights based on some loss: binary crossentropy is perfectly fit for a binary classifier. for determining how we use our loss, we use adam (this is the go-to method in practice generally).
'''
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

'''
max of 500 epochs of training
at likely best fit, wait 1% of processing time to be sure before stopping early
'''
epochs = 500
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
model.fit(X_train, y_train, batch_size=batch_size,
          epochs=epochs, callbacks=[earlystopping], validation_data=(X_val, y_val))

# predict the results. double-yeay!
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# let the user know that something happened, because this is not a jupyter notebook
index = 0
miss = 0
for received, expected in list(zip(y_test, y_pred)):
    if received[0] != expected[0]:
        miss += 1
        print('received is not expected!', received[0], expected[0],
              encoder_board.inverse_transform(X_train[index].reshape(1, -1))[0])
    index += 1

print('test accuracy', 100 * (1 - miss/len(y_test)))
