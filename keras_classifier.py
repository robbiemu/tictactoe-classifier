import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras import callbacks

'''
kinda like a jupyter notebook, but without the notebook
'''

dataset = pd.read_csv('samples-tictactoe.csv')
X = dataset.iloc[:, 0:9].values
y = dataset.iloc[:, 9:10].values

encoder_board = OneHotEncoder()
X = encoder_board.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

offset = -(len(X_train)//5)
X_val = X_train[offset:]
y_val = y_train[offset:]
X_train = X_train[:offset]
y_train = y_train[:offset]

model = Sequential()

model.add(Dense(units=14, kernel_initializer='uniform',
                activation='relu', input_dim=27))
model.add(Dense(units=14, kernel_initializer='uniform',
                activation='relu'))
model.add(Dense(units=14, kernel_initializer='uniform',
                activation='relu'))

model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

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

model.fit(X_train, y_train, batch_size=10,
          epochs=epochs, callbacks=[earlystopping], validation_data=(X_val, y_val))
y_pred = (model.predict(X_test) > 0.5).astype("int32")

index = 0
miss = 0
for received, expected in list(zip(y_test, y_pred)):
    if received[0] != expected[0]:
        miss += 1
        print('received is not expected!', received[0], expected[0],
              encoder_board.inverse_transform(X_train[index].reshape(1, -1))[0])
    index += 1

print('test accuracy', 100 * (1 - miss/len(y_test)))
