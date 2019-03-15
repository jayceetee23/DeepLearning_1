from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np

# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
dataset = pd.read_csv("Breast Cancer.csv", header=0).values

# Transform categorical features to numerical values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y = le.fit_transform(dataset[:, 1])

# print(dataset)

X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,2:31], y,
                                                    test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(40, input_dim=29, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, verbose=0,
                                     initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test, verbose=0))
