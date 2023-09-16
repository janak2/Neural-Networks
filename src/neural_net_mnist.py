
from load_data import load_mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from plotBoundary import plotDecisionBoundary
from keras.utils import *
from keras.optimizers import *


def neural_network_singlesmall(trainX, trainY, validationX, validationY, learning_rate):
    model = Sequential()
    model.add(Dense(16, input_shape=(784,), activation='relu'))
    model.add(Dense(10, activation='softmax'))
    one_hot_labels = to_categorical(trainY, num_classes=10)
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr = learning_rate), metrics=['accuracy'])
    model.fit(trainX, one_hot_labels, epochs=1000, batch_size=100, verbose = 0)
    predictNN = model.predict(validationX)
    scores = model.evaluate(validationX, to_categorical(validationY, num_classes=10))
    return scores

def neural_network_singlelarge(trainX, trainY, validationX, validationY, learning_rate):
    model = Sequential()
    model.add(Dense(175, input_shape=(784,), activation='relu'))
    model.add(Dense(10, activation='softmax'))
    one_hot_labels = to_categorical(trainY, num_classes=10)
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr = learning_rate), metrics=['accuracy'])
    model.fit(trainX, one_hot_labels, epochs=1000, batch_size=100, verbose = 0)
    predictNN = model.predict(validationX)
    scores = model.evaluate(validationX, to_categorical(validationY, num_classes=10))
    return scores

def neural_network_doublesmall(trainX, trainY, validationX, validationY, learning_rate,):
    model = Sequential()
    model.add(Dense(16, input_shape=(784,), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    one_hot_labels = to_categorical(trainY, num_classes=10)
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr = learning_rate), metrics=['accuracy'])
    model.fit(trainX, one_hot_labels, epochs=1000, batch_size=100, verbose = 0)
    predictNN = model.predict(validationX)
    scores = model.evaluate(validationX, to_categorical(validationY, num_classes=10))
    return scores

def neural_network_doublelarge(trainX, trainY, validationX, validationY, learning_rate):
    model = Sequential()
    model.add(Dense(175, input_shape=(784,), activation='relu'))
    model.add(Dense(175, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    one_hot_labels = to_categorical(trainY, num_classes=10)
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr = learning_rate), metrics=['accuracy'])
    model.fit(trainX, one_hot_labels, epochs=1000, batch_size=100, verbose = 0)
    predictNN = model.predict(validationX)
    scores = model.evaluate(validationX, to_categorical(validationY, num_classes=10))
    return scores
if __name__ == '__main__':

    validation_sets = [.0001, .001, .01, .1, 1]
    trainX, trainY, valX, valY, testX, testY = load_mnist()
    for validation in validation_sets:
    # do training
        print(" learning rate = " + str(validation) + "       " + "11111111111111111")
        print(neural_network_singlesmall(trainX, trainY, valX, valY, validation))
        print("learning rate = " + str(validation) + "       " + "22222222222222222")
        print(neural_network_singlelarge(trainX, trainY, valX, valY, validation))
        print("learning rate = " + str(validation) + "       " + "33333333333333333")
        print(neural_network_doublesmall(trainX, trainY, valX, valY, validation))
        print("learning rate = " + str(validation) + "       " + "44444444444444444")
        print(neural_network_doublelarge(trainX, trainY, valX, valY, validation))
