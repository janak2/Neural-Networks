
from load_data import load_data_from_txt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from plotBoundary import plotDecisionBoundary
from keras.utils import *
from keras.optimizers import *


def neural_network_singlesmall(trainX, trainY, validationX, validationY, learning_rate):
    model = Sequential()
    model.add(Dense(2, input_shape=(2,), activation='relu'))
    model.add(Dense(2, activation='softmax'))
    one_hot_labels = to_categorical(trainY, num_classes=2)
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr = learning_rate), metrics=['accuracy'])
    model.fit(trainX, one_hot_labels, epochs=1000, batch_size=100, verbose = 0)
    predictNN = model.predict(validationX)
    scores = model.evaluate(validationX, to_categorical(validationY, num_classes=2))
    return scores

def neural_network_singlelarge(trainX, trainY, validationX, validationY, learning_rate):
    model = Sequential()
    model.add(Dense(200, input_shape=(2,), activation='relu'))
    model.add(Dense(2, activation='softmax'))
    one_hot_labels = to_categorical(trainY, num_classes=2)
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr = learning_rate), metrics=['accuracy'])
    model.fit(trainX, one_hot_labels, epochs=1000, batch_size=100, verbose = 0)
    predictNN = model.predict(validationX)
    scores = model.evaluate(validationX, to_categorical(validationY, num_classes=2))
    return scores

def neural_network_doublesmall(trainX, trainY, validationX, validationY, learning_rate,):
    model = Sequential()
    model.add(Dense(2, input_shape=(2,), activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    one_hot_labels = to_categorical(trainY, num_classes=2)
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr = learning_rate), metrics=['accuracy'])
    model.fit(trainX, one_hot_labels, epochs=1000, batch_size=100, verbose = 0)
    predictNN = model.predict(validationX)
    scores = model.evaluate(validationX, to_categorical(validationY, num_classes=2))
    return scores

def neural_network_doublelarge(trainX, trainY, validationX, validationY, learning_rate):
    model = Sequential()
    model.add(Dense(200, input_shape=(2,), activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    one_hot_labels = to_categorical(trainY, num_classes=2)
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr = learning_rate), metrics=['accuracy'])
    model.fit(trainX, one_hot_labels, epochs=1000, batch_size=100, verbose = 0)
    predictNN = model.predict(validationX)
    scores = model.evaluate(validationX, to_categorical(validationY, num_classes=2))
    return scores
if __name__ == '__main__':

    data_sets = ["1","2","3","4"]
    validation_sets = [.0001, .001, .01, .1, 1]
    for name in data_sets:
        trainData, validationData, testData = load_data_from_txt(name)
        for validation in validation_sets:
    # do training
            print(" data set = " + name + "     learning rate = " + str(validation) + "       " + "11111111111111111")
            print(neural_network_singlesmall(trainData[0], trainData[1], validationData[0], validationData[1], validation))
            print(" data set = " + name + "     learning rate = " + str(validation) + "       " + "22222222222222222")
            print(neural_network_singlelarge(trainData[0], trainData[1], validationData[0], validationData[1], validation))
            print(" data set = " + name + "     learning rate = " + str(validation) + "       " + "33333333333333333")
            print(neural_network_doublesmall(trainData[0], trainData[1], validationData[0], validationData[1], validation))
            print(" data set = " + name + "     learning rate = " + str(validation) + "       " + "44444444444444444")
            print(neural_network_doublelarge(trainData[0], trainData[1], validationData[0], validationData[1], validation))
