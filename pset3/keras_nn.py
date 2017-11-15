
from load_data import load_data_from_txt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from plotBoundary import plotDecisionBoundary
from keras.utils import plot_model


def neural_network(trainX, trainY, validationX, validationY):

    model = Sequential()

    model.add(Dense(12, input_shape=(2,), activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(trainX, trainY, epochs=1000, batch_size=100)

    predictNN = model.predict(validationX)
    print ("Prediction: ", predictNN)

    scores = model.evaluate(validationX, validationY)

    #plot_model(model, to_file='model.png')

    #plotDecisionBoundary(validationX, validationY, model.predict, [-1, 0, 1], title = 'SVM Train')
    print ("score ", scores)
    print("{}: {}".format(model.metrics_names[1], scores[1]*100))


if __name__ == '__main__':

    name = "1"

    trainData, validationData, testData = load_data_from_txt(name)

    # do training
    neural_network(trainData[0], trainData[1], validationData[0], validationData[1])
