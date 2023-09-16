
import numpy as np

def load_data_from_txt(name):


    train = np.loadtxt('data/data' + name + '_' + 'train.csv', dtype=np.float64)

    trainX = np.array(train[:, 0:2])
    trainY = np.array(train[:, 2:3])
    trainY[trainY < 0] = 0
    trainY = np.squeeze(trainY)

    trainData = (trainX, trainY)


    validation = np.loadtxt('data/data' + name + '_' + 'validate.csv', dtype=np.float64)

    validationX = np.array(validation[:, 0:2])
    validationY = np.array(validation[:, 2:3])
    validationY[validationY < 0] = 0
    validationY = np.squeeze(validationY)

    validationData = (validationX, validationY)


    test = np.loadtxt('data/data' + name + '_' + 'test.csv', dtype=np.float64)

    testX = np.array(test[:, 0:2])
    testY = np.array(test[:, 2:3])
    testY[testY < 0] = 0
    testY = np.squeeze(testY)

    testData = (testX, testY)

    return trainData, validationData, testData

def load_mnist():

    path = 'data/mnist_digit_'
    ext = '.csv'

    trainXs = []
    trainYs = []
    valXs = []
    valYs = []
    testXs = []
    testYs = []

    for digit in range(10):


        # train data
        train = np.loadtxt(path + str(digit) + ext, dtype=np.float64)
        train = 2 * np.true_divide(train, 255) - 1
        trainXs.append(train)
        trainYs.append(digit*np.ones((200,)))

        # validation data
        validation = np.loadtxt(path + str(digit) + ext, dtype=np.float64)
        validation = 2 * np.true_divide(validation, 255) - 1
        valXs.append(validation)
        valYs.append(digit*np.ones((150,)))

        # test data
        test = np.loadtxt(path + str(digit) + ext, dtype=np.float64)
        test = 2 * np.true_divide(test, 255) - 1
        testXs.append(test)
        testYs.append(digit*np.ones((150,)))


    trainX = np.concatenate(trainXs)
    trainY = np.concatenate(trainYs)
    valX = np.concatenate(valXs)
    valY = np.concatenate(valYs)
    testX = np.concatenate(testXs)
    testY = np.concatenate(testYs)

    print ("Train X: ", trainX)
    print ("Train Y: ", trainY)

    return trainX, trainY, valX, valY, testX, testY












if __name__ == '__main__':
    load_mnist()
