import numpy as np
from plotBoundary import *
import pylab as pl
import math

path = "hw3_resources/hw3_resources/"

# import your LR training code
def classify_checker(weight, bias, x_value, y_value):
    exponent = -1*(np.dot(weight, x_value) + bias)
    function_value = 1/(1+math.e**exponent)
    if function_value <= 0.5:
        if y_value == -1:
            return True
        return False
    if y_value == 1:
        return True
    return False

def update(weight, bias, x_value, y_value, lamb, regularizer, learning_rate):
    exponent = -1*y_value*(np.dot(weight, x_value) + bias)
    delta_w = 1/(1+math.e**exponent)*-1*y_value*x_value*e**exponent + lamb*regularizer(weight)
    delta_w0 = 1/(1+math.e**exponent)*-1*y_value*e**exponent
    return_w = weight - learning_rate*delta_w
    return_w0 = bias - learning_rate*delta_w0
    return return_w, return_w0

def regularizer_l2(weight):
    return 2*weight
def regularizer_l1(weight):
    return_array = []
    for element in weight:
        if element >= 0:
            return_array.append(1)
        else:
            return_array.append(-1)
    return np.array(return_array)

def nn_trainer(L,m,k,X,Y):
    W = {}
    B = {}
    W[2] = [[1]*m]*len(X[0])
    B[2] = [[1]*m]
    for i in range(3,L+1):
        W[i] = [[1]*m]*m
        B[i] = [1]*m 
    W[1+L] = [[1]*k]*m
    B[1+L] = 
    z = {}
    a = {}
    for x in X:
        a[1] = x
        for l in range(2,L+1):
            z[l] = np.dot(W[l],a[l-1])

# parameters
L = 2 #number of layers
m = 3 #number of units in each layer
k = 3 #number of classes

print ('======Input======')
train = loadtxt(path + "data/data_3class.csv")
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
trained_weight, trained_bias, weight1, weight2 = nn_trainer(L,m,k,X,Y)
