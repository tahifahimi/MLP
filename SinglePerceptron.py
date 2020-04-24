from numpy.random import random_sample
from math import exp
import random
import matplotlib.pyplot as plt

def calculateY(param, param1, param2):
    pass


def calculateCost():
    pass


class SinglePerceptron :
    def __init__(self, train, test, resultOfTest):
        """
        Initialises the weights and b with random values between 0 and 1
        Sets the learning rate
        """
        random.seed(1)
        self.w = [random.random(), random.random()]
        self.train = train
        self.test = test
        self.resultOfTest = resultOfTest

        # initialize the  n_epoch and lr
        self.lr = 0.001
        self.n_epoch = 100
        self.bias = 0

    def learn(self):
        grad = [0,0]
        bGrad = 0
        """ compute the values of the W and bias for each epoch """
        for time in  range(self.n_epoch) :
            grad [0] = 0
            grad[1] = 0
            bGrad = 0
            for i in range(len(self.train)):
                # compute y
                y = calculateY(self.train[i][0],self.train[i][1],self.train[i][2])
                # compute cost
                cost = calculateCost()
                grad[0] += dcost / dW
                grad[1] += dcost / dW

            # assign the grads to the weights
            self.w[0] = self.w[0] – (self.lr * grad[0]) / len(self.train)
            self.w[1] = self.w[1] – (self.lr * grad[1]) / len(self.train)
            self.bias = self.bias – (self.lr * bGrad)/len(self.train)


    def draw(self,typeOfDraw):
        """ draw data with considering the draw type"""
        if typeOfDraw == "train" :
            colors = ["r", "b"]
            for i in range(len(self.train)):
                if self.train[i][2] == 1:
                    plt.scatter(self.train[i][0], self.train[i][1], color=colors[1])
                else:
                    plt.scatter(self.train[i][0], self.train[i][1], color=colors[0])
        else:
            colors = ["r", "b"]
            for i in range(len(self.test)):
                if self.test[i][2] == 1:
                    plt.scatter(self.test[i][0], self.test[i][1], color=colors[1])
                else:
                    plt.scatter(self.test[i][0], self.test[i][1], color=colors[0])
