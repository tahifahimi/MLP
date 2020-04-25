from numpy.random import random_sample
from math import exp, log
import random
import matplotlib.pyplot as plt



class SinglePerceptron :
    def __init__(self, train, test, resultOfTest):
        """
        Initialises the weights and b with random values between 0 and 1
        Sets the learning rate
        """
        random.seed(1)
        self.w = [0.01, 0.1]
        self.train = train
        self.test = test
        self.resultOfTest = resultOfTest

        # initialize the  n_epoch and lr
        self.lr = 0.05
        self.n_epoch = 300
        self.bias = 1

    def calculateY(self, x0, x1):
        """ find y of one coordinate
            W.X+b = x0w0 + x1w1 + b
            then find its sigmoid! """
        a = x0 * self.w[0] + x1*self.w[1] + self.bias
        return float(1)/(1+exp(a*-1))

    def calculateCost(self, yStart, y):
        """return the value of the cost for just one coordinate"""
        return yStart*log(y,(10))

    def calculateGradW(self, checker, x0, x1, y0, y):
        a = x0 * self.w[0] + x1 * self.w[1] + self.bias
        if checker=="w0":
            return (float(y0)/y*log(10))*x0*(exp(a*-1)/((1+exp(a*-1))*(1+exp(a*-1))))
        if checker=="w1":
            return (float(y0)/y*log(10))*x1*(exp(a*-1)/((1+exp(a*-1))*(1+exp(a*-1))))
        else:
            return (float(y0) / y * log(10)) * (exp(a * -1) / ((1 + exp(a * -1)) * (1 + exp(a * -1))))



    def learn(self):
        grad = [0,0]
        bGrad = 0
        """ compute the values of the W and bias for each epoch """
        for time in range(self.n_epoch) :
            grad[0] = 0
            grad[1] = 0
            bGrad = 0
            cost = 0
            modeledY =[]
            for i in range(len(self.train)):
                # compute y for the ith data in the train data
                y = self.calculateY(self.train[i][0], self.train[i][1])
                modeledY.append(y)
                # compute cost for the ith data
                cost += self.calculateCost(self.train[i][2], y)
                # grad[0] += dcost / dW0 is (y0/yln(10))*S(w0x0+w1x1+bias)(1-S(w0x0+w1x1+bias))* x0
                # grad[1] += dcost / dW1 is (y0/yln(10))*S(w0x0+w1x1+bias)(1-S(w0x0+w1x1+bias))* x1
                grad[0] += self.calculateGradW("w0", self.train[i][0], self.train[i][1], self.train[i][2], y)
                grad[1] += self.calculateGradW("w1", self.train[i][0], self.train[i][1], self.train[i][2], y)
                bGrad += self.calculateGradW("b", self.train[i][0], self.train[i][1], self.train[i][2], y)
            cost *= -1
            grad[0] *= -1
            grad[1] *= -1

            # draw each generation
            if time ==99 or time == 0:
                self.draw("train", time, modeledY)

            # assign the grads to the weights
            self.w[0] = self.w[0] - (self.lr * grad[0]) / len(self.train)
            self.w[1] = self.w[1] - (self.lr * grad[1]) / len(self.train)
            self.bias = self.bias - (self.lr * bGrad)/len(self.train)

            print("gen ",time)
            print(self.w[0]," ",self.w[1]," ",self.bias)

    def draw(self, typeOfDraw, generation, modeledY):
        """ draw data with considering the draw type"""
        if typeOfDraw == "train":
            colors = ["r", "b"]
            for i in range(len(self.train)):
                if modeledY[i] >= 0.5:
                    plt.scatter(self.train[i][0], self.train[i][1], color=colors[1])
                else:
                    plt.scatter(self.train[i][0], self.train[i][1], color=colors[0])
        else:
            colors = ["r", "b"]
            for i in range(len(self.test)):
                if modeledY[i] >= 0.5:
                    plt.scatter(self.test[i][0], self.test[i][1], color=colors[1])
                else:
                    plt.scatter(self.test[i][0], self.test[i][1], color=colors[0])

        # plt.show()
        plt.savefig(str(generation)+'SingleTrain.png')