from math import exp, log,log10
import matplotlib.pyplot as plt

class SinglePerceptron :
    def __init__(self, train, test, resultOfTest):
        """
        Initialises the weights and b with random values between 0 and 1
        Sets the learning rate
        """
        self.w = [0.1, 0.1]
        self.train = train
        self.test = test
        self.resultOfTest = resultOfTest

        # initialize the  n_epoch and lr
        self.lr = 0.1
        self.n_epoch = 1550
        self.bias = 1

    def calculateY(self, x0, x1):
        """ find y of one coordinate
            W.X+b = x0w0 + x1w1 + b
            then find its sigmoid! """
        a = x0 * self.w[0] + x1*self.w[1] + self.bias
        return float(1)/(1+exp(a*-1))

    def calculateCost(self, yStart, y):
        """return the value of the cost for just one coordinate"""
        return yStart*log10(y)+(1-yStart)*log10(1-y)
    
    def calculateGradW(self, checker, x0, x1, yt, y):
        a = x0 * self.w[0] + x1 * self.w[1] + self.bias
        if checker == "w0":
            return (float(yt-y)/float(y*(1-y)))*(exp(-1*a)/(1+exp(-1*a)*(1+exp(-1*a))))*x0
        if checker == "w1":
            return (float(yt-y)/float(y*(1-y)))*(exp(-1*a)/(1+exp(-1*a)*(1+exp(-1*a))))*x1
        if checker == "b":
            return (float(yt-y)/float(y*(1-y))) * (exp(a * -1) / ((1 + exp(a * -1)) * (1 + exp(a * -1))))

    def learn(self):
        """ compute the values of the W and bias for each epoch """
        for time in range(self.n_epoch) :
            grad = [0, 0]
            bGrad = 0
            cost = 0
            resultOfModel = []
            for i in range(len(self.train)):
                # compute y for the ith data in the train data
                y = self.calculateY(self.train[i][0], self.train[i][1])
                resultOfModel.append(y)
                # compute cost for the ith data
                cost += self.calculateCost(self.train[i][2], y)
                # compute the dcost/dw
                grad[0] += self.calculateGradW("w0", self.train[i][0], self.train[i][1], self.train[i][2], y)
                grad[1] += self.calculateGradW("w1", self.train[i][0], self.train[i][1], self.train[i][2], y)
                bGrad += self.calculateGradW("b", self.train[i][0], self.train[i][1], self.train[i][2], y)

            cost *= -1
            grad[0] *= float(-1)/log(10)
            grad[1] *= float(-1)/log(10)
            bGrad *= float(-1)/log(10)

            # draw each generation
            if time ==self.n_epoch-1 or time == 0:
                print("the final cost is ", cost)
                self.draw("train", time, self.passAccuracy(resultOfModel), resultOfModel)


            print("gen ", time)
            print(self.w[0], " ", self.w[1], " ", self.bias, " ", cost)
            # assign the grads to the weights
            self.w[0] = self.w[0] - (self.lr * grad[0]) / len(self.train)
            self.w[1] = self.w[1] - (self.lr * grad[1]) / len(self.train)
            self.bias = self.bias - (self.lr * bGrad)/len(self.train)



    def draw(self, typeOfDraw, generation, accuracy, resultOfModel):
        """ draw data with considering the draw type"""
        if typeOfDraw == "train":
            colors = ["r", "b"]
            for i in range(len(self.train)):
                if resultOfModel[i] >= 0.5:
                    plt.scatter(self.train[i][0], self.train[i][1], color=colors[1])
                else:
                    plt.scatter(self.train[i][0], self.train[i][1], color=colors[0])
        else:
            colors = ["r", "b"]
            for i in range(len(self.test)):
                if resultOfModel[i] >= 0.5:
                    plt.scatter(self.test[i][0], self.test[i][1], color=colors[1])
                else:
                    plt.scatter(self.test[i][0], self.test[i][1], color=colors[0])

        # plt.show()
        plt.savefig(str(generation+1)+'Single'+str(accuracy)+'.png')

    def passAccuracy(self, y):
        rightData = 0
        for l in range(len(y)):
            if y[l] >= 0.5:
                if self.train[l][2] == 1:
                    rightData += 1
            else:
                if self.train[l][2] == 0:
                    rightData += 1
        return "{:.2f}".format(float(rightData)/float(len(self.train)))