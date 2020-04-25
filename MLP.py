from math import exp
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, train, test, resultOfTest):
        """
        Initialises the weights(w,u,v) and bias(b0,b1,b2) with random values between 0 and 1
        Sets the learning rate and epoch
        """
        self.w = [0.1, 0.1]
        self.v = [0.1, 0.1]
        self.u = [0.1, 0.1]
        self.b0 = 1
        self.b1 = 1
        self.b2 = 1

        self.train = train
        self.test = test
        self.resultOfTest = resultOfTest

        self.lr = 0.05
        self.n_epoch = 300

    def calculateY(self, z0, z1):
        """ find y for each coordinate
            Z.U+b2 = z0u0 + z1u1 + b2
            then find its sigmoid! """
        a = z0 * self.u[0] + z1 * self.u[1] + self.b2
        return float(1) / (1 + exp(a * -1))

    def calculateZ0(self, x0, x1):
        a = x0 * self.w[0] + x1 * self.w[1] + self.b0
        return float(1) / (1 + exp(a * -1))

    def calculateZ1(self, x0, x1):
        a = x0 * self.v[0] + x1 * self.v[1] + self.b1
        return float(1) / (1 + exp(a * -1))

    def calculateCost(self, yt, y):
        """return the value of the cost for just one coordinate
            cost = (y - yt)^2 """
        return (y-yt)*(y-yt)


    def learn(self):
        """ compute the values of the W and bias for each epoch """
        for time in range(self.n_epoch):
            gradW = [0, 0]
            gradV = [0, 0]
            gradU = [0, 0]
            gradB0 = 0
            gradB1 = 0
            gradB2 = 0
            cost = 0
            resultOfModel = []
            for i in range(len(self.train)):
                # compute y for the ith data in the train data
                y = self.calculateY(self.train[i][0], self.train[i][1])
                resultOfModel.append(y)
                z0 = self.calculateZ0(self.train[i][0], self.train[i][1])
                z1 = self.calculateZ1(self.train[i][0], self.train[i][1])
                # compute cost for the ith data
                cost += self.calculateCost(self.train[i][2], y)
                # compute the gradients of all weights
                gradW[0] += self.calculateGradW("w0", self.train[i][0], self.train[i][1], self.train[i][2], y, z0, z1)
                gradW[1] += self.calculateGradW("w1", self.train[i][0], self.train[i][1], self.train[i][2], y, z0, z1)
                gradB0 += self.calculateGradW("b0", self.train[i][0], self.train[i][1], self.train[i][2], y, z0, z1)

                gradV[0] += self.calculateGradW("v0", self.train[i][0], self.train[i][1], self.train[i][2], y, z0, z1)
                gradV[1] += self.calculateGradW("v1", self.train[i][0], self.train[i][1], self.train[i][2], y, z0, z1)
                gradB1 += self.calculateGradW("b1", self.train[i][0], self.train[i][1], self.train[i][2], y, z0, z1)

                gradU[0] += self.calculateGradW("u0", self.train[i][0], self.train[i][1], self.train[i][2], y, z0, z1)
                gradU[1] += self.calculateGradW("u1", self.train[i][0], self.train[i][1], self.train[i][2], y, z0, z1)
                gradB2 += self.calculateGradW("b2", self.train[i][0], self.train[i][1], self.train[i][2], y, z0, z1)

            # draw each generation
            if time == 299 or time == 0:
                self.draw("train", time, resultOfModel)

            print("gen ", time)
            print(cost)
            # assign the grads to the weights
            self.w[0] = self.w[0] - (self.lr * gradW[0]) / len(self.train)
            self.w[1] = self.w[1] - (self.lr * gradW[1]) / len(self.train)
            self.b0 = self.bias - (self.lr * gradB0) / len(self.train)

            self.V[0] = self.v[0] - (self.lr * gradV[0]) / len(self.train)
            self.V[1] = self.v[1] - (self.lr * gradV[1]) / len(self.train)
            self.b1 = self.b1 - (self.lr * gradB1) / len(self.train)

            self.U[0] = self.U[0] - (self.lr * gradU[0]) / len(self.train)
            self.U[1] = self.U[1] - (self.lr * gradU[1]) / len(self.train)
            self.b2 = self.b2 - (self.lr * gradB2) / len(self.train)

    def draw(self, typeOfDraw, generation, resultOfModel):
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
        plt.savefig(str(generation) + 'SingleTrain.png')

    def calculateGradientW(self, checker, x0, x1, yt, y, z0, z1):
        a = x0 * self.w[0] + x1 * self.w[1] + self.b0
        l = z0 * self.u[0] + z1 * self.u[1] + self.b2
        if checker == "w0":
            return 2*(y-yt) * self.u[0] *(exp(-1 * l) / (1 + exp(-1 * l) * (1 + exp(-1 * l)))) * x0*(exp(-1 * a) / (1 + exp(-1 * a) * (1 + exp(-1 * a))))
        if checker == "w1":
            return 2*(y-yt) * self.u[0] *(exp(-1 * l) / (1 + exp(-1 * l) * (1 + exp(-1 * l)))) * x1*(exp(-1 * a) / (1 + exp(-1 * a) * (1 + exp(-1 * a))))
        if checker == "b0":
            return 2*(y-yt) * self.u[0] *(exp(-1 * l) / (1 + exp(-1 * l) * (1 + exp(-1 * l)))) * (exp(-1 * a) / (1 + exp(-1 * a) * (1 + exp(-1 * a))))

    def calculateGradientV(self, checker, x0, x1, yt, y, z0, z1):
        a = x0 * self.v[0] + x1 * self.v[1] + self.b1
        l = z0 * self.u[0] + z1 * self.u[1] + self.b2
        if checker == "v0":
            return 2 * (y - yt) * self.u[1] * (exp(-1 * l) / (1 + exp(-1 * l) * (1 + exp(-1 * l)))) * x0 * (
                        exp(-1 * a) / (1 + exp(-1 * a) * (1 + exp(-1 * a))))
        if checker == "v1":
            return 2 * (y - yt) * self.u[1] * (exp(-1 * l) / (1 + exp(-1 * l) * (1 + exp(-1 * l)))) * x1 * (
                        exp(-1 * a) / (1 + exp(-1 * a) * (1 + exp(-1 * a))))
        if checker == "b1":
            return 2 * (y - yt) * self.u[1] * (exp(-1 * l) / (1 + exp(-1 * l) * (1 + exp(-1 * l)))) * (
                        exp(-1 * a) / (1 + exp(-1 * a) * (1 + exp(-1 * a))))


    def calculateGradientU(self, checker, x0, x1, yt, y, z0, z1):
        l = z0 * self.u[0] + z1 * self.u[1] + self.b2
        if checker == "u0":
            return 2 * (y - yt) * self.u[0] * (exp(-1 * l) / (1 + exp(-1 * l) * (1 + exp(-1 * l))))
        if checker == "u1":
            return 2 * (y - yt) * self.u[1] * (exp(-1 * l) / (1 + exp(-1 * l) * (1 + exp(-1 * l))))
        if checker == "b2":
            return 2 * (y - yt) * (exp(-1 * l) / (1 + exp(-1 * l) * (1 + exp(-1 * l))))

