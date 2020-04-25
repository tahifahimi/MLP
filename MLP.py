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
        self.b0 = 2
        self.b1 = 2
        self.b2 = 2

        self.train = train
        self.test = test
        self.resultOfTest = resultOfTest

        self.lr = 0.1
        self.n_epoch = 600

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

    def calculateGradientW(self, checker, x0, x1, yt, y, z0, z1):
        """ dcost/dw = dcost/dy * dy/dz * dz/dw
            dcost/dwi = 2*(y-yt)* u0*S(z0 """
        A = z0 * self.u[0] + z1 * self.u[1] + self.b2
        B = x0 * self.w[0] + x1 * self.w[1] + self.b0
        if checker == "w0":
            return 2 * (y - yt) * self.u[0] * (exp(-1 * A) / (1 + exp(-1 * A) * (1 + exp(-1 * A)))) * x0 * (
                        exp(-1 * B) / (1 + exp(-1 * B) * (1 + exp(-1 * B))))
        if checker == "w1":
            return 2 * (y - yt) * self.u[0] * (exp(-1 * A) / (1 + exp(-1 * A) * (1 + exp(-1 * A)))) * x1 * (
                        exp(-1 * B) / (1 + exp(-1 * B) * (1 + exp(-1 * B))))
        if checker == "b0":
            return 2 * (y - yt) * self.u[0] * (exp(-1 * A) / (1 + exp(-1 * A) * (1 + exp(-1 * A)))) * (
                        exp(-1 * B) / (1 + exp(-1 * B) * (1 + exp(-1 * B))))

    def calculateGradientV(self, checker, x0, x1, yt, y, z0, z1):
        A = x0 * self.v[0] + x1 * self.v[1] + self.b1
        B = z0 * self.u[0] + z1 * self.u[1] + self.b2
        if checker == "v0":
            return 2 * (y - yt) * self.u[1] * (exp(-1 * B) / (1 + exp(-1 * B) * (1 + exp(-1 * B)))) * x0 * (
                    exp(-1 * A) / (1 + exp(-1 * A) * (1 + exp(-1 * A))))
        if checker == "v1":
            return 2 * (y - yt) * self.u[1] * (exp(-1 * B) / (1 + exp(-1 * B) * (1 + exp(-1 * B)))) * x1 * (
                    exp(-1 * A) / (1 + exp(-1 * A) * (1 + exp(-1 * A))))
        if checker == "b1":
            return 2 * (y - yt) * self.u[1] * (exp(-1 * B) / (1 + exp(-1 * B) * (1 + exp(-1 * B)))) * (
                    exp(-1 * A) / (1 + exp(-1 * A) * (1 + exp(-1 * A))))

    def calculateGradientU(self, checker, yt, y, z0, z1):
        l = z0 * self.u[0] + z1 * self.u[1] + self.b2
        if checker == "u0":
            return 2 * (y - yt) * z0 * (exp(-1 * l) / (1 + exp(-1 * l) * (1 + exp(-1 * l))))
        if checker == "u1":
            return 2 * (y - yt) * z1 * (exp(-1 * l) / (1 + exp(-1 * l) * (1 + exp(-1 * l))))
        if checker == "b2":
            return 2 * (y - yt) * (exp(-1 * l) / (1 + exp(-1 * l) * (1 + exp(-1 * l))))

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
                gradW[0] += self.calculateGradientW("w0", self.train[i][0], self.train[i][1], self.train[i][2], y, z0, z1)
                gradW[1] += self.calculateGradientW("w1", self.train[i][0], self.train[i][1], self.train[i][2], y, z0, z1)
                gradB0 += self.calculateGradientW("b0", self.train[i][0], self.train[i][1], self.train[i][2], y, z0, z1)

                gradV[0] += self.calculateGradientV("v0", self.train[i][0], self.train[i][1], self.train[i][2], y, z0, z1)
                gradV[1] += self.calculateGradientV("v1", self.train[i][0], self.train[i][1], self.train[i][2], y, z0, z1)
                gradB1 += self.calculateGradientV("b1", self.train[i][0], self.train[i][1], self.train[i][2], y, z0, z1)

                gradU[0] += self.calculateGradientU("u0", self.train[i][2], y, z0, z1)
                gradU[1] += self.calculateGradientU("u1", self.train[i][2], y, z0, z1)
                gradB2 += self.calculateGradientU("b2", self.train[i][2], y, z0, z1)

            # draw each generation
            if time == self.n_epoch-1 or time == 0 or time == self.n_epoch/2:
                self.draw("train", time, resultOfModel)

            print("gen ", time)
            print(cost)
            # assign the grads to the weights
            self.w[0] = self.w[0] - (self.lr * gradW[0]) / len(self.train)
            self.w[1] = self.w[1] - (self.lr * gradW[1]) / len(self.train)
            self.b0 = self.b0 - (self.lr * gradB0) / len(self.train)

            self.v[0] = self.v[0] - (self.lr * gradV[0]) / len(self.train)
            self.v[1] = self.v[1] - (self.lr * gradV[1]) / len(self.train)
            self.b1 = self.b1 - (self.lr * gradB1) / len(self.train)

            self.u[0] = self.u[0] - (self.lr * gradU[0]) / len(self.train)
            self.u[1] = self.u[1] - (self.lr * gradU[1]) / len(self.train)
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
        plt.savefig(str(generation+1) + 'MLPTrain.png')

