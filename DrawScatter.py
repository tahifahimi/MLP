import csv
import matplotlib.pyplot as plt
from SinglePerceptron import SinglePerceptron
from MLP import MLP

class drawScatter:
    def __init__(self, learn):
        self.loc = []
        self.learnFactor = learn

        self.test = []
        self.train = []
        self.resultOfTest = []

    def readFile(self, path):
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    self.loc.append([float(row[0]), float(row[1]), int(row[2])])
                    line_count += 1

    def drawChart(self,array,name):
        colors = ["r", "b"]
        print(name,len(array))
        for i in range(len(array)):
            if array[i][2] == 1:
                plt.scatter(array[i][0], array[i][1], color=colors[1])
            else:
                plt.scatter(array[i][0], array[i][1], color=colors[0])

        # save scatter
        plt.savefig(name+'.png')
        # plot scatter
        # plt.show()

    def seperateTrainData(self):
        numberOfTrainData = int(len(self.loc) * self.learnFactor)
        print(numberOfTrainData)
        for i in range(len(self.loc)):
            if i < numberOfTrainData:
                self.train.append(self.loc[i])
            else:
                self.resultOfTest.append(self.loc[i][2])
                # self.loc[i][2] = 0
                self.test.append(self.loc[i])

        print("number  of test data is ", len(self.test))
        self.drawChart(self.train, "train")
        # self.drawChart(test, "test")
        # f = open("testFile.txt", "a")
        # f.write(str(test[0]))
        # f.close()


if __name__ == "__main__":
    s = drawScatter(0.9)  # pass the learning factor to the class
    s.readFile('dataset.csv')
    s.drawChart(s.loc,"main")

    s.seperateTrainData()

    # now call the Single perceptron and pass data
    # perceptron = SinglePerceptron(s.train, s.test, s.resultOfTest)
    # perceptron.learn()

    # create MLP
    mlp = MLP(s.train, s.test, s.resultOfTest)
    mlp.learn()