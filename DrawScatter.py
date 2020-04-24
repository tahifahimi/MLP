import csv
import matplotlib.pyplot as plt

from SinglePerceptron import SinglePerceptron


class drawScatter:
    def __init__(self, learn):
        self.loc = []
        self.learnFactor = learn

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
            # print(f'Processed {line_count} lines.')

    def drawChart(self):
        print(len(self.loc))
        print(self.loc[0][0])
        print(self.loc[0][1])
        print(type(self.loc[0][0]))
        colors = ["r", "b"]
        for i in range(len(self.loc)):
            if self.loc[i][2] == 1:
                plt.scatter(self.loc[i][0], self.loc[i][1], color=colors[1])
            else:
                plt.scatter(self.loc[i][0], self.loc[i][1], color=colors[0])

        # save scatter
        plt.savefig('Scatter.png')
        # plot scatter
        plt.show()

    def trainAndTestData(self):
        train = []
        test = []
        resultOfTest = []

        numberOfTrainData = len(self.loc) * self.learnFactor
        for i in range(numberOfTrainData):
            train.append(self.loc[i])
        for i in range(len(self.loc) - numberOfTrainData):
            resultOfTest.append(self.loc[i][2])
            self.loc[i][2] = 0
            test.append(self.loc[i])

        # now call the Single perceptron and pass data
        perceptron = SinglePerceptron(train, test, resultOfTest)
        perceptron.draw('train')




if __name__ == "__main__" :
    scatter = drawScatter(0.8)  # pass the learning factor to the class
    scatter.readFile('dataset.csv')
    scatter.drawChart()
    scatter.trainAndTestData()