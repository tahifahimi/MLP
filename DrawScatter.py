import csv
import matplotlib.pyplot as plt
# import plotly.plotly as py
import numpy as np

class drawScatter:
    def __init__(self):
        self.loc = []

    def readFile(self, path):
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    # print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    # print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
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
        # plt.show()
        plt.savefig('Scatter.png')


if __name__ == "__main__":
    scatter = drawScatter()
    scatter.readFile('dataset.csv')
    scatter.drawChart()