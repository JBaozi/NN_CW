import csv
import pickle
import os
import codecs

import numpy as np

from urllib.request import urlopen

import matplotlib.pyplot as plt


class TravelingSalesmanProblem:
    """This class encapsulates the Traveling Salesman Problem.
    City coordinates are read from an online file and distance matrix is calculated.
    The data is serialized to disk.
    The total distance can be calculated for a path represented by a list of city indices.
    A plot can be created for a path represented by a list of city indices.

    :param name: The name of the corresponding TSPLIB problem, e.g. 'burma14' or 'bayg29'.
    """

    def __init__(self, name):
        """
        Creates an instance of a TSP

        :param name: name of the TSP problem
        """

        # инициализируем переменные экземпляра
        self.name = name
        self.locations = []
        self.distances = []
        self.tspSize = 0

        # инициализируем данные
        self.__initData()

    def __len__(self):
        """
        returns the length of the underlying TSP
        :return: the length of the underlying TSP (number of cities)
        """
        return self.tspSize

    def __initData(self):
        """Reads the serialized data, and if not available - calls __create_data() to prepare it
        """

        # пытаемся прочитать сериализованные данные
        try:
            self.locations = pickle.load(open(os.path.join("tsp-data", self.name + "-loc.pickle"), "rb"))
            self.distances = pickle.load(open(os.path.join("tsp-data", self.name + "-dist.pickle"), "rb"))
        except (OSError, IOError):
            pass

        # если сериализованные данные не найдены - создаем данные с нуля:
        if not self.locations or not self.distances:
            self.__createData()

        # устанавливаем проблему 'размер':
        self.tspSize = len(self.locations)

    def __createData(self):
        """Reads the desired TSP file from the Internet, extracts the city coordinates, calculates the distances
        between every two cities and uses them to populate a distance matrix (two-dimensional array).
        It then serializes the city locations and the calculated distances to disk using the pickle utility.
        """
        self.locations = []

        # открываем файл с разделителями-пробелами из url-адреса и читаем из него строки:
        with urlopen("http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/" + self.name + ".tsp") as f:
            reader = csv.reader(codecs.iterdecode(f, 'utf-8'), delimiter=" ", skipinitialspace=True)

            # пропускаем строки, пока не будет найдена одна из этих строк:
            for row in reader:
                if row[0] in ('DISPLAY_DATA_SECTION', 'NODE_COORD_SECTION'):
                    break

            # читаем строки данных до тех пор, пока не найдем EOF:
            for row in reader:
                if row[0] != 'EOF':
                    # удалим индекс в начале строки:
                    del row[0]

                    # преобразуем координаты x, y в ndarray:
                    self.locations.append(np.asarray(row, dtype=np.float32))
                else:
                    break

            # установим проблему 'размер':
            self.tspSize = len(self.locations)

            # выведем данные:
            print("length = {}, locations = {}".format(self.tspSize, self.locations))

            # инициализируем матрицу расстояний, заполнив ее нулями:
            self.distances = [[0] * self.tspSize for _ in range(self.tspSize)]

            # заполним матрицу расстояний вычисленными расстояниями:
            for i in range(self.tspSize):
                for j in range(i + 1, self.tspSize):
                    # вычислим евклидово расстояние между двумя ndarrays:
                    distance = np.linalg.norm(self.locations[j] - self.locations[i])
                    self.distances[i][j] = distance
                    self.distances[j][i] = distance
                    print("{}, {}: location1 = {}, location2 = {} => distance = {}".format(i, j, self.locations[i], self.locations[j], distance))

            # сериализируем местоположения и расстояния:
            if not os.path.exists("tsp-data"):
                os.makedirs("tsp-data")
            pickle.dump(self.locations, open(os.path.join("tsp-data", self.name + "-loc.pickle"), "wb"))
            pickle.dump(self.distances, open(os.path.join("tsp-data", self.name + "-dist.pickle"), "wb"))

    def getTotalDistance(self, indices):
        """Calculates the total distance of the path described by the given indices of the cities

        :param indices: A list of ordered city indices describing the given path.
        :return: total distance of the path described by the given indices
        """
        # расстояние между последним и первым городом:
        distance = self.distances[indices[-1]][indices[0]]

        # сложим расстояние между каждой парой следующих друг за другом городов:
        for i in range(len(indices) - 1):
            distance += self.distances[indices[i]][indices[i + 1]]

        return distance

    def plotData(self, indices):
        """plots the path described by the given indices of the cities

        :param indices: A list of ordered city indices describing the given path.
        :return: the resulting plot
        """

        # нарисуем точки, представляющие города:
        plt.scatter(*zip(*self.locations), marker='.', color='red')

        # создадим список соответствующих населенных пунктов города:
        locs = [self.locations[i] for i in indices]
        locs.append(locs[0])

        # проведем линию между каждой парой последовательных городов:
        plt.plot(*zip(*locs), linestyle='-', color='blue')

        return plt


# тестирование
def main():
    # создадим экземпляр проблемы:
    tsp = TravelingSalesmanProblem("bayg29")

    optimalSolution = [0, 27, 5, 11, 8, 25, 2, 28, 4, 20, 1, 19, 9, 3, 14, 17, 13, 16, 21, 10, 18, 24, 6, 22, 7, 26, 15, 12, 23]

    print("Problem name: " + tsp.name)
    print("Optimal solution = ", optimalSolution)
    print("Optimal distance = ", tsp.getTotalDistance(optimalSolution))

    # построим решение:
    plot = tsp.plotData(optimalSolution)
    plot.show()


if __name__ == "__main__":
    main()
