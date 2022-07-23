from math import sqrt

import numpy

import utils


class KNN:
    def __init__(self, n_neighbors=5, p=2, metric='minkowski'):
        self._X: list[list] = list()
        self._y: list[int] = list()
        self._n_neighbors = n_neighbors
        self._p = p
        self._metric = metric

    def fit(self, X, y):
        self._X = utils.validate_type(X)
        self._y = utils.validate_type(y)

    def predict(self, X: list):
        X = utils.validate_type(X)
        if isinstance(X[0], list):  # Lista de listas
            prediction_list = list()
            for test_row in X:
                neighbors = self._get_neighbors(test_row)
                output_values = [row[1] for row in neighbors]
                prediction_list.append(max(set(output_values), key=output_values.count))
            return numpy.asarray(prediction_list)
        else:
            neighbors = self._get_neighbors(X)
            output_values = [row[1] for row in neighbors]  # target list
            prediction = max(set(output_values), key=output_values.count)
            return numpy.asarray(prediction)

    def _eucliden_distance(self, row1: list, row2: list) -> float:
        """Calcula la distancia euclideana entre 2 vectores"""
        distance = 0.0
        for i in range(len(row1)):
            distance += (row1[i] - row2[i]) ** 2
        return sqrt(distance)

    def _get_neighbors(self, test_row: list) -> list[tuple[list, int]]:
        """Localiza los vecinos más cercanos"""
        distances = list()
        index = 0  # Save index for "y"
        for train_row in self._X:
            distance = self._eucliden_distance(test_row, train_row)
            distances.append((train_row, self._y[index], distance))
            index += 1
        distances.sort(key=lambda tup: tup[2])  # tup[1]: distance
        neighbors = list()
        for i in range(self._n_neighbors):
            neighbors.append((distances[i][0], distances[i][1]))
        return neighbors

    def score(self, X, y) -> float:
        X = utils.validate_type(X)
        y = utils.validate_type(y)
        correct = 0
        index = 0
        for row in X:
            prediction = self.predict(row)
            if y[index] == prediction:  # Predicción correta
                correct += 1
            index += 1
        return correct / float(len(X)) * 100.00

    def __sklearn_is_fitted__(self):
        return True


if __name__ == '__main__':
    # última fila debe ser encode
    import pandas as pd

    df = pd.read_csv('../iris.data.csv')

    X = df.iloc[:, :4]
    y = df.iloc[:, -1]

    y = utils.label_encoder(y)

    X_train, X_test, y_train, y_test = utils.train_test_split(X, y, test_size=0.33, random_state=42)

    knn = KNN(n_neighbors=4)
    knn.fit(X_train, y_train)

    result = knn.predict(X_test)
    print(result)

    value = knn.score(X_test, y_test)
    print('Score %s' % value)
