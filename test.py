import numpy
from pandas import read_csv
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


class KNearestNeighbour:
    def __init__(self, k):
        self.k = k

    def train(self, data, target):
        self.data_train = data
        self.target_train = target

    def predict(self, data_test):
        distances = self.get_distance(data_test)
        return self.get_prediction(distances)

    def get_distance(self, data_test):
        testRows = data_test.shape[0]
        trainRows = self.data_train.shape[0]
        distances = numpy.zeros((testRows, trainRows))

        for i in range(testRows):
            for j in range(trainRows):  # calculate distances for each row with formula
                distances[i, j] = numpy.sqrt(numpy.sum((data_test.iloc[i, :] - self.data_train.iloc[j, :]) ** 2))

        return distances

    def get_prediction(self, distances):
        testRows = distances.shape[0]
        testPredictRows = numpy.zeros(testRows)

        for i in range(testRows):
            closest = numpy.argsort(distances[i, :])
            closestClasses = self.target_train.iloc[closest[:self.k]]
            testPredictRows[i] = numpy.argmax(numpy.bincount(closestClasses.values))

        return testPredictRows


dataset = read_csv('heart.csv')
data = dataset.iloc[:169, 0:13]
targets = dataset.iloc[:169, 13]

#add new test data
x = data.values[0]
for value in range(len(x)):
    x[value] = float(input())


scaler = StandardScaler()
data.iloc[:] = scaler.fit_transform(data.iloc[:])

knn = KNearestNeighbour(13)
knn.train(data, targets)
predictionRes = knn.predict(data.iloc[:169])

print(f'Accuracy Score: {accuracy_score(targets, predictionRes)}')
print(f'F1 Score: {f1_score(targets, predictionRes)}')
print(predictionRes)
