import pandas
import numpy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


dataset = pandas.read_csv('heart.csv')

dataNoRes = dataset.iloc[:, 0:13]
dataRes = dataset.iloc[:, 13]
train_dataNoRes, test_dataNoRes, train_dataRes, test_dataRes = train_test_split(dataNoRes, dataRes, test_size=0.3)

# scaler = StandardScaler()
# train_dataNoRes = scaler.fit_transform(train_dataNoRes)
# test_dataNoRes = scaler.transform(test_dataNoRes)

classifier = KNeighborsClassifier(n_neighbors=7, metric='euclidean', p=2)
classifier.fit(train_dataNoRes, train_dataRes)
test_resultPredictions = classifier.predict(test_dataNoRes)

confusionMatrix = confusion_matrix(test_dataRes, test_resultPredictions)
print(confusionMatrix)

f1Score = f1_score(test_dataRes, test_resultPredictions)
print(f'F1 Score: {f1Score}')
accuracyScore = accuracy_score(test_dataRes, test_resultPredictions)
print(f'Accuracy Score: {accuracyScore}')