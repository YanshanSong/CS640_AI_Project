from numpy.random import normal
from sklearn import neighbors
import numpy as np

def initData():
    x1 = normal(50, 6, 200)
    y1 = normal(5, 0.5, 200)

    x2 = normal(30, 6, 200)
    y2 = normal(4, 0.5, 200)

    x3 = normal(45, 6, 200)
    y3 = normal(2.5, 0.5, 200)
    #initialize data

    x_val = np.concatenate((x1, x2, x3))
    y_val = np.concatenate((y1, y2, y3))
    x_diff = max(x_val) - min(x_val)
    y_diff = max(y_val) - min(y_val)
    x_normalized = [x / (x_diff) for x in x_val]
    y_normalized = [y / (y_diff) for y in y_val]
    xy_train_normalized = list(zip(x_normalized, y_normalized))

    labels_train = [1] * 200 + [2] * 200 + [3] * 200

    x1_test = normal(50, 6, 100)
    y1_test = normal(5, 0.5, 100)

    x2_test = normal(30, 6, 100)
    y2_test = normal(4, 0.5, 100)

    x3_test = normal(45, 6, 100)
    y3_test = normal(2.5, 0.5, 100)

    xy_test_normalized = list(zip(np.concatenate((x1_test, x2_test, x3_test)) / x_diff, \
                             np.concatenate((y1_test, y2_test, y3_test)) / y_diff))

    labels_test = [1] * 100 + [2] * 100 + [3] * 100

    return xy_train_normalized, labels_train, xy_test_normalized, labels_test


def knnModel(X_train, y_train, X_Test, y_test):
    model = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=1, p=2, metric='minkowski')
    model.fit(X_train, y_train)
    score = model.score(X_Test, y_test)
    print(score)

X_train, y_train, X_Test, y_test = initData()

knnModel(X_train, y_train, X_Test, y_test)

