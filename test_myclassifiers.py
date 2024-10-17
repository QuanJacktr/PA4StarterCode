# pylint: skip-file
import numpy as np
from scipy import stats

from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier

# note: order is actual/received student value, expected/solution
def test_simple_linear_regression_classifier_fit():
    np.random.seed(0)
    X_train = [[x] for x in range(100)]
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]

    def discretizer(x):
        if x >= 100:
            return "high"
        return "low"

    classifier = MySimpleLinearRegressionClassifier(discretizer)
    classifier.fit(X_train, y_train)

    assert isinstance(classifier.regressor, MySimpleLinearRegressor) # TODO: fix this
    assert abs(classifier.regressor.slope - 2) < 0.2
    assert abs(classifier.regressor.intercept) < 5

def test_simple_linear_regression_classifier_predict():
    np.random.seed(0)
    X_train = [[x] for x in range(100)]
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]

    def discretizer(x):
        if x >= 100:
            return "high"
        return "low"

    classifier = MySimpleLinearRegressionClassifier(discretizer)
    classifier.fit(X_train, y_train)
    
    X_test = [[25], [75], [150]]
    y_pred = classifier.predict(X_test)

    assert y_pred == ["low", "low", "high"]# TODO: fix this

def test_kneighbors_classifier_kneighbors():
    X_train = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train = ["bad", "bad", "good", "good"]

    knn = MyKNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    X_test = [[0.33, 1]]
    distances, neighbor_indices = knn.kneighbors(X_test)

    assert len(distances[0]) == 3
    assert len(neighbor_indices[0]) == 3
    assert neighbor_indices[0] == [0, 1, 2] # TODO: fix this

def test_kneighbors_classifier_predict():
    X_train = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train = ["bad", "bad", "good", "good"]

    knn = MyKNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    X_test = [[0.33, 1]]
    y_pred = knn.kneighbors(X_test)

    assert y_pred == ["bad"] # TODO: fix this

def test_dummy_classifier_fit():
    X_train = [[1], [2], [3], [4], [5]]
    y_train = ["A", "B", "A", "A", "B"]

    dummy = MyDummyClassifier()
    dummy.fit(X_train, y_train)

    assert dummy.most_common_label == "A" # TODO: fix this

def test_dummy_classifier_predict():
    X_train = [[1], [2], [3], [4], [5]]
    y_train = ["A", "B", "A", "A", "B"]

    dummy = MyDummyClassifier()
    dummy.fit(X_train, y_train)

    X_test = [[6], [7], [8]]
    y_pred = dummy.predict(X_test)

    assert y_pred == ["A", "A", "A"] # TODO: fix this
