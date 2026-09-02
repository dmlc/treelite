"""
Test coverage for scikit-learn tree models trained with data with missing values.
Adapted from https://github.com/scikit-learn/scikit-learn/blob/1.9.0/sklearn/tree/tests/test_tree.py
"""

import numpy as np
import pytest

import treelite

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
except ImportError:
    pytest.skip("scikit-learn not installed; skipping", allow_module_level=True)


@pytest.mark.parametrize("criterion", ["squared_error"])
def test_heuristic_default_direction(criterion):
    """
    When training data contains no missing values, scikit-learn uses the following
    heuristic for predicting with a missing value.
    For each binary test node,
    * Choose the left or right child node, whichever has the higher sample count.
    * If the two child nodes have identical sample counts, choose the right child node.
    """
    X = np.array([[0, 1, 2, 3, 8, 9, 11, 12, 15]]).T
    y = np.array([0.1, 0.2, 0.3, 0.2, 1.4, 1.4, 1.5, 1.6, 2.6])

    regr = RandomForestRegressor(
        random_state=42,
        max_depth=1,
        criterion=criterion,
        n_estimators=1,
        bootstrap=False,
    )
    regr.fit(X, y)
    tl_model = treelite.sklearn.import_model(regr)

    # There should be a single test node, with the test threshold being somewhere between 3 and 8.
    threshold = regr.estimators_[0].tree_.threshold[0]
    assert 3 < threshold < 8

    # Goes to right node because it has the most data points
    y_pred = regr.predict([[np.nan]])
    np.testing.assert_allclose(y_pred, [np.mean(y[-5:])])
    np.testing.assert_allclose(
        treelite.gtil.predict(tl_model, np.array([[np.nan]])).squeeze(),
        y_pred,
    )

    # Equal number of elements in both nodes
    X_equal = X[:-1]
    y_equal = y[:-1]

    regr = RandomForestRegressor(
        random_state=42,
        max_depth=1,
        criterion=criterion,
        n_estimators=1,
        bootstrap=False,
    )
    regr.fit(X_equal, y_equal)
    tl_model = treelite.sklearn.import_model(regr)

    # Goes to right node because the implementation sets:
    # missing_go_to_left = n_left > n_right, which is False
    y_pred = regr.predict([[np.nan]])
    np.testing.assert_allclose(y_pred, [np.mean(y_equal[-4:])])
    np.testing.assert_allclose(
        treelite.gtil.predict(tl_model, np.array([[np.nan]])).squeeze(),
        y_pred,
    )


@pytest.mark.parametrize("criterion", ["entropy", "gini"])
def test_best_splitter_three_classes(criterion):
    """
    Test the following scenario with a tree stump:
    Training data has three class labels: 0, 1, 2.
    All missing values are assigned class 0; all non-missing values are assigned
    either class 1 or 2.
    At predict time, the model should classify missing values as class 0.
    """
    missing_values_class = 0
    X = np.array([[np.nan] * 4 + [0, 1, 2, 3, 8, 9, 11, 12]]).T
    y = np.array([missing_values_class] * 4 + [1] * 4 + [2] * 4)
    clf = RandomForestClassifier(
        random_state=42,
        max_depth=2,
        criterion=criterion,
        n_estimators=1,
        bootstrap=False,
    )
    clf.fit(X, y)
    tl_model = treelite.sklearn.import_model(clf)

    X_test = np.array([[np.nan, 3, 12]]).T
    y_pred = clf.predict(X_test)
    np.testing.assert_array_equal(y_pred, [missing_values_class, 1, 2])

    tl_pred = treelite.gtil.predict(tl_model, X_test)
    tl_pred = np.argmax(tl_pred[:, 0, :], axis=1)
    np.testing.assert_array_equal(tl_pred, y_pred)


@pytest.mark.parametrize("criterion", ["entropy", "gini"])
def test_best_splitter_to_left(criterion):
    """
    Test the following scenario with a tree stump:
    Training data has two class labels: 0, 1.
    Class 0 gets only missing values.
    Class 1 gets only non-missing values.
    At predict time, the model should classify missing values as class 0.
    """
    X = np.array([[np.nan] * 4 + [0, 1, 2, 3, 4, 5]]).T
    y = np.array([0] * 4 + [1] * 6)

    clf = RandomForestClassifier(
        random_state=42,
        max_depth=2,
        criterion=criterion,
        n_estimators=1,
        bootstrap=False,
    )
    clf.fit(X, y)
    tl_model = treelite.sklearn.import_model(clf)

    X_test = np.array([[np.nan, 5, np.nan]]).T
    y_pred = clf.predict(X_test)
    np.testing.assert_array_equal(y_pred, [0, 1, 0])

    tl_pred = treelite.gtil.predict(tl_model, X_test)
    tl_pred = np.argmax(tl_pred[:, 0, :], axis=1)
    np.testing.assert_array_equal(tl_pred, y_pred)


@pytest.mark.parametrize("criterion", ["entropy", "gini"])
def test_best_splitter_to_right(criterion):
    """
    Test the following scenario with a tree stump:
    Training data has two class labels: 0, 1.
    Class 0 gets only non-missing values.
    Class 1 gets a mix of missing values and non-missing values.
    At predict time, the model should classify missing values as class 1.
    """
    X = np.array([[np.nan] * 4 + [0, 1, 2, 3, 4, 5]]).T
    y = np.array([1] * 4 + [0] * 4 + [1] * 2)

    clf = RandomForestClassifier(
        random_state=42,
        max_depth=2,
        criterion=criterion,
        n_estimators=1,
        bootstrap=False,
    )
    clf.fit(X, y)
    tl_model = treelite.sklearn.import_model(clf)

    X_test = np.array([[np.nan, 1.2, 4.8]]).T
    y_pred = clf.predict(X_test)
    np.testing.assert_array_equal(y_pred, [1, 0, 1])

    tl_pred = treelite.gtil.predict(tl_model, X_test)
    tl_pred = np.argmax(tl_pred[:, 0, :], axis=1)
    np.testing.assert_array_equal(tl_pred, y_pred)


@pytest.mark.parametrize("criterion", ["entropy", "gini"])
def test_best_splitter_missing_both_classes_has_nan(criterion):
    """
    Test the following scenario with a tree stump:
    Training data has two class labels: 0, 1.
    Each class gets 4 non-missing values and 1 missing value.
    """
    X = np.array([[1, 2, 3, 5, np.nan, 10, 20, 30, 60, np.nan]]).T
    y = np.array([0] * 5 + [1] * 5)

    clf = RandomForestClassifier(
        random_state=42,
        max_depth=1,
        criterion=criterion,
        n_estimators=1,
        bootstrap=False,
    )
    clf.fit(X, y)
    tl_model = treelite.sklearn.import_model(clf)

    X_test = np.array([[np.nan, 2.3, 34.2]]).T
    y_pred = clf.predict(X_test)

    # Missing value goes to the class at the right (here 1) because the implementation
    # searches right first.
    np.testing.assert_array_equal(y_pred, [1, 0, 1])

    tl_pred = treelite.gtil.predict(tl_model, X_test)
    tl_pred = np.argmax(tl_pred[:, 0, :], axis=1)
    np.testing.assert_array_equal(tl_pred, y_pred)
