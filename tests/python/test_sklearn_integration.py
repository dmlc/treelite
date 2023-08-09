"""Tests for scikit-learn integration"""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import data as hypothesis_callback
from hypothesis.strategies import floats, integers, just, sampled_from

import treelite
from treelite import TreeliteError

from .hypothesis_util import (
    standard_classification_datasets,
    standard_regression_datasets,
    standard_settings,
)

try:
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        IsolationForest,
        RandomForestClassifier,
        RandomForestRegressor,
    )
except ImportError:
    # skip this test suite if scikit-learn is not installed
    pytest.skip("scikit-learn not installed; skipping", allow_module_level=True)


@given(
    clazz=sampled_from(
        [
            RandomForestRegressor,
            ExtraTreesRegressor,
            GradientBoostingRegressor,
            HistGradientBoostingRegressor,
        ]
    ),
    dataset=standard_regression_datasets(),
    n_estimators=integers(min_value=5, max_value=50),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_skl_regressor(clazz, dataset, n_estimators, callback):
    """Scikit-learn regressor"""
    X, y = dataset
    kwargs = {"max_depth": 3, "random_state": 0}
    if clazz == GradientBoostingRegressor:
        kwargs["init"] = "zero"
    if clazz == HistGradientBoostingRegressor:
        kwargs["max_iter"] = n_estimators
    else:
        kwargs["n_estimators"] = n_estimators
    if clazz in [GradientBoostingClassifier, HistGradientBoostingClassifier]:
        kwargs["learning_rate"] = callback.draw(floats(min_value=0.01, max_value=1.0))
    clf = clazz(**kwargs)
    clf.fit(X, y)
    expected_pred = clf.predict(X)

    tl_model = treelite.sklearn.import_model(clf)
    out_pred = treelite.gtil.predict(tl_model, X)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=3)


@given(
    clazz=sampled_from(
        [
            RandomForestClassifier,
            ExtraTreesClassifier,
            GradientBoostingClassifier,
            HistGradientBoostingClassifier,
        ]
    ),
    dataset=standard_classification_datasets(n_classes=just(2)),
    n_estimators=integers(min_value=5, max_value=50),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_skl_binary_classifier(clazz, dataset, n_estimators, callback):
    """Scikit-learn binary classifier"""
    X, y = dataset
    kwargs = {"max_depth": 3, "random_state": 0}
    if clazz == GradientBoostingClassifier:
        kwargs["init"] = "zero"
    if clazz == HistGradientBoostingClassifier:
        kwargs["max_iter"] = n_estimators
    else:
        kwargs["n_estimators"] = n_estimators
    if clazz in [GradientBoostingClassifier, HistGradientBoostingClassifier]:
        kwargs["learning_rate"] = callback.draw(floats(min_value=0.01, max_value=1.0))
    clf = clazz(**kwargs)
    clf.fit(X, y)
    expected_prob = clf.predict_proba(X)[:, 1]

    tl_model = treelite.sklearn.import_model(clf)
    out_prob = treelite.gtil.predict(tl_model, X)
    np.testing.assert_almost_equal(out_prob, expected_prob, decimal=5)


@given(
    clazz=sampled_from(
        [
            RandomForestClassifier,
            ExtraTreesClassifier,
            GradientBoostingClassifier,
            HistGradientBoostingClassifier,
        ]
    ),
    dataset=standard_classification_datasets(
        n_classes=integers(min_value=3, max_value=10), n_informative=just(5)
    ),
    n_estimators=integers(min_value=5, max_value=50),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_skl_multiclass_classifier(clazz, dataset, n_estimators, callback):
    """Scikit-learn multi-class classifier"""
    X, y = dataset
    kwargs = {"max_depth": 3, "random_state": 0}
    if clazz == GradientBoostingClassifier:
        kwargs["init"] = "zero"
    if clazz == HistGradientBoostingClassifier:
        kwargs["max_iter"] = n_estimators
    else:
        kwargs["n_estimators"] = n_estimators
    if clazz in [GradientBoostingClassifier, HistGradientBoostingClassifier]:
        kwargs["learning_rate"] = callback.draw(floats(min_value=0.01, max_value=1.0))
    clf = clazz(**kwargs)
    clf.fit(X, y)
    expected_prob = clf.predict_proba(X)

    if clazz == HistGradientBoostingClassifier:
        with pytest.raises(
            TreeliteError,
            match=r".*HistGradientBoostingClassifier with n_classes > 2 is not supported yet.*",
        ):
            treelite.sklearn.import_model(clf)
    else:
        tl_model = treelite.sklearn.import_model(clf)
        out_prob = treelite.gtil.predict(tl_model, X)
        np.testing.assert_almost_equal(out_prob, expected_prob, decimal=5)


@given(standard_regression_datasets())
@settings(**standard_settings())
def test_skl_converter_iforest(dataset):
    """Scikit-learn isolation forest"""
    X, _ = dataset
    clf = IsolationForest(max_samples=64, random_state=0, n_estimators=10)
    clf.fit(X)
    expected_pred = clf._compute_chunked_score_samples(X)  # pylint: disable=W0212

    tl_model = treelite.sklearn.import_model(clf)
    out_pred = treelite.gtil.predict(tl_model, X)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=2)


def test_skl_hist_gradient_boosting_with_categorical():
    """Scikit-learn HistGradientBoostingClassifier, with categorical splits"""
    # We don't yet support HistGradientBoostingClassifier with categorical splits
    # So make sure that an exception is thrown properly
    rng = np.random.RandomState(0)
    n_samples = 1000
    f1 = rng.rand(n_samples)
    f2 = rng.randint(4, size=n_samples)
    X = np.c_[f1, f2]
    y = np.zeros(shape=n_samples)
    y[X[:, 1] % 2 == 0] = 1
    clf = HistGradientBoostingClassifier(max_iter=20, categorical_features=[1])
    clf.fit(X, y)
    np.testing.assert_array_equal(clf.is_categorical_, [False, True])

    with pytest.raises(
        NotImplementedError, match=r"Categorical splits are not yet supported.*"
    ):
        treelite.sklearn.import_model(clf)
