"""Tests for scikit-learn integration"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis.strategies import data as hypothesis_callback
from hypothesis.strategies import floats, integers, just, sampled_from

import treelite

from .hypothesis_util import (
    standard_classification_datasets,
    standard_regression_datasets,
    standard_settings,
)
from .util import has_pandas, to_categorical

try:
    from sklearn.dummy import DummyClassifier, DummyRegressor
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
    # Skip this test suite if scikit-learn is not installed
    pytest.skip("scikit-learn not installed; skipping", allow_module_level=True)

pytestmark = pytest.mark.skipif(not has_pandas(), reason="Pandas required")


@given(
    clazz=sampled_from(
        [
            RandomForestRegressor,
            ExtraTreesRegressor,
            GradientBoostingRegressor,
            HistGradientBoostingRegressor,
        ]
    ),
    n_estimators=integers(min_value=5, max_value=10),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_skl_regressor(clazz, n_estimators, callback):
    """Scikit-learn regressor"""
    if clazz in [RandomForestRegressor, ExtraTreesRegressor]:
        n_targets = callback.draw(integers(min_value=1, max_value=3))
    else:
        n_targets = callback.draw(just(1))
    X, y = callback.draw(standard_regression_datasets(n_targets=just(n_targets)))
    kwargs = {"max_depth": 3, "random_state": 0}
    if clazz == HistGradientBoostingRegressor:
        kwargs["max_iter"] = n_estimators
    else:
        kwargs["n_estimators"] = n_estimators
    if clazz in [GradientBoostingRegressor, HistGradientBoostingRegressor]:
        kwargs["learning_rate"] = callback.draw(floats(min_value=0.01, max_value=1.0))
    else:
        kwargs["n_jobs"] = -1
    if clazz == GradientBoostingClassifier:
        kwargs["init"] = callback.draw(
            sampled_from([None, DummyRegressor(strategy="mean"), "zero"])
        )
    clf = clazz(**kwargs)
    clf.fit(X, y)

    tl_model = treelite.sklearn.import_model(clf)
    out_pred = treelite.gtil.predict(tl_model, X)
    if n_targets > 1:
        expected_pred = np.transpose(clf.predict(X)[:, :, np.newaxis], axes=(1, 0, 2))
    else:
        expected_pred = clf.predict(X).reshape((1, X.shape[0], -1))
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
    dataset=standard_classification_datasets(
        n_classes=integers(min_value=2, max_value=5),
    ),
    n_estimators=integers(min_value=3, max_value=10),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_skl_classifier(clazz, dataset, n_estimators, callback):
    """Scikit-learn binary classifier"""
    X, y = dataset
    n_classes = len(np.unique(y))
    kwargs = {"max_depth": 3, "random_state": 0}
    if clazz == HistGradientBoostingClassifier:
        kwargs["max_iter"] = n_estimators
    else:
        kwargs["n_estimators"] = n_estimators
    if clazz in [GradientBoostingClassifier, HistGradientBoostingClassifier]:
        kwargs["learning_rate"] = callback.draw(floats(min_value=0.01, max_value=1.0))
    else:
        kwargs["n_jobs"] = -1
    if clazz == GradientBoostingClassifier:
        kwargs["init"] = callback.draw(
            sampled_from([None, DummyClassifier(strategy="prior"), "zero"])
        )
    clf = clazz(**kwargs)
    clf.fit(X, y)

    tl_model = treelite.sklearn.import_model(clf)

    if clazz == GradientBoostingClassifier and kwargs["init"] == "zero":
        base_scores = tl_model.get_header_accessor().get_field("base_scores")
        expected_base_scores_shape = (clf.n_classes_ if clf.n_classes_ > 2 else 1,)
        assert base_scores.shape == expected_base_scores_shape
        np.testing.assert_equal(base_scores, np.zeros(expected_base_scores_shape))

    out_prob = treelite.gtil.predict(tl_model, X)
    expected_prob = clf.predict_proba(X)[np.newaxis, :, :]
    if (
        clazz in [GradientBoostingClassifier, HistGradientBoostingClassifier]
        and n_classes == 2
    ):
        expected_prob = expected_prob[:, :, 1:]
    np.testing.assert_almost_equal(out_prob, expected_prob, decimal=5)


@given(dataset=standard_regression_datasets())
@settings(**standard_settings())
def test_skl_converter_iforest(dataset):
    """Scikit-learn isolation forest"""
    X, _ = dataset
    clf = IsolationForest(
        max_samples=64,
        n_estimators=10,
        n_jobs=-1,
        random_state=0,
    )
    clf.fit(X)
    expected_pred = clf._compute_chunked_score_samples(X)  # pylint: disable=W0212
    expected_pred = expected_pred.reshape((1, -1, 1))

    tl_model = treelite.sklearn.import_model(clf)
    out_pred = treelite.gtil.predict(tl_model, X)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=2)


@given(
    dataset=standard_classification_datasets(
        n_classes=integers(min_value=2, max_value=4),
    ),
    num_boost_round=integers(min_value=5, max_value=20),
    use_categorical=sampled_from([True, False]),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
@pytest.mark.xfail(
    reason="HistGradientBoostingClassifier parser doesn't work with scikit-learn 1.4.0+"
)
def test_skl_hist_gradient_boosting_with_categorical(
    dataset, num_boost_round, use_categorical, callback
):
    """Scikit-learn HistGradientBoostingClassifier, with categorical splits"""
    X, y = dataset
    if use_categorical:
        n_categorical = callback.draw(integers(min_value=1, max_value=X.shape[1]))
        df, X_pred = to_categorical(X, n_categorical=n_categorical, invalid_frac=0.0)
        cat_col_bitmap = df.dtypes == "category"
        categorical_features = [
            i for i in range(len(cat_col_bitmap)) if cat_col_bitmap[i]
        ]
    else:
        df = pd.DataFrame(X)
        categorical_features = None
        X_pred = X.copy()

    clf = HistGradientBoostingClassifier(
        max_iter=num_boost_round,
        categorical_features=categorical_features,
        early_stopping=False,
        max_depth=1,
    )
    clf.fit(df, y)

    tl_model = treelite.sklearn.import_model(clf)
    out_pred = treelite.gtil.predict(tl_model, X_pred)
    expected_pred = clf.predict_proba(X_pred)[np.newaxis, :, 1:]
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=4)
