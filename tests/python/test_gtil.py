"""
Tests for General Tree Inference Library (GTIL). The tests ensure that GTIL produces correct
prediction results for a variety of tree models.
"""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import data as hypothesis_callback
from hypothesis.strategies import integers, just, sampled_from

import treelite

from .hypothesis_util import (
    standard_classification_datasets,
    standard_regression_datasets,
    standard_settings,
)

try:
    import xgboost as xgb
except ImportError:
    # skip this test suite if XGBoost is not installed
    pytest.skip("XGBoost not installed; skipping", allow_module_level=True)


try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    # skip this test suite if scikit-learn is not installed
    pytest.skip("scikit-learn not installed; skipping", allow_module_level=True)


@given(
    dataset=standard_regression_datasets(),
    predict_type=sampled_from(["leaf_id", "score_per_tree"]),
    num_boost_round=integers(min_value=5, max_value=50),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_predict_special_with_regressor(
    dataset, predict_type, num_boost_round, callback
):
    # pylint: disable=too-many-locals
    """Test predict_leaf / predict_per_tree with XGBoost regressor"""
    X, y = dataset
    sample_size = callback.draw(integers(min_value=1, max_value=X.shape[0]))
    dtrain = xgb.DMatrix(X, label=y)
    param = {
        "max_depth": 8,
        "eta": 1,
        "verbosity": 0,
        "objective": "reg:squarederror",
        "base_score": 0,
    }
    xgb_model = xgb.train(
        param,
        dtrain,
        num_boost_round=num_boost_round,
    )
    model: treelite.Model = treelite.Model.from_xgboost(xgb_model)
    assert model.num_tree == num_boost_round

    X_sample = X[0:sample_size]
    if predict_type == "leaf_id":
        leaf_pred = treelite.gtil.predict_leaf(model, X_sample)
        assert leaf_pred.shape == (sample_size, num_boost_round)
        xgb_leaf_pred = xgb_model.predict(xgb.DMatrix(X_sample), pred_leaf=True)
        assert np.array_equal(leaf_pred, xgb_leaf_pred)
    else:
        pred_per_tree = treelite.gtil.predict_per_tree(model, X_sample)
        assert pred_per_tree.shape == (sample_size, num_boost_round)
        pred = xgb_model.predict(xgb.DMatrix(X_sample), output_margin=True)
        np.testing.assert_almost_equal(np.sum(pred_per_tree, axis=1), pred, decimal=3)


@given(
    predict_type=sampled_from(["leaf_id", "score_per_tree"]),
    dataset=standard_classification_datasets(n_classes=just(2)),
    num_boost_round=integers(min_value=5, max_value=50),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_predict_special_with_binary_classifier(
    predict_type, dataset, num_boost_round, callback
):
    # pylint: disable=too-many-locals
    """Test predict_leaf / predict_per_tree with XGBoost binary classifier"""
    X, y = dataset
    sample_size = callback.draw(integers(min_value=1, max_value=X.shape[0]))

    dtrain = xgb.DMatrix(X, label=y)
    param = {
        "objective": "binary:logistic",
        "base_score": 0.5,
        "learning_rate": 0.1,
        "max_depth": 6,
    }
    xgb_model = xgb.train(
        param,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
    )
    model: treelite.Model = treelite.Model.from_xgboost(xgb_model)
    assert model.num_tree == num_boost_round

    X_sample = X[0:sample_size]
    if predict_type == "leaf_id":
        leaf_pred = treelite.gtil.predict_leaf(model, X_sample)
        assert leaf_pred.shape == (sample_size, num_boost_round)
        xgb_leaf_pred = xgb_model.predict(xgb.DMatrix(X_sample), pred_leaf=True)
        assert np.array_equal(leaf_pred, xgb_leaf_pred)
    else:
        pred_per_tree = treelite.gtil.predict_per_tree(model, X_sample)
        assert pred_per_tree.shape == (sample_size, num_boost_round)
        pred = xgb_model.predict(xgb.DMatrix(X_sample), output_margin=True)
        np.testing.assert_almost_equal(np.sum(pred_per_tree, axis=1), pred, decimal=3)


@given(
    predict_type=sampled_from(["leaf_id", "score_per_tree"]),
    dataset=standard_classification_datasets(
        n_classes=integers(min_value=3, max_value=10), n_informative=just(5)
    ),
    num_boost_round=integers(min_value=5, max_value=50),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_predict_special_with_multiclass_classifier_grove_per_class(
    predict_type, dataset, num_boost_round, callback
):
    # pylint: disable=too-many-locals
    """
    Test predict_leaf / predict_per_tree with XGBoost multiclass classifier
    (grove-per-class)
    """
    X, y = dataset
    sample_size = callback.draw(integers(min_value=1, max_value=X.shape[0]))
    num_class = np.max(y) + 1
    dtrain = xgb.DMatrix(X, label=y)
    param = {
        "max_depth": 10,
        "eta": 0.05,
        "num_class": num_class,
        "verbosity": 0,
        "objective": "multi:softprob",
        "metric": "mlogloss",
        "base_score": 0,
    }
    xgb_model = xgb.train(
        param,
        dtrain,
        num_boost_round=num_boost_round,
    )
    model: treelite.Model = treelite.Model.from_xgboost(xgb_model)
    assert model.num_tree == num_boost_round * num_class

    X_sample = X[0:sample_size]
    if predict_type == "leaf_id":
        leaf_pred = treelite.gtil.predict_leaf(model, X_sample)
        assert leaf_pred.shape == (sample_size, model.num_tree)
        xgb_leaf_pred = xgb_model.predict(xgb.DMatrix(X_sample), pred_leaf=True)
        assert np.array_equal(leaf_pred, xgb_leaf_pred)
    else:
        pred_per_tree = treelite.gtil.predict_per_tree(model, X_sample)
        assert pred_per_tree.shape == (sample_size, model.num_tree)
        sum_by_class = np.column_stack(
            tuple(
                np.sum(pred_per_tree[:, class_id::num_class], axis=1)
                for class_id in range(num_class)
            )
        )
        pred = xgb_model.predict(xgb.DMatrix(X_sample), output_margin=True)
        np.testing.assert_almost_equal(sum_by_class, pred, decimal=3)


@given(
    n_estimators=integers(min_value=10, max_value=50),
    dataset=standard_classification_datasets(
        n_classes=integers(min_value=3, max_value=10), n_informative=just(5)
    ),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_predict_per_tree_with_multiclass_classifier_vector_leaf(
    n_estimators, dataset, callback
):
    """Test predict_per_tree with Scikit-learn multi-class classifier (vector leaf)"""
    X, y = dataset
    num_class = np.max(y) + 1
    sample_size = callback.draw(integers(min_value=1, max_value=X.shape[0]))
    kwargs = {"max_depth": 3, "random_state": 0, "n_estimators": n_estimators}
    clf = RandomForestClassifier(**kwargs)
    clf.fit(X, y)
    model: treelite.Model = treelite.sklearn.import_model(clf)

    X_sample = X[0:sample_size]
    pred_per_tree = treelite.gtil.predict_per_tree(model, X_sample)
    assert pred_per_tree.shape == (sample_size, n_estimators, num_class)
    avg_by_class = np.sum(pred_per_tree, axis=1) / n_estimators
    expected_prob = clf.predict_proba(X_sample)
    np.testing.assert_almost_equal(avg_by_class, expected_prob, decimal=3)
