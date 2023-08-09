"""Tests for LightGBM integration"""
import os

import numpy as np
import pytest
import scipy
from hypothesis import given, settings
from hypothesis.strategies import integers, just, sampled_from

import treelite

from .hypothesis_util import (
    standard_classification_datasets,
    standard_regression_datasets,
    standard_settings,
)
from .metadata import dataset_db
from .util import TemporaryDirectory, load_txt

try:
    import lightgbm as lgb
except ImportError:
    # skip this test suite if LightGBM is not installed
    pytest.skip("LightGBM not installed; skipping", allow_module_level=True)


try:
    from sklearn.datasets import load_svmlight_file
    from sklearn.model_selection import train_test_split
except ImportError:
    # skip this test suite if scikit-learn is not installed
    pytest.skip("scikit-learn not installed; skipping", allow_module_level=True)


@given(
    objective=sampled_from(["regression", "regression_l1", "huber"]),
    reg_sqrt=sampled_from([True, False]),
    dataset=standard_regression_datasets(),
)
@settings(**standard_settings())
def test_lightgbm_regression(objective, reg_sqrt, dataset):
    # pylint: disable=too-many-locals
    """Test LightGBM regressor"""
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    dtrain = lgb.Dataset(X_train, y_train, free_raw_data=False)
    dtest = lgb.Dataset(X_test, y_test, reference=dtrain, free_raw_data=False)
    param = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": objective,
        "reg_sqrt": reg_sqrt,
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.05,
    }
    lgb_model = lgb.train(
        param,
        dtrain,
        num_boost_round=10,
        valid_sets=[dtrain, dtest],
        valid_names=["train", "test"],
    )
    expected_pred = lgb_model.predict(X_test)

    with TemporaryDirectory() as tmpdir:
        lgb_model_path = os.path.join(tmpdir, "boston_lightgbm.txt")
        lgb_model.save_model(lgb_model_path)

        tl_model = treelite.Model.load(lgb_model_path, model_format="lightgbm")
        out_pred = treelite.gtil.predict(tl_model, X_test)
        np.testing.assert_almost_equal(out_pred, expected_pred, decimal=4)


@given(
    dataset=standard_classification_datasets(n_classes=just(2)),
    objective=sampled_from(["binary", "xentlambda", "xentropy"]),
    num_boost_round=integers(min_value=5, max_value=50),
)
@settings(**standard_settings())
def test_lightgbm_binary_classification(dataset, objective, num_boost_round):
    # pylint: disable=too-many-locals
    """Test LightGBM binary classifier"""
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    dtrain = lgb.Dataset(X_train, y_train, free_raw_data=False)
    dtest = lgb.Dataset(X_test, y_test, reference=dtrain, free_raw_data=False)
    param = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": objective,
        "metric": "auc",
        "num_leaves": 7,
        "learning_rate": 0.1,
    }
    lgb_model = lgb.train(
        param,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dtest],
        valid_names=["train", "test"],
    )
    expected_prob = lgb_model.predict(X_test)

    with TemporaryDirectory() as tmpdir:
        lgb_model_path = os.path.join(tmpdir, "breast_cancer_lightgbm.txt")
        lgb_model.save_model(lgb_model_path)

        tl_model = treelite.Model.load(lgb_model_path, model_format="lightgbm")
        out_prob = treelite.gtil.predict(tl_model, X_test)
        np.testing.assert_almost_equal(out_prob, expected_prob, decimal=5)


@given(
    dataset=standard_classification_datasets(
        n_classes=integers(min_value=3, max_value=10), n_informative=just(5)
    ),
    objective=sampled_from(["multiclass", "multiclassova"]),
    boosting_type=sampled_from(["gbdt", "rf"]),
    num_boost_round=integers(min_value=5, max_value=50),
)
@settings(**standard_settings())
def test_lightgbm_multiclass_classification(
    dataset, objective, boosting_type, num_boost_round
):
    # pylint: disable=too-many-locals
    """Test LightGBM multi-class classifier"""
    X, y = dataset
    num_class = np.max(y) + 1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    dtrain = lgb.Dataset(X_train, y_train, free_raw_data=False)
    dtest = lgb.Dataset(X_test, y_test, reference=dtrain, free_raw_data=False)
    param = {
        "task": "train",
        "boosting": boosting_type,
        "objective": objective,
        "metric": "multi_logloss",
        "num_class": num_class,
        "num_leaves": 31,
        "learning_rate": 0.05,
    }
    if boosting_type == "rf":
        param.update({"bagging_fraction": 0.8, "bagging_freq": 1})
    lgb_model = lgb.train(
        param,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dtest],
        valid_names=["train", "test"],
    )
    expected_pred = lgb_model.predict(X_test)

    with TemporaryDirectory() as tmpdir:
        lgb_model_path = os.path.join(tmpdir, "iris_lightgbm.txt")
        lgb_model.save_model(lgb_model_path)

        tl_model = treelite.Model.load(lgb_model_path, model_format="lightgbm")
        out_pred = treelite.gtil.predict(tl_model, X_test)
        np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


def test_lightgbm_categorical_data():
    """Test LightGBM with toy categorical data"""
    dataset = "toy_categorical"
    lgb_model_path = dataset_db[dataset].model
    tl_model = treelite.Model.load(lgb_model_path, model_format="lightgbm")

    X, _ = load_svmlight_file(dataset_db[dataset].dtest, zero_based=True)
    expected_pred = load_txt(dataset_db[dataset].expected_margin)
    out_pred = treelite.gtil.predict(tl_model, X.toarray())
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


def test_lightgbm_sparse_ranking_model(tmpdir):
    """Generate a LightGBM ranking model with highly sparse data."""
    rng = np.random.default_rng(seed=2020)
    X = scipy.sparse.random(
        m=10, n=206947, format="csr", dtype=np.float64, random_state=0, density=0.0001
    )
    X.data = rng.standard_normal(size=X.data.shape[0], dtype=np.float64)
    y = rng.integers(low=0, high=5, size=X.shape[0])

    params = {
        "objective": "lambdarank",
        "num_leaves": 32,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "min_gain_to_split": 0.0,
        "learning_rate": 1.0,
        "min_data_in_leaf": 1,
    }

    lgb_model_path = os.path.join(tmpdir, "sparse_ranking_lightgbm.txt")

    dtrain = lgb.Dataset(X, label=y, group=[X.shape[0]])
    lgb_model = lgb.train(params, dtrain, num_boost_round=1)
    lgb_out = lgb_model.predict(X)
    lgb_model.save_model(lgb_model_path)

    tl_model = treelite.Model.load(lgb_model_path, model_format="lightgbm")

    # GTIL doesn't yet support sparse matrix; so use NaN to represent missing values
    Xa = X.toarray()
    Xa[Xa == 0] = "nan"
    out = treelite.gtil.predict(tl_model, Xa)

    np.testing.assert_almost_equal(out, lgb_out)


def test_lightgbm_sparse_categorical_model():
    """Test LightGBM with high-cardinality categorical features"""
    dataset = "sparse_categorical"
    lgb_model_path = dataset_db[dataset].model
    tl_model = treelite.Model.load(lgb_model_path, model_format="lightgbm")

    X, _ = load_svmlight_file(
        dataset_db[dataset].dtest, zero_based=True, n_features=tl_model.num_feature
    )
    expected_pred = load_txt(dataset_db[dataset].expected_margin)
    # GTIL doesn't yet support sparse matrix; so use NaN to represent missing values
    Xa = X.toarray()
    Xa[Xa == 0] = "nan"
    out_pred = treelite.gtil.predict(tl_model, Xa, pred_margin=True)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)
