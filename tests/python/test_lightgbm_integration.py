"""Tests for LightGBM integration"""

import pathlib

import numpy as np
import pytest
import scipy

import treelite

try:
    from hypothesis import given, settings
    from hypothesis.strategies import data as hypothesis_callback
    from hypothesis.strategies import integers, just, sampled_from
except ImportError:
    pytest.skip("hypothesis not installed; skipping", allow_module_level=True)

try:
    import lightgbm as lgb
except ImportError:
    pytest.skip("LightGBM not installed; skipping", allow_module_level=True)

try:
    from sklearn.datasets import load_svmlight_file
except ImportError:
    pytest.skip("scikit-learn not installed; skipping", allow_module_level=True)

from .hypothesis_util import (
    standard_classification_datasets,
    standard_regression_datasets,
    standard_settings,
)
from .metadata import dataset_db
from .util import TemporaryDirectory, load_txt, to_categorical


@given(
    objective=sampled_from(["regression", "regression_l1", "huber"]),
    reg_sqrt=sampled_from([True, False]),
    dataset=standard_regression_datasets(),
    use_categorical=sampled_from([True, False]),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_lightgbm_regression(
    objective,
    reg_sqrt,
    dataset,
    use_categorical,
    callback,
):
    # pylint: disable=too-many-locals
    """Test LightGBM regressor"""
    X, y = dataset
    if use_categorical:
        n_categorical = callback.draw(integers(min_value=1, max_value=X.shape[1]))
        df, X_pred = to_categorical(X, n_categorical=n_categorical, invalid_frac=0.1)
        dtrain = lgb.Dataset(
            df, label=y, free_raw_data=False, categorical_feature="auto"
        )
    else:
        dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
        X_pred = X.copy()

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
        valid_sets=[dtrain],
        valid_names=["train"],
    )
    expected_pred = lgb_model.predict(X_pred).reshape((-1, 1, 1))

    with TemporaryDirectory() as tmpdir:
        lgb_model_path = pathlib.Path(tmpdir) / "lightgbm_model.txt"
        lgb_model.save_model(lgb_model_path)

        tl_model = treelite.frontend.load_lightgbm_model(lgb_model_path)
        out_pred = treelite.gtil.predict(tl_model, X_pred)
        np.testing.assert_almost_equal(out_pred, expected_pred, decimal=4)


@given(
    dataset=standard_classification_datasets(n_classes=just(2)),
    objective=sampled_from(["binary", "xentlambda", "xentropy"]),
    num_boost_round=integers(min_value=5, max_value=50),
    use_categorical=sampled_from([True, False]),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_lightgbm_binary_classification(
    dataset,
    objective,
    num_boost_round,
    use_categorical,
    callback,
):
    # pylint: disable=too-many-locals
    """Test LightGBM binary classifier"""
    X, y = dataset
    if use_categorical:
        n_categorical = callback.draw(integers(min_value=1, max_value=X.shape[1]))
        df, X_pred = to_categorical(X, n_categorical=n_categorical, invalid_frac=0.1)
        dtrain = lgb.Dataset(
            df, label=y, free_raw_data=False, categorical_feature="auto"
        )
    else:
        dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
        X_pred = X.copy()

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
        valid_sets=[dtrain],
        valid_names=["train"],
    )
    expected_prob = lgb_model.predict(X_pred).reshape((-1, 1, 1))

    with TemporaryDirectory() as tmpdir:
        lgb_model_path = pathlib.Path(tmpdir) / "breast_cancer_lightgbm.txt"
        lgb_model.save_model(lgb_model_path)

        tl_model = treelite.frontend.load_lightgbm_model(lgb_model_path)
        out_prob = treelite.gtil.predict(tl_model, X_pred)
        np.testing.assert_almost_equal(out_prob, expected_prob, decimal=5)


@given(
    dataset=standard_classification_datasets(
        n_classes=integers(min_value=3, max_value=10), n_informative=just(5)
    ),
    objective=sampled_from(["multiclass", "multiclassova"]),
    boosting_type=sampled_from(["gbdt", "rf"]),
    num_boost_round=integers(min_value=5, max_value=50),
    use_categorical=sampled_from([True, False]),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_lightgbm_multiclass_classification(
    dataset,
    objective,
    boosting_type,
    num_boost_round,
    use_categorical,
    callback,
):
    # pylint: disable=too-many-locals,too-many-arguments
    """Test LightGBM multi-class classifier"""
    X, y = dataset
    num_class = np.max(y) + 1
    if use_categorical:
        n_categorical = callback.draw(integers(min_value=1, max_value=X.shape[1]))
        df, X_pred = to_categorical(X, n_categorical=n_categorical, invalid_frac=0.1)
        dtrain = lgb.Dataset(
            df, label=y, free_raw_data=False, categorical_feature="auto"
        )
    else:
        dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
        X_pred = X.copy()
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
        valid_sets=[dtrain],
        valid_names=["train"],
    )
    expected_pred = lgb_model.predict(X_pred).reshape((-1, 1, num_class))

    with TemporaryDirectory() as tmpdir:
        lgb_model_path = pathlib.Path(tmpdir) / "iris_lightgbm.txt"
        lgb_model.save_model(lgb_model_path)

        tl_model = treelite.frontend.load_lightgbm_model(lgb_model_path)
        out_pred = treelite.gtil.predict(tl_model, X_pred)
        np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


def test_lightgbm_categorical_data():
    """Test LightGBM with toy categorical data"""
    dataset = "toy_categorical"
    lgb_model_path = dataset_db[dataset].model
    tl_model = treelite.frontend.load_lightgbm_model(lgb_model_path)

    X, _ = load_svmlight_file(dataset_db[dataset].dtest, zero_based=True)
    expected_pred = load_txt(dataset_db[dataset].expected_margin).reshape((-1, 1, 1))
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

    lgb_model_path = pathlib.Path(tmpdir) / "sparse_ranking_lightgbm.txt"

    dtrain = lgb.Dataset(X, label=y, group=np.array([X.shape[0]], dtype=np.int32))
    lgb_model = lgb.train(params, dtrain, num_boost_round=1)
    lgb_out = lgb_model.predict(X).reshape((-1, 1, 1))
    lgb_model.save_model(lgb_model_path)

    tl_model = treelite.frontend.load_lightgbm_model(lgb_model_path)
    out = treelite.gtil.predict(tl_model, X, nthread=1)
    np.testing.assert_almost_equal(out, lgb_out)


def test_lightgbm_sparse_categorical_model():
    """Test LightGBM with high-cardinality categorical features"""
    dataset = "sparse_categorical"
    lgb_model_path = dataset_db[dataset].model
    tl_model = treelite.frontend.load_lightgbm_model(lgb_model_path)

    X, _ = load_svmlight_file(
        dataset_db[dataset].dtest, zero_based=True, n_features=tl_model.num_feature
    )
    expected_pred = load_txt(dataset_db[dataset].expected_margin).reshape((-1, 1, 1))
    out_pred = treelite.gtil.predict(tl_model, X, pred_margin=True)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)
