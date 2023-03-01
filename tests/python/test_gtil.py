"""
Tests for General Tree Inference Library (GTIL). The tests ensure that GTIL produces correct
prediction results for a variety of tree models.
"""
import json
import os

import numpy as np
import pytest
import scipy
from hypothesis import assume, given, settings
from hypothesis.strategies import data as hypothesis_callback
from hypothesis.strategies import integers, just, sampled_from
from sklearn.datasets import load_breast_cancer, load_iris, load_svmlight_file
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split

import treelite

from .hypothesis_util import (
    standard_classification_datasets,
    standard_regression_datasets,
    standard_settings,
)
from .metadata import dataset_db
from .util import TemporaryDirectory, load_txt

try:
    import xgboost as xgb
except ImportError:
    # skip this test suite if XGBoost is not installed
    pytest.skip("XGBoost not installed; skipping", allow_module_level=True)

try:
    import lightgbm as lgb
except ImportError:
    # skip this test suite if LightGBM is not installed
    pytest.skip("LightGBM not installed; skipping", allow_module_level=True)


@given(
    clazz=sampled_from(
        [RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor]
    ),
    dataset=standard_regression_datasets(),
)
@settings(**standard_settings())
def test_skl_regressor(clazz, dataset):
    """Scikit-learn regressor"""
    X, y = dataset
    kwargs = {"max_depth": 3, "random_state": 0, "n_estimators": 10}
    if clazz == GradientBoostingRegressor:
        kwargs["init"] = "zero"
    clf = clazz(**kwargs)
    clf.fit(X, y)
    expected_pred = clf.predict(X)

    tl_model = treelite.sklearn.import_model(clf)
    out_pred = treelite.gtil.predict(tl_model, X)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=4)


@pytest.mark.parametrize(
    "clazz", [RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier]
)
def test_skl_binary_classifier(clazz):
    """Scikit-learn binary classifier"""
    X, y = load_breast_cancer(return_X_y=True)
    kwargs = {"max_depth": 3, "random_state": 0, "n_estimators": 10}
    if clazz == GradientBoostingClassifier:
        kwargs["init"] = "zero"
    clf = clazz(**kwargs)
    clf.fit(X, y)
    expected_prob = clf.predict_proba(X)[:, 1]

    tl_model = treelite.sklearn.import_model(clf)
    out_prob = treelite.gtil.predict(tl_model, X)
    np.testing.assert_almost_equal(out_prob, expected_prob, decimal=5)


@pytest.mark.parametrize(
    "clazz", [RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier]
)
def test_skl_multiclass_classifier(clazz):
    """Scikit-learn multi-class classifier"""
    X, y = load_iris(return_X_y=True)
    kwargs = {"max_depth": 3, "random_state": 0, "n_estimators": 10}
    if clazz == GradientBoostingClassifier:
        kwargs["init"] = "zero"
    clf = clazz(**kwargs)
    clf.fit(X, y)
    expected_prob = clf.predict_proba(X)

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


@given(
    objective=sampled_from(
        [
            "reg:linear",
            "reg:squarederror",
            "reg:squaredlogerror",
            "reg:pseudohubererror",
        ]
    ),
    model_format=sampled_from(["binary", "json"]),
    num_parallel_tree=integers(min_value=1, max_value=10),
    dataset=standard_regression_datasets(),
)
@settings(**standard_settings())
def test_xgb_regression(objective, model_format, num_parallel_tree, dataset):
    # pylint: disable=too-many-locals
    """Test XGBoost with regression data"""
    X, y = dataset
    if objective == "reg:squaredlogerror":
        assume(np.all(y > -1))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {
        "max_depth": 8,
        "eta": 1,
        "verbosity": 0,
        "objective": objective,
        "num_parallel_tree": num_parallel_tree,
    }
    num_round = 10
    xgb_model = xgb.train(
        param,
        dtrain,
        num_boost_round=num_round,
        evals=[(dtrain, "train"), (dtest, "test")],
    )
    with TemporaryDirectory() as tmpdir:
        if model_format == "json":
            model_name = "model.json"
            model_path = os.path.join(tmpdir, model_name)
            xgb_model.save_model(model_path)
            tl_model = treelite.Model.load(
                filename=model_path, model_format="xgboost_json"
            )
        else:
            model_name = "model.model"
            model_path = os.path.join(tmpdir, model_name)
            xgb_model.save_model(model_path)
            tl_model = treelite.Model.load(filename=model_path, model_format="xgboost")
        assert (
            len(json.loads(tl_model.dump_as_json())["trees"])
            == num_round * num_parallel_tree
        )

        out_pred = treelite.gtil.predict(tl_model, X_test)
        expected_pred = xgb_model.predict(dtest)
        np.testing.assert_almost_equal(out_pred, expected_pred, decimal=3)


@pytest.mark.parametrize("num_parallel_tree", [1, 3, 5])
@pytest.mark.parametrize("model_format", ["binary", "json"])
@pytest.mark.parametrize("objective", ["multi:softmax", "multi:softprob"])
def test_xgb_iris(tmpdir, objective, model_format, num_parallel_tree):
    # pylint: disable=too-many-locals
    """Test XGBoost with Iris data (multi-class classification)"""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    num_class = 3
    param = {
        "max_depth": 6,
        "eta": 0.05,
        "num_class": num_class,
        "verbosity": 0,
        "objective": objective,
        "metric": "mlogloss",
        "num_parallel_tree": num_parallel_tree,
    }
    num_round = 3
    xgb_model = xgb.train(
        param,
        dtrain,
        num_boost_round=num_round,
        evals=[(dtrain, "train"), (dtest, "test")],
    )

    if model_format == "json":
        model_name = "iris.json"
        model_path = os.path.join(tmpdir, model_name)
        xgb_model.save_model(model_path)
        tl_model = treelite.Model.load(filename=model_path, model_format="xgboost_json")
    else:
        model_name = "iris.model"
        model_path = os.path.join(tmpdir, model_name)
        xgb_model.save_model(model_path)
        tl_model = treelite.Model.load(filename=model_path, model_format="xgboost")
    expected_num_tree = num_class * num_round * num_parallel_tree
    assert len(json.loads(tl_model.dump_as_json())["trees"]) == expected_num_tree

    out_pred = treelite.gtil.predict(tl_model, X_test)
    expected_pred = xgb_model.predict(dtest)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@pytest.mark.parametrize("model_format", ["binary", "json"])
@pytest.mark.parametrize(
    "objective,max_label",
    [
        ("binary:logistic", 2),
        ("binary:hinge", 2),
        ("binary:logitraw", 2),
        ("count:poisson", 4),
        ("rank:pairwise", 5),
        ("rank:ndcg", 5),
        ("rank:map", 5),
    ],
    ids=[
        "binary:logistic",
        "binary:hinge",
        "binary:logitraw",
        "count:poisson",
        "rank:pairwise",
        "rank:ndcg",
        "rank:map",
    ],
)
def test_xgb_nonlinear_objective(tmpdir, objective, max_label, model_format):
    # pylint: disable=too-many-locals
    """Test XGBoost with non-linear objectives with dummy data"""
    nrow = 16
    ncol = 8
    rng = np.random.default_rng(seed=0)
    X = rng.standard_normal(size=(nrow, ncol), dtype=np.float32)
    y = rng.integers(0, max_label, size=nrow)
    assert np.min(y) == 0
    assert np.max(y) == max_label - 1

    num_round = 4
    dtrain = xgb.DMatrix(X, label=y)
    if objective.startswith("rank:"):
        dtrain.set_group([nrow])
    xgb_model = xgb.train(
        {"objective": objective, "base_score": 0.5, "seed": 0},
        dtrain=dtrain,
        num_boost_round=num_round,
    )

    objective_tag = objective.replace(":", "_")
    if model_format == "json":
        model_name = f"nonlinear_{objective_tag}.json"
    else:
        model_name = f"nonlinear_{objective_tag}.bin"
    model_path = os.path.join(tmpdir, model_name)
    xgb_model.save_model(model_path)

    tl_model = treelite.Model.load(
        filename=model_path,
        model_format=("xgboost_json" if model_format == "json" else "xgboost"),
    )

    out_pred = treelite.gtil.predict(tl_model, X)
    expected_pred = xgb_model.predict(dtrain)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@pytest.mark.parametrize("in_memory", [True, False])
def test_xgb_categorical_split(in_memory):
    """Test toy XGBoost model with categorical splits"""
    dataset = "xgb_toy_categorical"
    if in_memory:
        xgb_model = xgb.Booster(model_file=dataset_db[dataset].model)
        tl_model = treelite.Model.from_xgboost(xgb_model)
    else:
        tl_model = treelite.Model.load(
            dataset_db[dataset].model, model_format="xgboost_json"
        )

    X, _ = load_svmlight_file(dataset_db[dataset].dtest, zero_based=True)
    expected_pred = load_txt(dataset_db[dataset].expected_margin)
    out_pred = treelite.gtil.predict(tl_model, X.toarray())
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@pytest.mark.parametrize("model_format", ["binary", "json"])
def test_xgb_dart(tmpdir, model_format):
    # pylint: disable=too-many-locals
    """Test XGBoost DART model with dummy data"""
    nrow = 16
    ncol = 8
    rng = np.random.default_rng(seed=0)
    X = rng.standard_normal(size=(nrow, ncol))
    y = rng.integers(0, 2, size=nrow)
    assert np.min(y) == 0
    assert np.max(y) == 1

    num_round = 50
    dtrain = xgb.DMatrix(X, label=y)
    param = {
        "booster": "dart",
        "max_depth": 5,
        "learning_rate": 0.1,
        "objective": "binary:logistic",
        "sample_type": "uniform",
        "normalize_type": "tree",
        "rate_drop": 0.1,
        "skip_drop": 0.5,
    }
    xgb_model = xgb.train(param, dtrain=dtrain, num_boost_round=num_round)

    if model_format == "json":
        model_name = "dart.json"
        model_path = os.path.join(tmpdir, model_name)
        xgb_model.save_model(model_path)
        tl_model = treelite.Model.load(filename=model_path, model_format="xgboost_json")
    else:
        tl_model = treelite.Model.from_xgboost(xgb_model)
    out_pred = treelite.gtil.predict(tl_model, X)
    expected_pred = xgb_model.predict(dtrain)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


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


@pytest.mark.parametrize("objective", ["binary", "xentlambda", "xentropy"])
def test_lightgbm_binary_classification(tmpdir, objective):
    # pylint: disable=too-many-locals
    """Test LightGBM binary classifier"""
    lgb_model_path = os.path.join(tmpdir, "breast_cancer_lightgbm.txt")
    X, y = load_breast_cancer(return_X_y=True)
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
        num_boost_round=10,
        valid_sets=[dtrain, dtest],
        valid_names=["train", "test"],
    )
    expected_prob = lgb_model.predict(X_test)
    lgb_model.save_model(lgb_model_path)

    tl_model = treelite.Model.load(lgb_model_path, model_format="lightgbm")
    out_prob = treelite.gtil.predict(tl_model, X_test)
    np.testing.assert_almost_equal(out_prob, expected_prob, decimal=5)


@pytest.mark.parametrize("boosting_type", ["gbdt", "rf"])
@pytest.mark.parametrize("objective", ["multiclass", "multiclassova"])
def test_lightgbm_multiclass_classification(tmpdir, objective, boosting_type):
    # pylint: disable=too-many-locals
    """Test LightGBM multi-class classifier"""
    lgb_model_path = os.path.join(tmpdir, "iris_lightgbm.txt")
    X, y = load_iris(return_X_y=True)
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
        "num_class": 3,
        "num_leaves": 31,
        "learning_rate": 0.05,
    }
    if boosting_type == "rf":
        param.update({"bagging_fraction": 0.8, "bagging_freq": 1})
    lgb_model = lgb.train(
        param,
        dtrain,
        num_boost_round=10,
        valid_sets=[dtrain, dtest],
        valid_names=["train", "test"],
    )
    expected_pred = lgb_model.predict(X_test)
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


@given(
    dataset=standard_regression_datasets(),
    predict_type=sampled_from(["leaf_id", "score_per_tree"]),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_predict_special_with_regressor(dataset, predict_type, callback):
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
    num_round = 10
    xgb_model = xgb.train(
        param,
        dtrain,
        num_boost_round=num_round,
    )
    model: treelite.Model = treelite.Model.from_xgboost(xgb_model)
    assert model.num_tree == num_round

    X_sample = X[0:sample_size]
    if predict_type == "leaf_id":
        leaf_pred = treelite.gtil.predict_leaf(model, X_sample)
        assert leaf_pred.shape == (sample_size, num_round)
        xgb_leaf_pred = xgb_model.predict(xgb.DMatrix(X_sample), pred_leaf=True)
        assert np.array_equal(leaf_pred, xgb_leaf_pred)
    else:
        pred_per_tree = treelite.gtil.predict_per_tree(model, X_sample)
        assert pred_per_tree.shape == (sample_size, num_round)
        pred = xgb_model.predict(xgb.DMatrix(X_sample), output_margin=True)
        np.testing.assert_almost_equal(np.sum(pred_per_tree, axis=1), pred, decimal=3)


@given(
    predict_type=sampled_from(["leaf_id", "score_per_tree"]),
    dataset=standard_classification_datasets(n_classes=just(2)),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_predict_special_with_binary_classifier(predict_type, dataset, callback):
    # pylint: disable=too-many-locals
    """Test predict_leaf / predict_per_tree with XGBoost binary classifier"""
    X, y = dataset
    sample_size = callback.draw(integers(min_value=1, max_value=X.shape[0]))

    num_round = 10
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
        num_boost_round=num_round,
    )
    model: treelite.Model = treelite.Model.from_xgboost(xgb_model)
    assert model.num_tree == num_round

    X_sample = X[0:sample_size]
    if predict_type == "leaf_id":
        leaf_pred = treelite.gtil.predict_leaf(model, X_sample)
        assert leaf_pred.shape == (sample_size, num_round)
        xgb_leaf_pred = xgb_model.predict(xgb.DMatrix(X_sample), pred_leaf=True)
        assert np.array_equal(leaf_pred, xgb_leaf_pred)
    else:
        pred_per_tree = treelite.gtil.predict_per_tree(model, X_sample)
        assert pred_per_tree.shape == (sample_size, num_round)
        pred = xgb_model.predict(xgb.DMatrix(X_sample), output_margin=True)
        np.testing.assert_almost_equal(np.sum(pred_per_tree, axis=1), pred, decimal=3)


@given(
    predict_type=sampled_from(["leaf_id", "score_per_tree"]),
    dataset=standard_classification_datasets(
        n_classes=integers(min_value=3, max_value=10), n_informative=just(5)
    ),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_predict_special_with_multi_classifier_grove_per_class(
    predict_type, dataset, callback
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
    num_round = 10
    xgb_model = xgb.train(
        param,
        dtrain,
        num_boost_round=num_round,
    )
    model: treelite.Model = treelite.Model.from_xgboost(xgb_model)
    assert model.num_tree == num_round * num_class

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
