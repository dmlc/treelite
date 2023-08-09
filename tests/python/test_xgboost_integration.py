"""Tests for XGBoost integration"""
# pylint: disable=R0201, R0915
import json
import os

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis.strategies import data as hypothesis_callback
from hypothesis.strategies import integers, just, lists, sampled_from

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
    from sklearn.datasets import load_svmlight_file
    from sklearn.model_selection import train_test_split
except ImportError:
    # skip this test suite if scikit-learn is not installed
    pytest.skip("scikit-learn not installed; skipping", allow_module_level=True)


@given(
    dataset=standard_regression_datasets(),
    objective=sampled_from(
        [
            "reg:linear",
            "reg:squarederror",
            "reg:squaredlogerror",
            "reg:pseudohubererror",
        ]
    ),
    model_format=sampled_from(["binary", "json"]),
    num_boost_round=integers(min_value=5, max_value=50),
    num_parallel_tree=integers(min_value=1, max_value=5),
)
@settings(**standard_settings())
def test_xgb_regression(
    dataset, objective, model_format, num_boost_round, num_parallel_tree
):
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
    xgb_model = xgb.train(
        param,
        dtrain,
        num_boost_round=num_boost_round,
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
            == num_boost_round * num_parallel_tree
        )

        out_pred = treelite.gtil.predict(tl_model, X_test)
        expected_pred = xgb_model.predict(dtest)
        np.testing.assert_almost_equal(out_pred, expected_pred, decimal=3)


@given(
    dataset=standard_classification_datasets(
        n_classes=integers(min_value=3, max_value=10), n_informative=just(5)
    ),
    objective=sampled_from(["multi:softmax", "multi:softprob"]),
    model_format=sampled_from(["binary", "json"]),
    num_boost_round=integers(min_value=5, max_value=50),
    num_parallel_tree=integers(min_value=1, max_value=5),
)
@settings(**standard_settings())
def test_xgb_multiclass_classifier(
    dataset, objective, model_format, num_boost_round, num_parallel_tree
):
    # pylint: disable=too-many-locals
    """Test XGBoost with Iris data (multi-class classification)"""
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    num_class = np.max(y) + 1
    param = {
        "max_depth": 6,
        "eta": 0.05,
        "num_class": num_class,
        "verbosity": 0,
        "objective": objective,
        "metric": "mlogloss",
        "num_parallel_tree": num_parallel_tree,
    }
    xgb_model = xgb.train(
        param,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dtest, "test")],
    )

    with TemporaryDirectory() as tmpdir:
        if model_format == "json":
            model_name = "iris.json"
            model_path = os.path.join(tmpdir, model_name)
            xgb_model.save_model(model_path)
            tl_model = treelite.Model.load(
                filename=model_path, model_format="xgboost_json"
            )
        else:
            model_name = "iris.model"
            model_path = os.path.join(tmpdir, model_name)
            xgb_model.save_model(model_path)
            tl_model = treelite.Model.load(filename=model_path, model_format="xgboost")
        expected_num_tree = num_class * num_boost_round * num_parallel_tree
        assert len(json.loads(tl_model.dump_as_json())["trees"]) == expected_num_tree

        out_pred = treelite.gtil.predict(tl_model, X_test)
        expected_pred = xgb_model.predict(dtest)
        np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@given(
    objective_pair=sampled_from(
        [
            ("binary:logistic", 2),
            ("binary:hinge", 2),
            ("binary:logitraw", 2),
            ("count:poisson", 4),
            ("rank:pairwise", 5),
            ("rank:ndcg", 5),
            ("rank:map", 5),
        ],
    ),
    model_format=sampled_from(["binary", "json"]),
    num_boost_round=integers(min_value=5, max_value=50),
    callback=hypothesis_callback(),
)
@settings(**standard_settings())
def test_xgb_nonlinear_objective(
    objective_pair, model_format, num_boost_round, callback
):
    # pylint: disable=too-many-locals
    """Test XGBoost with non-linear objectives with synthetic data"""
    objective, num_class = objective_pair
    X, y = callback.draw(
        standard_classification_datasets(
            n_classes=just(num_class), n_informative=just(5)
        )
    )
    assert np.min(y) == 0
    assert np.max(y) == num_class - 1

    dtrain = xgb.DMatrix(X, label=y)
    if objective.startswith("rank:"):
        dtrain.set_group([X.shape[0]])
    xgb_model = xgb.train(
        {"objective": objective, "base_score": 0.5, "seed": 0},
        dtrain=dtrain,
        num_boost_round=num_boost_round,
    )

    objective_tag = objective.replace(":", "_")
    if model_format == "json":
        model_name = f"nonlinear_{objective_tag}.json"
    else:
        model_name = f"nonlinear_{objective_tag}.bin"
    with TemporaryDirectory() as tmpdir:
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


@given(
    dataset=standard_classification_datasets(n_classes=just(2)),
    model_format=sampled_from(["binary", "json"]),
    num_boost_round=integers(min_value=5, max_value=50),
)
@settings(**standard_settings())
def test_xgb_dart(dataset, model_format, num_boost_round):
    # pylint: disable=too-many-locals
    """Test XGBoost DART model with dummy data"""
    X, y = dataset

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
    xgb_model = xgb.train(param, dtrain=dtrain, num_boost_round=num_boost_round)

    with TemporaryDirectory() as tmpdir:
        if model_format == "json":
            model_name = "dart.json"
            model_path = os.path.join(tmpdir, model_name)
            xgb_model.save_model(model_path)
            tl_model = treelite.Model.load(
                filename=model_path, model_format="xgboost_json"
            )
        else:
            tl_model = treelite.Model.from_xgboost(xgb_model)
        out_pred = treelite.gtil.predict(tl_model, X)
        expected_pred = xgb_model.predict(dtrain)
        np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@given(
    lists(integers(min_value=0, max_value=20), min_size=1, max_size=10),
    sampled_from(["string", "object", "list"]),
    sampled_from([True, False]),
)
@settings(print_blob=True, deadline=None)
def test_extra_field_in_xgb_json(random_integer_seq, extra_field_type, use_tempfile):
    # pylint: disable=too-many-locals,too-many-arguments
    """
    Test if we can handle extra fields in XGBoost JSON model file
    Insert an extra field at a random place and then load the model into Treelite,
    as follows:
    * Use the Hypothesis package to generate a random sequence of integers.
    * Then use the integer sequence to navigate through the XGBoost model JSON object.
    * Lastly, after navigating, insert the extra field at that location.
    """
    np.random.seed(0)
    nrow = 16
    ncol = 8
    X = np.random.randn(nrow, ncol)
    y = np.random.randint(0, 2, size=nrow)
    assert np.min(y) == 0
    assert np.max(y) == 1

    dtrain = xgb.DMatrix(X, label=y)
    param = {
        "max_depth": 1,
        "eta": 1,
        "objective": "binary:logistic",
        "verbosity": 0,
    }
    bst = xgb.train(
        param,
        dtrain,
        num_boost_round=1,
    )
    model_json_str = bst.save_raw(raw_format="json").decode(encoding="utf-8")
    model_obj = json.loads(model_json_str)

    def get_extra_field_value():
        if extra_field_type == "object":
            return {}
        if extra_field_type == "string":
            return "extra"
        if extra_field_type == "list":
            return []
        return None

    def insert_extra_field(model_obj, seq):
        if (not seq) or (not model_obj):
            if isinstance(model_obj, dict):
                model_obj["extra_field"] = get_extra_field_value()
                return True
        elif isinstance(model_obj, list):
            idx = seq[0] % len(model_obj)
            subobj = model_obj[idx]
            if isinstance(subobj, (dict, list)):
                return insert_extra_field(subobj, seq[1:])
        elif isinstance(model_obj, dict):
            idx = seq[0] % len(model_obj)
            subobj = list(model_obj.items())[idx][1]
            if isinstance(subobj, (dict, list)):
                if not insert_extra_field(subobj, seq[1:]):
                    model_obj["extra_field"] = get_extra_field_value()
                return True
            model_obj["extra_field"] = get_extra_field_value()
            return True
        return False

    insert_extra_field(model_obj, random_integer_seq)
    new_model_str = json.dumps(model_obj)
    assert "extra_field" in new_model_str
    if use_tempfile:
        with TemporaryDirectory() as tmpdir:
            new_model_path = os.path.join(tmpdir, "new_model.json")
            with open(new_model_path, "w", encoding="utf-8") as f:
                f.write(new_model_str)
            treelite.Model.load(
                new_model_path, model_format="xgboost_json", allow_unknown_field=True
            )
    else:
        treelite.Model.from_xgboost_json(new_model_str, allow_unknown_field=True)
