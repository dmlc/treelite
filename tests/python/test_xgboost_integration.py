# -*- coding: utf-8 -*-
"""Tests for XGBoost integration"""
# pylint: disable=R0201, R0915
import json
import math
import os

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis.strategies import integers, lists, sampled_from
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import treelite
import treelite_runtime
from treelite.contrib import _libext

from .hypothesis_util import standard_regression_datasets, standard_settings
from .metadata import dataset_db
from .util import TemporaryDirectory, check_predictor, os_compatible_toolchains

try:
    import xgboost as xgb
except ImportError:
    # skip this test suite if XGBoost is not installed
    pytest.skip("XGBoost not installed; skipping", allow_module_level=True)


@given(
    toolchain=sampled_from(os_compatible_toolchains()),
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
def test_xgb_regression(toolchain, objective, model_format, num_parallel_tree, dataset):
    # pylint: disable=too-many-locals
    """Test a random regression dataset"""
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
    bst = xgb.train(
        param,
        dtrain,
        num_boost_round=num_round,
        evals=[(dtrain, "train"), (dtest, "test")],
    )
    with TemporaryDirectory() as tmpdir:
        if model_format == "json":
            model_name = "regression.json"
            model_path = os.path.join(tmpdir, model_name)
            bst.save_model(model_path)
            model = treelite.Model.load(
                filename=model_path, model_format="xgboost_json"
            )
        else:
            model_name = "regression.model"
            model_path = os.path.join(tmpdir, model_name)
            bst.save_model(model_path)
            model = treelite.Model.load(filename=model_path, model_format="xgboost")

        assert model.num_feature == dtrain.num_col()
        assert model.num_class == 1
        assert model.num_tree == num_round * num_parallel_tree
        libpath = os.path.join(tmpdir, "regression" + _libext())
        model.export_lib(
            toolchain=toolchain,
            libpath=libpath,
            params={"parallel_comp": model.num_tree},
            verbose=True,
        )

        predictor = treelite_runtime.Predictor(libpath=libpath, verbose=True)
        assert predictor.num_feature == dtrain.num_col()
        assert predictor.num_class == 1
        assert predictor.pred_transform == "identity"
        assert predictor.global_bias == 0.5
        assert predictor.sigmoid_alpha == 1.0
        dmat = treelite_runtime.DMatrix(X_test, dtype="float32")
        out_pred = predictor.predict(dmat)
        expected_pred = bst.predict(dtest)
        np.testing.assert_almost_equal(out_pred, expected_pred, decimal=3)


@pytest.mark.parametrize("num_parallel_tree", [1, 3, 5])
@pytest.mark.parametrize("model_format", ["binary", "json"])
@pytest.mark.parametrize(
    "objective,expected_pred_transform",
    [("multi:softmax", "max_index"), ("multi:softprob", "softmax")],
    ids=["multi:softmax", "multi:softprob"],
)
@pytest.mark.parametrize("toolchain", os_compatible_toolchains())
def test_xgb_iris(
    tmpdir,
    toolchain,
    objective,
    model_format,
    expected_pred_transform,
    num_parallel_tree,
):
    # pylint: disable=too-many-locals, too-many-arguments
    """Test Iris data (multi-class classification)"""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    num_class = 3
    num_round = 10
    param = {
        "max_depth": 6,
        "eta": 0.05,
        "num_class": num_class,
        "verbosity": 0,
        "objective": objective,
        "metric": "mlogloss",
        "num_parallel_tree": num_parallel_tree,
    }
    bst = xgb.train(
        param,
        dtrain,
        num_boost_round=num_round,
        evals=[(dtrain, "train"), (dtest, "test")],
    )

    if model_format == "json":
        model_name = "iris.json"
        model_path = os.path.join(tmpdir, model_name)
        bst.save_model(model_path)
        model = treelite.Model.load(filename=model_path, model_format="xgboost_json")
    else:
        model_name = "iris.model"
        model_path = os.path.join(tmpdir, model_name)
        bst.save_model(model_path)
        model = treelite.Model.load(filename=model_path, model_format="xgboost")
    assert model.num_feature == dtrain.num_col()
    assert model.num_class == num_class
    assert model.num_tree == num_round * num_class * num_parallel_tree
    libpath = os.path.join(tmpdir, "iris" + _libext())
    model.export_lib(toolchain=toolchain, libpath=libpath, params={}, verbose=True)

    predictor = treelite_runtime.Predictor(libpath=libpath, verbose=True)
    assert predictor.num_feature == dtrain.num_col()
    assert predictor.num_class == num_class
    assert predictor.pred_transform == expected_pred_transform
    assert predictor.global_bias == 0.5
    assert predictor.sigmoid_alpha == 1.0
    dmat = treelite_runtime.DMatrix(X_test, dtype="float32")
    out_pred = predictor.predict(dmat)
    expected_pred = bst.predict(dtest)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@pytest.mark.parametrize("model_format", ["binary", "json"])
@pytest.mark.parametrize(
    "objective,max_label,expected_global_bias",
    [
        ("binary:logistic", 2, 0),
        ("binary:hinge", 2, 0.5),
        ("binary:logitraw", 2, 0.5),
        ("count:poisson", 4, math.log(0.5)),
        ("rank:pairwise", 5, 0.5),
        ("rank:ndcg", 5, 0.5),
        ("rank:map", 5, 0.5),
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
@pytest.mark.parametrize("toolchain", os_compatible_toolchains())
def test_nonlinear_objective(
    tmpdir, objective, max_label, expected_global_bias, toolchain, model_format
):
    # pylint: disable=too-many-locals,too-many-arguments
    """Test non-linear objectives with dummy data"""
    np.random.seed(0)
    nrow = 16
    ncol = 8
    X = np.random.randn(nrow, ncol)
    y = np.random.randint(0, max_label, size=nrow)
    assert np.min(y) == 0
    assert np.max(y) == max_label - 1

    num_round = 4
    dtrain = xgb.DMatrix(X, label=y)
    if objective.startswith("rank:"):
        dtrain.set_group([nrow])
    bst = xgb.train(
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
    bst.save_model(model_path)

    model = treelite.Model.load(
        filename=model_path,
        model_format=("xgboost_json" if model_format == "json" else "xgboost"),
    )
    assert model.num_feature == dtrain.num_col()
    assert model.num_class == 1
    assert model.num_tree == num_round
    libpath = os.path.join(tmpdir, objective_tag + _libext())
    model.export_lib(toolchain=toolchain, libpath=libpath, params={}, verbose=True)

    expected_pred_transform = {
        "binary:logistic": "sigmoid",
        "binary:hinge": "hinge",
        "binary:logitraw": "identity",
        "count:poisson": "exponential",
        "rank:pairwise": "identity",
        "rank:ndcg": "identity",
        "rank:map": "identity",
    }

    predictor = treelite_runtime.Predictor(libpath=libpath, verbose=True)
    assert predictor.num_feature == dtrain.num_col()
    assert predictor.num_class == 1
    assert predictor.pred_transform == expected_pred_transform[objective]
    np.testing.assert_almost_equal(
        predictor.global_bias, expected_global_bias, decimal=5
    )
    assert predictor.sigmoid_alpha == 1.0
    dmat = treelite_runtime.DMatrix(X, dtype="float32")
    out_pred = predictor.predict(dmat)
    expected_pred = bst.predict(dtrain)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@given(
    toolchain=sampled_from(os_compatible_toolchains()),
    dataset=standard_regression_datasets(),
)
@settings(**standard_settings())
def test_xgb_deserializers(toolchain, dataset):
    # pylint: disable=too-many-locals
    """Test a random regression dataset and test serializers"""
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {"max_depth": 8, "eta": 1, "silent": 1, "objective": "reg:linear"}
    num_round = 10
    bst = xgb.train(
        param,
        dtrain,
        num_boost_round=num_round,
        evals=[(dtrain, "train"), (dtest, "test")],
    )

    with TemporaryDirectory() as tmpdir:
        # Serialize xgboost model
        model_bin_path = os.path.join(tmpdir, "serialized.model")
        bst.save_model(model_bin_path)
        model_json_path = os.path.join(tmpdir, "serialized.json")
        bst.save_model(model_json_path)
        model_json_str = bst.save_raw(raw_format="json")

        # Construct Treelite models from xgboost serializations
        model_bin = treelite.Model.load(model_bin_path, model_format="xgboost")
        model_json = treelite.Model.load(model_json_path, model_format="xgboost_json")
        model_json_str = treelite.Model.from_xgboost_json(model_json_str)

        # Compile models to libraries
        model_bin_lib = os.path.join(tmpdir, "bin{}".format(_libext()))
        model_bin.export_lib(
            toolchain=toolchain,
            libpath=model_bin_lib,
            params={"parallel_comp": model_bin.num_tree},
        )
        model_json_lib = os.path.join(tmpdir, "json{}".format(_libext()))
        model_json.export_lib(
            toolchain=toolchain,
            libpath=model_json_lib,
            params={"parallel_comp": model_json.num_tree},
        )
        model_json_str_lib = os.path.join(tmpdir, "json_str{}".format(_libext()))
        model_json_str.export_lib(
            toolchain=toolchain,
            libpath=model_json_str_lib,
            params={"parallel_comp": model_json_str.num_tree},
        )

        # Generate predictors from compiled libraries
        predictor_bin = treelite_runtime.Predictor(model_bin_lib)
        assert predictor_bin.num_feature == dtrain.num_col()
        assert predictor_bin.num_class == 1
        assert predictor_bin.pred_transform == "identity"
        assert predictor_bin.global_bias == pytest.approx(0.5)
        assert predictor_bin.sigmoid_alpha == pytest.approx(1.0)

        predictor_json = treelite_runtime.Predictor(model_json_lib)
        assert predictor_json.num_feature == dtrain.num_col()
        assert predictor_json.num_class == 1
        assert predictor_json.pred_transform == "identity"
        assert predictor_json.global_bias == pytest.approx(0.5)
        assert predictor_json.sigmoid_alpha == pytest.approx(1.0)

        predictor_json_str = treelite_runtime.Predictor(model_json_str_lib)
        assert predictor_json_str.num_feature == dtrain.num_col()
        assert predictor_json_str.num_class == 1
        assert predictor_json_str.pred_transform == "identity"
        assert predictor_json_str.global_bias == pytest.approx(0.5)
        assert predictor_json_str.sigmoid_alpha == pytest.approx(1.0)

        # Run inference with each predictor
        dmat = treelite_runtime.DMatrix(X_test, dtype="float32")
        bin_pred = predictor_bin.predict(dmat)
        json_pred = predictor_json.predict(dmat)
        json_str_pred = predictor_json_str.predict(dmat)

        expected_pred = bst.predict(dtest)
        np.testing.assert_almost_equal(bin_pred, expected_pred, decimal=4)
        np.testing.assert_almost_equal(json_pred, expected_pred, decimal=4)
        np.testing.assert_almost_equal(json_str_pred, expected_pred, decimal=4)


@pytest.mark.parametrize("parallel_comp", [None, 5])
@pytest.mark.parametrize("quantize", [True, False])
@pytest.mark.parametrize("toolchain", os_compatible_toolchains())
def test_xgb_categorical_split(tmpdir, toolchain, quantize, parallel_comp):
    """Test toy XGBoost model with categorical splits"""
    dataset = "xgb_toy_categorical"
    model = treelite.Model.load(dataset_db[dataset].model, model_format="xgboost_json")
    libpath = os.path.join(tmpdir, dataset_db[dataset].libname + _libext())

    params = {
        "quantize": (1 if quantize else 0),
        "parallel_comp": (parallel_comp if parallel_comp else 0),
    }
    model.export_lib(toolchain=toolchain, libpath=libpath, params=params, verbose=True)

    predictor = treelite_runtime.Predictor(libpath)
    check_predictor(predictor, dataset)


@pytest.mark.parametrize("model_format", ["binary", "json"])
@pytest.mark.parametrize("toolchain", os_compatible_toolchains())
def test_xgb_dart(tmpdir, toolchain, model_format):
    # pylint: disable=too-many-locals,too-many-arguments
    """Test dart booster with dummy data"""
    np.random.seed(0)
    nrow = 16
    ncol = 8
    X = np.random.randn(nrow, ncol)
    y = np.random.randint(0, 2, size=nrow)
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
    bst = xgb.train(param, dtrain=dtrain, num_boost_round=num_round)

    if model_format == "json":
        model_json_path = os.path.join(tmpdir, "serialized.json")
        bst.save_model(model_json_path)
        model = treelite.Model.load(model_json_path, model_format="xgboost_json")
    else:
        model_bin_path = os.path.join(tmpdir, "serialized.model")
        bst.save_model(model_bin_path)
        model = treelite.Model.load(model_bin_path, model_format="xgboost")

    assert model.num_feature == dtrain.num_col()
    assert model.num_class == 1
    assert model.num_tree == num_round
    libpath = os.path.join(tmpdir, "dart" + _libext())
    model.export_lib(toolchain=toolchain, libpath=libpath, params={}, verbose=True)

    predictor = treelite_runtime.Predictor(libpath=libpath, verbose=True)
    assert predictor.num_feature == dtrain.num_col()
    assert predictor.num_class == 1
    assert predictor.pred_transform == "sigmoid"
    np.testing.assert_almost_equal(predictor.global_bias, 0, decimal=5)
    assert predictor.sigmoid_alpha == 1.0
    dmat = treelite_runtime.DMatrix(X, dtype="float32")
    out_pred = predictor.predict(dmat)
    expected_pred = bst.predict(dtrain)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@given(
    lists(integers(min_value=0, max_value=20), min_size=1, max_size=10),
    sampled_from(["string", "object", "list"]),
    sampled_from([True, False])
)
def test_extra_field_in_xgb_json(
    random_integer_seq,
    extra_field_type,
    use_tempfile
):
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
        if isinstance(model_obj, list):
            idx = seq[0] % len(model_obj)
            subobj = model_obj[idx]
            if isinstance(subobj, (dict, list)):
                return insert_extra_field(subobj, seq[1:])
        if isinstance(model_obj, dict):
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
            treelite.Model.load(new_model_path, model_format="xgboost_json",
                                allow_unknown_field=True)
    else:
        treelite.Model.from_xgboost_json(new_model_str, allow_unknown_field=True)
