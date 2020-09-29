# -*- coding: utf-8 -*-
"""Tests for XGBoost integration"""
# pylint: disable=R0201
import math
import os

import numpy as np
import pytest
import treelite
import treelite_runtime
from treelite.util import has_sklearn
from treelite.contrib import _libext
from .util import os_compatible_toolchains

try:
    import xgboost
except ImportError:
    # skip this test suite if XGBoost is not installed
    pytest.skip('XGBoost not installed; skipping', allow_module_level=True)


@pytest.mark.skipif(not has_sklearn(), reason='Needs scikit-learn')
@pytest.mark.parametrize('toolchain', os_compatible_toolchains())
def test_xgb_boston(tmpdir, toolchain):  # pylint: disable=too-many-locals
    """Test Boston data (regression)"""
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    dtrain = xgboost.DMatrix(X_train, label=y_train)
    dtest = xgboost.DMatrix(X_test, label=y_test)
    param = {'max_depth': 8, 'eta': 1, 'silent': 1, 'objective': 'reg:linear'}
    num_round = 10
    bst = xgboost.train(param, dtrain, num_boost_round=num_round,
                        evals=[(dtrain, 'train'), (dtest, 'test')])

    model = treelite.Model.from_xgboost(bst)
    assert model.num_feature == dtrain.num_col()
    assert model.num_output_group == 1
    assert model.num_tree == num_round
    libpath = os.path.join(tmpdir, 'boston' + _libext())
    model.export_lib(toolchain=toolchain, libpath=libpath, params={'parallel_comp': model.num_tree},
                     verbose=True)

    predictor = treelite_runtime.Predictor(libpath=libpath, verbose=True)
    assert predictor.num_feature == dtrain.num_col()
    assert predictor.num_output_group == 1
    assert predictor.pred_transform == 'identity'
    assert predictor.global_bias == 0.5
    assert predictor.sigmoid_alpha == 1.0
    batch = treelite_runtime.Batch.from_npy2d(X_test)
    out_pred = predictor.predict(batch)
    expected_pred = bst.predict(dtest)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@pytest.mark.skipif(not has_sklearn(), reason='Needs scikit-learn')
@pytest.mark.parametrize('toolchain', os_compatible_toolchains())
def test_xgb_iris(tmpdir, toolchain):  # pylint: disable=too-many-locals
    """Test Iris data (multi-class classification)"""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    dtrain = xgboost.DMatrix(X_train, label=y_train)
    dtest = xgboost.DMatrix(X_test, label=y_test)
    num_class = 3
    num_round = 10
    param = {'max_depth': 6, 'eta': 0.05, 'num_class': num_class, 'verbosity': 0,
             'objective': 'multi:softmax', 'metric': 'mlogloss'}
    bst = xgboost.train(param, dtrain, num_boost_round=num_round,
                        evals=[(dtrain, 'train'), (dtest, 'test')])

    model = treelite.Model.from_xgboost(bst)
    assert model.num_feature == dtrain.num_col()
    assert model.num_output_group == num_class
    assert model.num_tree == num_round * num_class
    libpath = os.path.join(tmpdir, 'iris' + _libext())
    model.export_lib(toolchain=toolchain, libpath=libpath, params={}, verbose=True)

    predictor = treelite_runtime.Predictor(libpath=libpath, verbose=True)
    assert predictor.num_feature == dtrain.num_col()
    assert predictor.num_output_group == num_class
    assert predictor.pred_transform == 'max_index'
    assert predictor.global_bias == 0.5
    assert predictor.sigmoid_alpha == 1.0
    batch = treelite_runtime.Batch.from_npy2d(X_test)
    out_pred = predictor.predict(batch)
    expected_pred = bst.predict(dtest)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@pytest.mark.parametrize('toolchain', os_compatible_toolchains())
@pytest.mark.parametrize('objective,max_label,global_bias',
                         [('binary:logistic', 2, 0), ('count:poisson', 4, math.log(0.5))],
                         ids=['binary:logistic', 'count:poisson'])
def test_nonlinear_objective(tmpdir, objective, max_label, global_bias, toolchain):
    # pylint: disable=too-many-locals
    """Test non-linear objectives with dummy data"""
    np.random.seed(0)
    nrow = 16
    ncol = 8
    X = np.random.randn(nrow, ncol)
    y = np.random.randint(0, max_label, size=nrow)
    assert np.min(y) == 0
    assert np.max(y) == max_label - 1

    num_round = 4
    dtrain = xgboost.DMatrix(X, y)
    bst = xgboost.train({'objective': objective}, dtrain=dtrain, num_boost_round=num_round)

    model = treelite.Model.from_xgboost(bst)
    assert model.num_feature == dtrain.num_col()
    assert model.num_output_group == 1
    assert model.num_tree == num_round
    libpath = os.path.join(tmpdir, f'{objective.replace(":", "_")}' + _libext())
    model.export_lib(toolchain=toolchain, libpath=libpath, params={}, verbose=True)

    expected_pred_transform = {'binary:logistic': 'sigmoid', 'count:poisson': 'exponential'}

    predictor = treelite_runtime.Predictor(libpath=libpath, verbose=True)
    assert predictor.num_feature == dtrain.num_col()
    assert predictor.num_output_group == 1
    assert predictor.pred_transform == expected_pred_transform[objective]
    np.testing.assert_almost_equal(predictor.global_bias, global_bias, decimal=5)
    assert predictor.sigmoid_alpha == 1.0
    batch = treelite_runtime.Batch.from_npy2d(X)
    out_pred = predictor.predict(batch)
    expected_pred = bst.predict(dtrain)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


@pytest.mark.skipif(not has_sklearn(), reason='Needs scikit-learn')
@pytest.mark.parametrize('toolchain', os_compatible_toolchains())
def test_xgb_deserializers(tmpdir, toolchain):
    # pylint: disable=too-many-locals
    """Test Boston data (regression)"""
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    # Train xgboost model
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    dtrain = xgboost.DMatrix(X_train, label=y_train)
    dtest = xgboost.DMatrix(X_test, label=y_test)
    param = {'max_depth': 8, 'eta': 1, 'silent': 1, 'objective': 'reg:linear'}
    num_round = 10
    bst = xgboost.train(param, dtrain, num_boost_round=num_round,
                        evals=[(dtrain, 'train'), (dtest, 'test')])

    # Serialize xgboost model
    model_bin_path = os.path.join(tmpdir, 'serialized.model')
    bst.save_model(model_bin_path)
    model_json_path = os.path.join(tmpdir, 'serialized.json')
    bst.save_model(model_json_path)

    # Construct Treelite models from xgboost serializations
    model_bin = treelite.Model.load(
        model_bin_path, model_format='xgboost'
    )
    model_json = treelite.Model.load(
        model_json_path, model_format='xgboost_json'
    )
    with open(model_json_path) as file_:
        json_str = file_.read()
    model_json_str = treelite.Model.from_xgboost_json(json_str)

    # Compile models to libraries
    model_bin_lib = os.path.join(tmpdir, 'bin{}'.format(_libext()))
    model_bin.export_lib(
        toolchain=toolchain,
        libpath=model_bin_lib,
        params={'parallel_comp': model_bin.num_tree}
    )
    model_json_lib = os.path.join(tmpdir, 'json{}'.format(_libext()))
    model_json.export_lib(
        toolchain=toolchain,
        libpath=model_json_lib,
        params={'parallel_comp': model_json.num_tree}
    )
    model_json_str_lib = os.path.join(tmpdir, 'json_str{}'.format(_libext()))
    model_json_str.export_lib(
        toolchain=toolchain,
        libpath=model_json_str_lib,
        params={'parallel_comp': model_json_str.num_tree}
    )

    # Generate predictors from compiled libraries
    predictor_bin = treelite_runtime.Predictor(model_bin_lib)
    assert predictor_bin.num_feature == dtrain.num_col()
    assert predictor_bin.num_output_group == 1
    assert predictor_bin.pred_transform == 'identity'
    assert predictor_bin.global_bias == pytest.approx(0.5)
    assert predictor_bin.sigmoid_alpha == pytest.approx(1.0)

    predictor_json = treelite_runtime.Predictor(model_json_lib)
    assert predictor_json.num_feature == dtrain.num_col()
    assert predictor_json.num_output_group == 1
    assert predictor_json.pred_transform == 'identity'
    assert predictor_json.global_bias == pytest.approx(0.5)
    assert predictor_json.sigmoid_alpha == pytest.approx(1.0)

    predictor_json_str = treelite_runtime.Predictor(model_json_str_lib)
    assert predictor_json_str.num_feature == dtrain.num_col()
    assert predictor_json_str.num_output_group == 1
    assert predictor_json_str.pred_transform == 'identity'
    assert predictor_json_str.global_bias == pytest.approx(0.5)
    assert predictor_json_str.sigmoid_alpha == pytest.approx(1.0)

    # Run inference with each predictor
    batch = treelite_runtime.Batch.from_npy2d(X_test)
    bin_pred = predictor_bin.predict(batch)
    json_pred = predictor_json.predict(batch)
    json_str_pred = predictor_json_str.predict(batch)

    expected_pred = bst.predict(dtest)
    np.testing.assert_almost_equal(bin_pred, expected_pred, decimal=5)
    np.testing.assert_almost_equal(json_pred, expected_pred, decimal=5)
    np.testing.assert_almost_equal(json_str_pred, expected_pred, decimal=5)
