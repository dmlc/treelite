# -*- coding: utf-8 -*-
"""Tests for XGBoost integration"""
# pylint: disable=R0201
import unittest
import math
import numpy as np
import pytest
import treelite
import treelite_runtime
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from .util import os_compatible_toolchains, libname, assert_almost_equal

try:
    import xgboost
except ImportError:
    # skip this test suite if XGBoost is not installed
    pytest.skip('XGBoost not installed; skipping', allow_module_level=True)


class TestXGBoostIntegration(unittest.TestCase):
    """Test suite for integrating with XGBoost"""
    def test_xgb_boston(self):  # pylint: disable=R0914
        """Test Boston data (regression)"""
        X, y = load_boston(return_X_y=True)
        X_train, X_test, y_train, y_test \
            = train_test_split(X, y, test_size=0.2, shuffle=False)
        dtrain = xgboost.DMatrix(X_train, label=y_train)
        dtest = xgboost.DMatrix(X_test, label=y_test)
        param = {'max_depth': 8, 'eta': 1, 'silent': 1, 'objective': 'reg:linear'}
        num_round = 10
        watchlist = [(dtrain, 'train'), (dtest, 'test')]
        bst = xgboost.train(param, dtrain, num_round, watchlist)

        expected_pred = bst.predict(dtest)

        model = treelite.Model.from_xgboost(bst)
        libpath = libname('./boston{}')
        batch = treelite_runtime.Batch.from_npy2d(X_test)
        for toolchain in os_compatible_toolchains():
            model.export_lib(toolchain=toolchain, libpath=libpath,
                             params={}, verbose=True)
            predictor = treelite_runtime.Predictor(libpath=libpath, verbose=True)
            out_pred = predictor.predict(batch)
            assert_almost_equal(out_pred, expected_pred)
            assert predictor.num_feature == 13
            assert predictor.num_output_group == 1
            assert predictor.pred_transform == 'identity'
            assert predictor.global_bias == 0.5
            assert predictor.sigmoid_alpha == 1.0

    def test_xgb_iris(self):  # pylint: disable=R0914
        """Test Iris data (multi-class classification)"""
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test \
            = train_test_split(X, y, test_size=0.2, shuffle=False)
        dtrain = xgboost.DMatrix(X_train, label=y_train)
        dtest = xgboost.DMatrix(X_test, label=y_test)
        param = {'max_depth': 6, 'eta': 0.05, 'num_class': 3, 'silent': 1,
                 'objective': 'multi:softmax', 'metric': 'mlogloss'}
        num_round = 10
        watchlist = [(dtrain, 'train'), (dtest, 'test')]
        bst = xgboost.train(param, dtrain, num_round, watchlist)

        expected_pred = bst.predict(dtest)

        model = treelite.Model.from_xgboost(bst)
        assert model.num_output_group == 3
        assert model.num_feature == dtrain.num_col()

        libpath = libname('./iris{}')
        batch = treelite_runtime.Batch.from_npy2d(X_test)
        for toolchain in os_compatible_toolchains():
            model.export_lib(toolchain=toolchain, libpath=libpath,
                             params={}, verbose=True)
            predictor = treelite_runtime.Predictor(libpath=libpath, verbose=True)
            out_pred = predictor.predict(batch)
            assert_almost_equal(out_pred, expected_pred)
            assert predictor.num_feature == 4
            assert predictor.num_output_group == 3
            assert predictor.pred_transform == 'max_index'
            assert predictor.global_bias == 0.5
            assert predictor.sigmoid_alpha == 1.0

    def run_non_linear_objective(self, objective, max_label, global_bias):  # pylint: disable=R0914
        """Helper function to test non-linear objectives"""
        np.random.seed(0)
        nrow = 16
        ncol = 8
        X = np.random.randn(nrow, ncol)
        y = np.random.randint(0, max_label, size=nrow)
        assert y.min() == 0
        assert y.max() == max_label - 1

        dtrain = xgboost.DMatrix(X, y)
        booster = xgboost.train({'objective': objective}, dtrain=dtrain,
                                num_boost_round=4)
        expected_pred = booster.predict(dtrain)
        model = treelite.Model.from_xgboost(booster)
        libpath = libname('./' + objective + '{}')
        batch = treelite_runtime.Batch.from_npy2d(X)
        for toolchain in os_compatible_toolchains():
            model.export_lib(toolchain=toolchain, libpath=libpath,
                             params={}, verbose=True)
            predictor = treelite_runtime.Predictor(libpath=libpath, verbose=True)
            out_pred = predictor.predict(batch)
            assert_almost_equal(out_pred, expected_pred)
            assert predictor.num_feature == ncol
            np.testing.assert_almost_equal(predictor.global_bias, global_bias,
                                           decimal=6)

    def test_logisitc(self):
        """Test binary:logistic objective with dummy data"""
        self.run_non_linear_objective('binary:logistic', 2, 0)

    def test_poisson(self):
        """Test count:poisson objective with dummy data"""
        self.run_non_linear_objective('count:poisson', 4, math.log(0.5))
