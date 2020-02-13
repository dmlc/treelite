# -*- coding: utf-8 -*-
"""Tests for XGBoost integration"""
from __future__ import print_function
import unittest
import math
import numpy as np
import pytest
import treelite
import treelite.runtime
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from util import os_compatible_toolchains, libname, assert_almost_equal

try:
  import xgboost
except ImportError:
  # skip this test suite if XGBoost is not installed
  pytest.skip('XGBoost not installed; skipping', allow_module_level=True)

class TestXGBoostIntegration(unittest.TestCase):
  def test_xgb(self):
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
    batch = treelite.runtime.Batch.from_npy2d(X_test)
    for toolchain in os_compatible_toolchains():
      model.export_lib(toolchain=toolchain, libpath=libpath,
                       params={}, verbose=True)
      predictor = treelite.runtime.Predictor(libpath=libpath, verbose=True)
      out_pred = predictor.predict(batch)
      assert_almost_equal(out_pred, expected_pred)
      assert predictor.num_feature == 13
      assert predictor.num_output_group == 1
      assert predictor.pred_transform == 'identity'
      assert predictor.global_bias == 0.5
      assert predictor.sigmoid_alpha == 1.0

  def test_xgb_iris(self):
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
    batch = treelite.runtime.Batch.from_npy2d(X_test)
    for toolchain in os_compatible_toolchains():
      model.export_lib(toolchain=toolchain, libpath=libpath,
                       params={}, verbose=True)
      predictor = treelite.runtime.Predictor(libpath=libpath, verbose=True)
      out_pred = predictor.predict(batch)
      assert_almost_equal(out_pred, expected_pred)
      assert predictor.num_feature == 4
      assert predictor.num_output_group == 3
      assert predictor.pred_transform == 'max_index'
      assert predictor.global_bias == 0.5
      assert predictor.sigmoid_alpha == 1.0


  def run_non_linear_objective(self, objective, max_label, global_bias):
    np.random.seed(0)
    kRows = 16
    kCols = 8
    X = np.random.randn(kRows, kCols)
    y = np.random.randint(0, max_label, size=kRows)
    assert y.min() == 0
    assert y.max() == max_label - 1

    dtrain = xgboost.DMatrix(X, y)
    booster = xgboost.train({'objective': objective}, dtrain=dtrain,
                            num_boost_round=4)
    expected_pred = booster.predict(dtrain)
    model = treelite.Model.from_xgboost(booster)
    libpath = libname('./'+objective+'{}')
    batch = treelite.runtime.Batch.from_npy2d(X)
    for toolchain in os_compatible_toolchains():
      model.export_lib(toolchain=toolchain, libpath=libpath,
                       params={}, verbose=True)
      predictor = treelite.runtime.Predictor(libpath=libpath, verbose=True)
      out_pred = predictor.predict(batch)
      assert_almost_equal(out_pred, expected_pred)
      assert predictor.num_feature == kCols
      np.testing.assert_almost_equal(predictor.global_bias, global_bias,
                                     decimal=6)

  def test_logisitc(self):
    self.run_non_linear_objective('binary:logistic', 2, 0)

  def test_posson(self):
    self.run_non_linear_objective('count:poisson', 4, math.log(0.5))
