# -*- coding: utf-8 -*-
"""Tests for XGBoost integration"""
from __future__ import print_function
import unittest
import os
import numpy as np
import xgboost
import treelite
import treelite.runtime
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from util import os_compatible_toolchains, libname

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
      assert np.allclose(out_pred, expected_pred, atol=1e-11, rtol=1e-6), \
        np.max(np.abs(out_pred - expected_pred))
