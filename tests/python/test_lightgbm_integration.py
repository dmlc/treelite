# -*- coding: utf-8 -*-
"""Tests for LightGBM integration"""
from __future__ import print_function
import unittest
import os
import numpy as np
import nose
import treelite
import treelite.runtime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from util import os_compatible_toolchains, libname

class TestLightGBMIntegration(unittest.TestCase):
  def test_lightgbm(self):
    try:
      import lightgbm
    except ImportError:
      raise nose.SkipTest()  # skip this test if LightGBM is not installed

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test \
      = train_test_split(X, y, test_size=0.2, shuffle=False)
    dtrain = lightgbm.Dataset(X_train, y_train)
    dtest = lightgbm.Dataset(X_test, y_test, reference=dtrain)
    param = {'task': 'train', 'boosting_type': 'gbdt',
             'objective': 'multiclass', 'metric': 'multi_logloss',
             'num_class': 3, 'num_leaves': 31, 'learning_rate': 0.05}
    num_round = 10
    watchlist = [dtrain, dtest]
    watchlist_names = ['train', 'test']
    bst = lightgbm.train(param, dtrain, num_round, watchlist, watchlist_names)
    bst.save_model('./iris_lightgbm.txt')

    expected_pred = bst.predict(X_test)

    model = treelite.Model.load('./iris_lightgbm.txt', model_format='lightgbm')
    libpath = libname('./iris{}')
    batch = treelite.runtime.Batch.from_npy2d(X_test)
    for toolchain in os_compatible_toolchains():
      model.export_lib(toolchain=toolchain, libpath=libpath,
                       params={}, verbose=True)
      predictor = treelite.runtime.Predictor(libpath=libpath, verbose=True)
      out_pred = predictor.predict(batch)
      assert np.allclose(out_pred, expected_pred, atol=1e-11, rtol=1e-6)
