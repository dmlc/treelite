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
from util import os_compatible_toolchains, libname, assert_almost_equal,\
                 run_pipeline_test

dpath = os.path.abspath(os.path.join(os.getcwd(), 'tests/examples/'))

class TestLightGBMIntegration(unittest.TestCase):
  def test_lightgbm_multiclass_classification(self):
    try:
      import lightgbm
    except ImportError:
      raise nose.SkipTest()  # skip this test if LightGBM is not installed

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test \
      = train_test_split(X, y, test_size=0.2, shuffle=False)
    dtrain = lightgbm.Dataset(X_train, y_train, free_raw_data=False)
    dtest = lightgbm.Dataset(X_test, y_test, reference=dtrain, free_raw_data=False)
    param = {'task': 'train', 'boosting_type': 'gbdt',
             'metric': 'multi_logloss', 'num_class': 3,
             'num_leaves': 31, 'learning_rate': 0.05}
    num_round = 10
    watchlist = [dtrain, dtest]
    watchlist_names = ['train', 'test']
    for objective in ['multiclass', 'multiclassova']:
      param['objective'] = objective
      bst = lightgbm.train(param, dtrain, num_round, watchlist, watchlist_names)
      bst.save_model('./iris_lightgbm.txt')

      expected_pred = bst.predict(X_test)

      model = treelite.Model.load('./iris_lightgbm.txt', model_format='lightgbm')
      libpath = libname('./iris_{}{{}}'.format(objective))
      batch = treelite.runtime.Batch.from_npy2d(X_test)
      for toolchain in os_compatible_toolchains():
        model.export_lib(toolchain=toolchain, libpath=libpath,
                         params={}, verbose=True)
        predictor = treelite.runtime.Predictor(libpath=libpath, verbose=True)
        out_pred = predictor.predict(batch)
        assert_almost_equal(out_pred, expected_pred)

  def test_lightgbm_binary_classification(self):
    try:
      import lightgbm
    except ImportError:
      raise nose.SkipTest()  # skip this test if LightGBM is not installed

    dtrain_path = os.path.join(dpath, 'mushroom/agaricus.train')
    dtest_path = os.path.join(dpath, 'mushroom/agaricus.test')
    dtrain = lightgbm.Dataset(dtrain_path)
    dtest = lightgbm.Dataset(dtest_path, reference=dtrain)
    watchlist = [dtrain, dtest]
    watchlist_names = ['train', 'test']
    param = {'task': 'train', 'boosting_type': 'gbdt',
             'metric': 'auc', 'num_leaves': 7, 'learning_rate': 0.1}
    num_round = 10

    for objective in ['binary', 'xentlambda', 'xentropy']:
      param['objective'] = objective
      bst = lightgbm.train(param, dtrain, num_round, watchlist, watchlist_names)
      bst.save_model('./mushroom_lightgbm.txt')

      expected_prob = bst.predict(dtest_path)
      expected_margin = bst.predict(dtest_path, raw_score=True)

      model = treelite.Model.load('./mushroom_lightgbm.txt',
                                  model_format='lightgbm')
      libpath = libname('./agaricus_{}{{}}'.format(objective))
      batch = treelite.runtime.Batch.from_csr(treelite.DMatrix(dtest_path))
      for toolchain in os_compatible_toolchains():
        model.export_lib(toolchain=toolchain, libpath=libpath,
                         params={}, verbose=True)
        predictor = treelite.runtime.Predictor(libpath, verbose=True)
        out_prob = predictor.predict(batch)
        assert_almost_equal(out_prob, expected_prob)
        out_margin = predictor.predict(batch, pred_margin=True)
        assert_almost_equal(out_margin, expected_margin)

  def test_categorical_data(self):
    """
    LightGBM is able to produce categorical splits directly, so that
    categorical data don't have to be one-hot encoded. Test if Treelite is
    able to handle categorical splits.

    This toy example contains two features, both of which are categorical.
    The first has cardinality 3 and the second 5. The label was generated using
    the formula

       y = f(x0) + g(x1) + [noise with std=0.1]

    where f and g are given by the tables

       x0  f(x0)        x1  g(x1)
        0    -20         0     -2
        1    -10         1     -1
        2      0         2      0
                         3      1
                         4      2
    """

    for model_path, dtest_path, libname_fmt, \
        expected_prob_path, expected_margin_path, multiclass in \
        [('toy_categorical/toy_categorical_model.txt',
          'toy_categorical/toy_categorical.test', './toycat{}',
          None, 'toy_categorical/toy_categorical.test.pred', False)]:
      model_path = os.path.join(dpath, model_path)
      model = treelite.Model.load(model_path, model_format='lightgbm')
      for use_quantize in [False, True]:
        for use_parallel_comp in [None, 2]:
          run_pipeline_test(model=model, dtest_path=dtest_path,
                            libname_fmt=libname_fmt,
                            expected_prob_path=expected_prob_path,
                            expected_margin_path=expected_margin_path,
                            multiclass=multiclass, use_annotation=None,
                            use_quantize=use_quantize,
                            use_parallel_comp=use_parallel_comp)

  def test_sparse_categorical_model(self):
    """
    LightGBM is able to produce categorical splits directly, so that
    categorical data don't have to be one-hot encoded. Test if Treelite is
    able to handle categorical splits.

    This example produces a model with high-cardinality categorical variables.
    The training data has many missing values, so we need to match LightGBM
    when it comes to handling missing values
    """

    for model_path, dtest_path, libname_fmt, \
        expected_prob_path, expected_margin_path, multiclass in \
        [('sparse_categorical/sparse_categorical_model.txt',
          'sparse_categorical/sparse_categorical.test', './sparsecat{}',
          None, 'sparse_categorical/sparse_categorical.test.margin', False)]:
      model_path = os.path.join(dpath, model_path)
      model = treelite.Model.load(model_path, model_format='lightgbm')
      for use_quantize in [False, True]:
        for use_parallel_comp in [None, 2]:
          run_pipeline_test(model=model, dtest_path=dtest_path,
                            libname_fmt=libname_fmt,
                            expected_prob_path=expected_prob_path,
                            expected_margin_path=expected_margin_path,
                            multiclass=multiclass, use_annotation=None,
                            use_quantize=use_quantize,
                            use_parallel_comp=use_parallel_comp,
                            use_toolchains=['gcc'])
