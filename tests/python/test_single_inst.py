# -*- coding: utf-8 -*-
"""Tests for single-instance prediction"""
from __future__ import print_function
import os
import unittest
from nose.tools import nottest
from sklearn.datasets import load_svmlight_file
import numpy as np
import treelite
import treelite.runtime
from util import load_txt, os_compatible_toolchains, libname, make_annotation,\
                 assert_almost_equal

dpath = os.path.abspath(os.path.join(os.getcwd(), 'tests/examples/'))

class TestSingleInst(unittest.TestCase):
  @nottest
  def run_pipeline_test(self, model, dtest_path, libname_fmt,
                        expected_prob_path, expected_margin_path,
                        multiclass, use_annotation, use_quantize):
    dpath = os.path.abspath(os.path.join(os.getcwd(), 'tests/examples/'))
    dtest_path = os.path.join(dpath, dtest_path)
    libpath = libname(libname_fmt)
    X_test, _ = load_svmlight_file(dtest_path, zero_based=True)

    expected_prob_path = os.path.join(dpath, expected_prob_path)
    expected_margin_path = os.path.join(dpath, expected_margin_path)
    expected_prob = load_txt(expected_prob_path)
    expected_margin = load_txt(expected_margin_path)
    if multiclass:
      nrow = X_test.shape[0]
      expected_prob = expected_prob.reshape((nrow, -1))
      expected_margin = expected_margin.reshape((nrow, -1))
    params = {}
    if use_annotation is not None:
      params['annotate_in'] = use_annotation
    if use_quantize:
      params['quantize'] = 1

    for toolchain in os_compatible_toolchains():
      model.export_lib(toolchain=toolchain, libpath=libpath,
                       params=params, verbose=True)
      predictor = treelite.runtime.Predictor(libpath=libpath, verbose=True)
      for i in range(X_test.shape[0]):
        x = X_test[i,:]
        # Scipy CSR matrix
        out_prob = predictor.predict_instance(x)
        out_margin = predictor.predict_instance(x, pred_margin=True)
        assert_almost_equal(out_prob, expected_prob[i])
        assert_almost_equal(out_margin, expected_margin[i])
        # NumPy 1D array with 0 as missing value
        x = x.toarray().flatten()
        out_prob = predictor.predict_instance(x, missing=0.0)
        out_margin = predictor.predict_instance(x, missing=0.0, pred_margin=True)
        assert_almost_equal(out_prob, expected_prob[i])
        assert_almost_equal(out_margin, expected_margin[i])
        # NumPy 1D array with np.nan as missing value
        np.place(x, x == 0.0, [np.nan])
        out_prob = predictor.predict_instance(x, missing=np.nan)
        out_margin = predictor.predict_instance(x, missing=np.nan, pred_margin=True)
        assert_almost_equal(out_prob, expected_prob[i])
        assert_almost_equal(out_margin, expected_margin[i])
        # NumPy 1D array with np.nan as missing value
        # (default when `missing` parameter is unspecified)
        out_prob = predictor.predict_instance(x)
        out_margin = predictor.predict_instance(x, pred_margin=True)
        assert_almost_equal(out_prob, expected_prob[i])
        assert_almost_equal(out_margin, expected_margin[i])

  def test_single_inst(self):
    for model_path, dtrain_path, dtest_path, libname_fmt, \
        expected_prob_path, expected_margin_path, multiclass in \
        [('mushroom/mushroom.model', 'mushroom/agaricus.train',
          'mushroom/agaricus.test', './agaricus{}',
          'mushroom/agaricus.test.prob',
          'mushroom/agaricus.test.margin', False),
         ('dermatology/dermatology.model', 'dermatology/dermatology.train',
          'dermatology/dermatology.test', './dermatology{}',
          'dermatology/dermatology.test.prob',
          'dermatology/dermatology.test.margin', True)]:
      model_path = os.path.join(dpath, model_path)
      model = treelite.Model.load(model_path, model_format='xgboost')
      make_annotation(model=model, dtrain_path=dtrain_path,
                      annotation_path='./annotation.json')
      for use_annotation in ['./annotation.json', None]:
        for use_quantize in [True, False]:
          self.run_pipeline_test(model=model, dtest_path=dtest_path,
                                 libname_fmt=libname_fmt,
                                 expected_prob_path=expected_prob_path,
                                 expected_margin_path=expected_margin_path,
                                 multiclass=multiclass,
                                 use_annotation=use_annotation,
                                 use_quantize=use_quantize)
