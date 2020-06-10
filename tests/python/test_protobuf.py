# -*- coding: utf-8 -*-
"""Tests for reading/writing Protocol Buffers"""
from __future__ import print_function
import unittest
import os
import numpy as np
import treelite
import treelite_runtime
from util import run_pipeline_test

dpath = os.path.abspath(os.path.join(os.getcwd(), 'tests/examples/'))

class TestProtobuf(unittest.TestCase):
  def test_round_trip(self):
    for model_format, model_path, dtest_path, libname_fmt, \
        expected_prob_path, expected_margin_path, multiclass in \
        [('xgboost', 'mushroom/mushroom.model', 'mushroom/agaricus.test',
          './agaricus{}', 'mushroom/agaricus.test.prob',
          'mushroom/agaricus.test.margin', False),
         ('xgboost', 'dermatology/dermatology.model',
          'dermatology/dermatology.test', './dermatology{}',
          'dermatology/dermatology.test.prob',
          'dermatology/dermatology.test.margin', True),
         ('lightgbm', 'toy_categorical/toy_categorical_model.txt',
          'toy_categorical/toy_categorical.test', './toycat{}',
          None, 'toy_categorical/toy_categorical.test.pred', False)]:
      model_path = os.path.join(dpath, model_path)
      model = treelite.Model.load(model_path, model_format=model_format)
      model.export_protobuf('./my.buffer')
      model2 = treelite.Model.load('./my.buffer', model_format='protobuf')
      for use_quantize in [False, True]:
        run_pipeline_test(model=model2, dtest_path=dtest_path,
                          libname_fmt=libname_fmt,
                          expected_prob_path=expected_prob_path,
                          expected_margin_path=expected_margin_path,
                          multiclass=multiclass, use_annotation=None,
                          use_quantize=use_quantize, use_parallel_comp=None)
