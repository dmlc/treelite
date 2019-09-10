# -*- coding: utf-8 -*-
"""Tests for reading/writing Protocol Buffers"""
from __future__ import print_function
import unittest
import os
import numpy as np
import treelite
import treelite.runtime
from util import run_pipeline_test, make_annotation

dpath = os.path.abspath(os.path.join(os.getcwd(), 'tests/examples/'))

class TestCodeFolding(unittest.TestCase):
  def test_code_folding(self):
    for model_format, model_path, dtrain_path, dtest_path, libname_fmt, \
        expected_prob_path, expected_margin_path, multiclass, \
        use_parallel_comp in \
        [('xgboost', 'mushroom/mushroom.model', 'mushroom/agaricus.train',
          'mushroom/agaricus.test', './agaricus{}', 'mushroom/agaricus.test.prob',
          'mushroom/agaricus.test.margin', False, 2),
         ('xgboost', 'dermatology/dermatology.model',
          'dermatology/dermatology.train', 'dermatology/dermatology.test',
          './dermatology{}', 'dermatology/dermatology.test.prob',
          'dermatology/dermatology.test.margin', True, None)]:
      model_path = os.path.join(dpath, model_path)
      model = treelite.Model.load(model_path, model_format=model_format)
      if dtrain_path is not None:
        make_annotation(model=model, dtrain_path=dtrain_path,
                        annotation_path='./annotation.json')
        use_annotation = './annotation.json'
      else:
        use_annotation = None
      for use_quantize in [False, True]:
        for use_code_folding in [0.0, 1.0, 2.0, 3.0]:
          run_pipeline_test(model=model, dtest_path=dtest_path,
                            libname_fmt=libname_fmt,
                            expected_prob_path=expected_prob_path,
                            expected_margin_path=expected_margin_path,
                            multiclass=multiclass, use_annotation=use_annotation,
                            use_quantize=use_quantize,
                            use_parallel_comp=use_parallel_comp,
                            use_code_folding=use_code_folding)
