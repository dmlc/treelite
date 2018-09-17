# -*- coding: utf-8 -*-
"""Suite of basic tests"""
from __future__ import print_function
import unittest
import os
import subprocess
from zipfile import ZipFile
import numpy as np
from sklearn.datasets import load_svmlight_file
import treelite
import treelite.runtime
from util import load_txt, os_compatible_toolchains, os_platform, libname, \
                 run_pipeline_test, make_annotation, assert_almost_equal

dpath = os.path.abspath(os.path.join(os.getcwd(), 'tests/examples/'))

class TestBasic(unittest.TestCase):
  def test_basic(self):
    """
    Test a basic workflow: load a model, compile and export as shared lib,
    and make predictions
    """
    for model_path, dtrain_path, dtest_path, libname_fmt, \
        expected_prob_path, expected_margin_path, multiclass in \
        [('mushroom/mushroom.model', 'mushroom/agaricus.train',
          'mushroom/agaricus.test', './agaricus{}',
          'mushroom/agaricus.test.prob',
          'mushroom/agaricus.test.margin', False),
         ('dermatology/dermatology.model', 'dermatology/dermatology.train',
          'dermatology/dermatology.test', './dermatology{}',
          'dermatology/dermatology.test.prob',
          'dermatology/dermatology.test.margin', True),
         ('letor/mq2008.model', 'letor/mq2008.train',
          'letor/mq2008.test', './mq2008{}',
          None, 'letor/mq2008.test.pred', False)]:
      model_path = os.path.join(dpath, model_path)
      model = treelite.Model.load(model_path, model_format='xgboost')
      make_annotation(model=model, dtrain_path=dtrain_path,
                      annotation_path='./annotation.json')
      for use_annotation in ['./annotation.json', None]:
        for use_quantize in [True, False]:
          for use_parallel_comp in [None, 2]:
            run_pipeline_test(model=model, dtest_path=dtest_path,
                              libname_fmt=libname_fmt,
                              expected_prob_path=expected_prob_path,
                              expected_margin_path=expected_margin_path,
                              multiclass=multiclass, use_annotation=use_annotation,
                              use_quantize=use_quantize,
                              use_parallel_comp=use_parallel_comp)

  def test_srcpkg(self):
    """Test feature to export a source tarball"""
    model_path = os.path.join(dpath, 'mushroom/mushroom.model')
    dmat_path = os.path.join(dpath, 'mushroom/agaricus.test')
    libpath = libname('./mushroom/mushroom{}')
    model = treelite.Model.load(model_path, model_format='xgboost')

    toolchain = os_compatible_toolchains()[0]
    model.export_srcpkg(platform=os_platform(), toolchain=toolchain,
                        pkgpath='./srcpkg.zip', libname=libpath,
                        params={}, verbose=True)
    with ZipFile('./srcpkg.zip', 'r') as zip_ref:
      zip_ref.extractall('.')
    subprocess.call(['make', '-C', 'mushroom'])

    predictor = treelite.runtime.Predictor(libpath='./mushroom', verbose=True)

    X, _ = load_svmlight_file(dmat_path, zero_based=True)
    dmat = treelite.DMatrix(X)
    batch = treelite.runtime.Batch.from_csr(dmat)

    expected_prob_path = os.path.join(dpath, 'mushroom/agaricus.test.prob')
    expected_prob = load_txt(expected_prob_path)
    out_prob = predictor.predict(batch)
    assert_almost_equal(out_prob, expected_prob)
