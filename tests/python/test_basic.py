# -*- coding: utf-8 -*-
"""Suite of basic tests"""
from __future__ import print_function
import unittest
import os
import subprocess
from zipfile import ZipFile
import numpy as np
import treelite
import treelite.runtime
from util import load_txt, os_compatible_toolchains, os_platform, libname

dpath = os.path.abspath(os.path.join(os.getcwd(), 'tests/examples/'))

class TestBasic(unittest.TestCase):
  def test_basic(self):
    """
    Test a basic workflow: load a model, compile and export as shared lib,
    and make predictions
    """
    def run_test(model_path, dtrain_path, dtest_path, libname_fmt,
                 expected_prob_path, expected_margin_path,
                 multiclass, use_annotation, use_quantize):
      model_path = os.path.join(dpath, model_path)
      dtrain_path = os.path.join(dpath, dtrain_path)
      dtest_path = os.path.join(dpath, dtest_path)
      libpath = libname(libname_fmt)
      model = treelite.Model.load(model_path, model_format='xgboost')
      dtest = treelite.DMatrix(dtest_path)
      batch = treelite.runtime.Batch.from_csr(dtest)

      expected_prob_path = os.path.join(dpath, expected_prob_path)
      expected_margin_path = os.path.join(dpath, expected_margin_path)
      expected_prob = load_txt(expected_prob_path)
      expected_margin = load_txt(expected_margin_path)
      if multiclass:
        nrow = dtest.shape[0]
        expected_prob = expected_prob.reshape((nrow, -1))
        expected_margin = expected_margin.reshape((nrow, -1))
      params = {}
      if use_annotation:
        dtrain = treelite.DMatrix(dtrain_path)
        annotator = treelite.Annotator()
        annotator.annotate_branch(model=model, dmat=dtrain, verbose=True)
        annotator.save(path='./annotation.json')
        params['annotate_in'] = './annotation.json'
      if use_quantize:
        params['quantize'] = 1

      for toolchain in os_compatible_toolchains():
        model.export_lib(toolchain=toolchain, libpath=libpath,
                         params=params, verbose=True)
        predictor = treelite.runtime.Predictor(libpath=libpath, verbose=True)
        out_prob = predictor.predict(batch)
        assert np.allclose(out_prob, expected_prob, atol=1e-11, rtol=1e-8)
        out_margin = predictor.predict(batch, pred_margin=True)
        assert np.allclose(out_margin, expected_margin, atol=1e-11, rtol=1e-8)

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
      for use_annotation in [True, False]:
        for use_quantize in [True, False]:
          run_test(model_path=model_path, dtrain_path=dtrain_path,
                   dtest_path=dtest_path, libname_fmt=libname_fmt,
                   expected_prob_path=expected_prob_path,
                   expected_margin_path=expected_margin_path,
                   multiclass=multiclass, use_annotation=use_annotation,
                   use_quantize=use_quantize)

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

    dmat = treelite.DMatrix(dmat_path)
    batch = treelite.runtime.Batch.from_csr(dmat)

    expected_prob_path = os.path.join(dpath, 'mushroom/agaricus.test.prob')
    expected_prob = load_txt(expected_prob_path)
    out_prob = predictor.predict(batch)
    assert np.allclose(out_prob, expected_prob, atol=1e-11, rtol=1e-8)
