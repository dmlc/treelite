# -*- coding: utf-8 -*-
"""Suite of basic tests"""
from __future__ import print_function
import unittest
import sys
import os
import subprocess
from zipfile import ZipFile
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
import treelite
import treelite.runtime
import pytest
from util import load_txt, os_compatible_toolchains, os_platform, libname, \
                 run_pipeline_test, make_annotation, assert_almost_equal

dpath = os.path.abspath(os.path.join(os.getcwd(), 'tests/examples/'))

class TestBasic(unittest.TestCase):
  def test_basic(self):
    """
    Test a basic workflow: load a model, compile and export as shared lib,
    and make predictions
    """

    is_linux = sys.platform.startswith('linux')

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
          for use_parallel_comp in [None, 2]:
            run_pipeline_test(model=model, dtest_path=dtest_path,
                              libname_fmt=libname_fmt,
                              expected_prob_path=expected_prob_path,
                              expected_margin_path=expected_margin_path,
                              multiclass=multiclass, use_annotation=use_annotation,
                              use_quantize=use_quantize,
                              use_parallel_comp=use_parallel_comp)
      for use_elf in [True, False] if is_linux else [False]:
        run_pipeline_test(model=model, dtest_path=dtest_path,
                          libname_fmt=libname_fmt,
                          expected_prob_path=expected_prob_path,
                          expected_margin_path=expected_margin_path,
                          multiclass=multiclass, use_elf=use_elf,
                          use_compiler='failsafe')
      if not is_linux:
        # Expect to see an exception when using ELF in non-Linux OS
        with pytest.raises(treelite.common.util.TreeliteError):
          run_pipeline_test(model=model, dtest_path=dtest_path,
                            libname_fmt=libname_fmt,
                            expected_prob_path=expected_prob_path,
                            expected_margin_path=expected_margin_path,
                            multiclass=multiclass, use_elf=True,
                            use_compiler='failsafe')

    # LETOR
    model_path = os.path.join(dpath, 'letor/mq2008.model')
    model = treelite.Model.load(model_path, model_format='xgboost')
    make_annotation(model=model, dtrain_path='letor/mq2008.train',
                    annotation_path='./annotation.json')
    run_pipeline_test(model=model, dtest_path='letor/mq2008.test',
                      libname_fmt='./mq2008{}',
                      expected_prob_path=None,
                      expected_margin_path='letor/mq2008.test.pred',
                      multiclass=False, use_annotation='./annotation.json',
                      use_quantize=1, use_parallel_comp=700,
                      use_toolchains=['msvc' if os_platform() == 'windows' else 'gcc'])
    run_pipeline_test(model=model, dtest_path='letor/mq2008.test',
                      libname_fmt='./mq2008{}',
                      expected_prob_path=None,
                      expected_margin_path='letor/mq2008.test.pred',
                      multiclass=False, use_elf=is_linux,
                      use_compiler='failsafe')

  @pytest.mark.skipif(os_platform() == 'windows', reason='Make unavailable on Windows')
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

  def test_deficient_matrix(self):
    """
    Test if Treelite correctly handles sparse matrix with fewer columns
    than the training data used for the model. In this case, the matrix
    should be padded with zeros.
    """
    model_path = os.path.join(dpath, 'mushroom/mushroom.model')
    libpath = libname('./mushroom{}')
    model = treelite.Model.load(model_path, model_format='xgboost')
    toolchain = os_compatible_toolchains()[0]
    model.export_lib(toolchain=toolchain, libpath=libpath,
                     params={'quantize': 1}, verbose=True)
    X = csr_matrix(([], ([], [])), shape=(3, 3))
    batch = treelite.runtime.Batch.from_csr(X)
    predictor = treelite.runtime.Predictor(libpath=libpath, verbose=True)
    predictor.predict(batch)  # should not crash

  def test_too_wide_matrix(self):
    """
    Test if Treelite correctly handles sparse matrix with more columns
    than the training data used for the model. In this case, an exception
    should be thrown
    """
    model_path = os.path.join(dpath, 'mushroom/mushroom.model')
    libpath = libname('./mushroom{}')
    model = treelite.Model.load(model_path, model_format='xgboost')
    toolchain = os_compatible_toolchains()[0]
    model.export_lib(toolchain=toolchain, libpath=libpath,
                     params={'quantize': 1}, verbose=True)
    X = csr_matrix(([], ([], [])), shape=(3, 1000))
    batch = treelite.runtime.Batch.from_csr(X)
    predictor = treelite.runtime.Predictor(libpath=libpath, verbose=True)
    import treelite_runtime
    err = treelite_runtime.common.util.TreeliteError
    pytest.raises(err, predictor.predict, batch)  # should crash

  def test_tree_limit_setting(self):
    """
    Test Model.set_tree_limit
    """
    model_path = os.path.join(dpath, 'mushroom/mushroom.model')
    model = treelite.Model.load(model_path, model_format='xgboost')
    assert(model.num_tree == 2)
    pytest.raises(Exception, model.set_tree_limit, 0)
    pytest.raises(Exception, model.set_tree_limit, 3)
    model.set_tree_limit(1)
    assert(model.num_tree == 1)

    model_path = os.path.join(dpath, 'dermatology/dermatology.model')
    model = treelite.Model.load(model_path, model_format='xgboost')
    assert(model.num_tree == 60)
    model.set_tree_limit(30)
    assert(model.num_tree == 30)
    model.set_tree_limit(10)
    assert(model.num_tree == 10)
