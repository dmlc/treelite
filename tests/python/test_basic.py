# -*- coding: utf-8 -*-
"""Suite of basic tests"""
from __future__ import print_function
import numpy as np
import treelite
import treelite.runtime
import unittest
import os
import json
import subprocess
from zipfile import ZipFile
from treelite.common.util import TemporaryDirectory
from util import load_txt, os_compatible_toolchains, os_platform, libname

dpath = os.path.abspath(os.path.join(os.getcwd(), 'tests/examples/'))

class TestBasic(unittest.TestCase):
  def test_basic(self):
    """
    Test a basic workflow: load a model, compile and export as shared lib,
    and make predictions
    """
    model_path = os.path.join(dpath, 'mushroom/mushroom.model')
    dmat_path = os.path.join(dpath, 'mushroom/agaricus.txt.test')
    libpath = libname('./agaricus{}')
    model = treelite.Model.load(model_path, model_format='xgboost')
    dmat = treelite.DMatrix(dmat_path)
    batch = treelite.runtime.Batch.from_csr(dmat)

    expected_prob_path = os.path.join(dpath, 'mushroom/agaricus.txt.test.prob')
    expected_margin_path = os.path.join(dpath, 'mushroom/agaricus.txt.test.margin')
    expected_prob = load_txt(expected_prob_path)
    expected_margin = load_txt(expected_margin_path)

    for toolchain in os_compatible_toolchains():
      model.export_lib(toolchain=toolchain, libpath=libpath, verbose=True)
      predictor = treelite.runtime.Predictor(libpath=libpath, verbose=True)
      out_prob = predictor.predict(batch)
      assert np.allclose(out_prob, expected_prob, atol=1e-11, rtol=1e-8)
      out_margin = predictor.predict(batch, pred_margin=True)
      assert np.allclose(out_margin, expected_margin, atol=1e-11, rtol=1e-8)

  def test_srcpkg(self):
    """Test feature to export a source tarball"""
    model_path = os.path.join(dpath, 'mushroom/mushroom.model')
    dmat_path = os.path.join(dpath, 'mushroom/agaricus.txt.test')
    libpath = libname('./mushroom/mushroom{}')
    model = treelite.Model.load(model_path, model_format='xgboost')

    toolchain = os_compatible_toolchains()[0]
    model.export_srcpkg(platform=os_platform(), toolchain=toolchain,
      pkgpath='./srcpkg.zip', libname=libpath, params={}, verbose=True)
    with ZipFile('./srcpkg.zip', 'r') as zip_ref:
      zip_ref.extractall('.')
    subprocess.call(['make', '-C', 'mushroom'])

    predictor = treelite.runtime.Predictor(libpath=libpath, verbose=True)

    dmat = treelite.DMatrix(dmat_path)
    batch = treelite.runtime.Batch.from_csr(dmat)

    expected_prob_path = os.path.join(dpath, 'mushroom/agaricus.txt.test.prob')
    expected_prob = load_txt(expected_prob_path)
    out_prob = predictor.predict(batch)
    assert np.allclose(out_prob, expected_prob, atol=1e-11, rtol=1e-8)
