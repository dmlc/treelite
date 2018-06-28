# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import treelite
import treelite.runtime
import unittest
import os
import json
from sys import platform as _platform
from treelite.common.util import TemporaryDirectory

dpath = os.path.abspath(os.path.join(os.getcwd(), 'tests/examples/'))

def load_txt(filename):
  content = []
  with open(filename, 'r') as f:
    for line in f:
      content.append(float(line))
  return np.array(content)

class TestBasic(unittest.TestCase):
  def test_basic(self):
    model_path = os.path.join(dpath, 'mushroom/mushroom.model')
    dmat_path = os.path.join(dpath, 'mushroom/agaricus.txt.test')
    libpath = './agaricus' + treelite.contrib._libext()
    model = treelite.Model.load(model_path, model_format='xgboost')
    dmat = treelite.DMatrix(dmat_path)
    batch = treelite.runtime.Batch.from_csr(dmat)

    expected_prob_path = os.path.join(dpath, 'mushroom/agaricus.txt.test.prob')
    expected_margin_path = os.path.join(dpath, 'mushroom/agaricus.txt.test.margin')
    expected_prob = load_txt(expected_prob_path)
    expected_margin = load_txt(expected_margin_path)

    if _platform == 'darwin':
      toolchains = ['gcc-7', 'clang']
    elif _platform == 'win32':
      toolchains = ['msvc']
    else:
      toolchains = ['gcc', 'clang']
    for toolchain in toolchains:
      model.export_lib(toolchain=toolchain, libpath=libpath, verbose=True)
      predictor = treelite.runtime.Predictor(libpath=libpath, verbose=True)
      out_prob = predictor.predict(batch)
      assert np.allclose(out_prob, expected_prob, atol=1e-11, rtol=1e-8)
      out_margin = predictor.predict(batch, pred_margin=True)
      assert np.allclose(out_margin, expected_margin, atol=1e-11, rtol=1e-8)
