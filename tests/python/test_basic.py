# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import treelite
import treelite.runtime
import unittest
import os
import json
from treelite.common.util import TemporaryDirectory

dpath = os.path.abspath(os.path.join(os.getcwd(), 'tests/examples/'))

class TestBasic(unittest.TestCase):
  def test_basic(self):
    try:
      model_path = os.path.join(dpath, 'mushroom/mushroom.model')
      dmat_path = os.path.join(dpath, 'mushroom/agaricus.txt.test')
      libpath = './agaricus' + treelite.contrib._libext()
      model = treelite.Model.load(model_path, model_format='xgboost')
      model.export_lib(toolchain='gcc', libpath=libpath, verbose=True)
      predictor = treelite.runtime.Predictor(libpath=libpath, verbose=True)
      dmat = treelite.DMatrix(dmat_path)
      batch = treelite.runtime.Batch.from_csr(dmat)
      out_pred = predictor.predict(batch)
    except Exception as e:
      with TemporaryDirectory() as tempdir:
        model.compile(dirpath=tempdir, params={}, verbose=True)
        with open(os.path.join(tempdir, 'main.c')) as f:
          for line in f:
            print(line, end='')
      raise
