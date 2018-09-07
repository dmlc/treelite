# -*- coding: utf-8 -*-
"""Performance test for single-instance prediction using Iris data"""
from __future__ import print_function
from sklearn.datasets import load_iris
import numpy as np
import xgboost
import treelite
import treelite.runtime
import importlib.util
import os
import time

def test_iris_single_inst():
  spec = importlib.util.spec_from_file_location(
    'util',
    os.path.join(os.path.dirname(__file__), os.pardir, 'python', 'util.py'))
  util = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(util)

  X, y = load_iris(return_X_y=True)
  dtrain = xgboost.DMatrix(X, label=y)
  param = {'max_depth': 6, 'eta': 1, 'silent': 1, 'objective': 'multi:softprob',
           'num_class': 3}
  num_round = 10
  watchlist = [(dtrain, 'train')]
  bst = xgboost.train(param, dtrain, num_round, watchlist)

  model = treelite.Model.from_xgboost(bst)
  libpath = util.libname('./iris{}')
  toolchain = util.os_compatible_toolchains()[0]
  model.export_lib(toolchain=toolchain, libpath=libpath, params={'quantize': 1})
  predictor = treelite.runtime.Predictor(libpath=libpath)

  print('1. Using data matrix slices')
  record = []
  for _ in range(100):
    tstart = time.time()
    for i in range(X.shape[0]):
      out_prob = predictor.predict_instance(X[i,:])
    tend = time.time()
    record.append(tend - tstart)
  print('Processed {} data instances in {} seconds on average (std = {})'\
    .format(X.shape[0], np.mean(record), np.std(record)))

  print('2. Using dictionaries')
  data = [{k: X[i, k] for k in range(4)} for i in range(X.shape[0])]
  record = []
  for _ in range(100):
    tstart = time.time()
    for i in range(X.shape[0]):
      out_prob = predictor.predict_instance(data[i])
    tend = time.time()
    record.append(tend - tstart)
  print('Processed {} data instances in {} seconds on average (std = {})'\
    .format(X.shape[0], np.mean(record), np.std(record)))

if __name__ == '__main__':
  test_iris_single_inst()
