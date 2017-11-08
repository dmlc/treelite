import treelite
import treelite.runtime
import xgboost
from fast_svmlight_loader import load_svmlight
import sys
import os
import numpy as np
import time

files = [('allstate/allstate0.train.libsvm',
          'allstate/allstate.json',
          'allstate/allstate.so',
          'allstate/allstate-1600-0.010000.model'),
         ('yahoo/yahoo.train',
          'yahoo/yahoo.json',
          'yahoo/yahoo.so',
          'yahoo/yahoo-1600-0.01.model')]

for datafile, annotfile, libfile, modfile in files:
  model = treelite.Model.load(modfile, 'xgboost')
  dmat = treelite.DMatrix(datafile)
  annotator = treelite.Annotator()
  annotator.annotate_branch(model, dmat, verbose=True)
  annotator.save(annotfile)
  model.export_lib(toolchain='gcc', libpath=libfile,
                   params={'quantize':1,
                           'annotate_in': annotfile},
                   verbose=True)

for datafile, _, libfile, modfile in files:
  print('Loading data file {}'.format(datafile))
  start = time.time()
  dmat = load_svmlight(filename=datafile, verbose=False)
  end = time.time()
  print('Done loading data file {} in {} sec'.format(datafile, end-start))
  X = dmat['data']

  dtrain = xgboost.DMatrix(datafile)

  predictor = treelite.runtime.Predictor(libfile, verbose=True)
  bst = xgboost.Booster()
  bst.load_model(modfile)

  nrow = X.shape[0]
  for batchsize in np.logspace(np.log10(100), np.log10(nrow), 10).astype(np.int):
    for i in range(100):
      rbegin = np.random.randint(0, nrow - batchsize + 1)
      rend = rbegin + batchsize
      batch = treelite.runtime.Batch.from_csr(X, rbegin, rend)
      dtrain_slice = dtrain.slice(np.arange(rbegin, rend).tolist())

      start = time.time()
      predictor.predict(batch, verbose=True)
      end = time.time()
      print('batchsize = {}, treelite: {} sec'.format(batchsize, end-start))

      start = time.time()
      bst.predict(dtrain_slice)
      end = time.time()
      print('batchsize = {}, XGBoost: {} sec'.format(batchsize, end-start))
