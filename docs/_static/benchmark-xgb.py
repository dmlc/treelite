import xgboost
import sys
import os
import numpy as np
import time

datafile = sys.argv[1]
libfile = sys.argv[2]
modfile = sys.argv[3]

print('Loading data file {}'.format(datafile))
start = time.time()
dtrain = xgboost.DMatrix(datafile)
end = time.time()
print('Done loading data file {} in {} sec'.format(datafile, end-start))
bst = xgboost.Booster()
bst.load_model(modfile)
nrow = dtrain.num_row()
for batchsize in np.logspace(np.log10(100), 5, 10).astype(np.int):
  print('*** batchsize = {}'.format(batchsize))
  for i in range(100):
    rbegin = np.random.randint(0, nrow - batchsize + 1)
    rend = rbegin + batchsize
    dtrain_slice = dtrain.slice(np.arange(rbegin, rend).tolist())
    bst.predict(dtrain_slice, output_margin=True)
