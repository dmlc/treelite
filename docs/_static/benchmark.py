import treelite
import treelite_runtime
from fast_svmlight_loader import load_svmlight
import sys
import os
import numpy as np
import time

datafile = sys.argv[1]
libfile = sys.argv[2]
modfile = sys.argv[3]

print('Loading data file {}'.format(datafile))
start = time.time()
dmat = load_svmlight(filename=datafile, verbose=False)
end = time.time()
print('Done loading data file {} in {} sec'.format(datafile, end-start))
X = dmat['data']
predictor = treelite_runtime.Predictor(libfile, verbose=True, include_master_thread=True)
nrow = X.shape[0]
for batchsize in np.logspace(np.log10(100), 5, 10).astype(np.int):
  print('*** batchsize = {}'.format(batchsize))
  for i in range(300):
    rbegin = np.random.randint(0, nrow - batchsize + 1)
    rend = rbegin + batchsize
    dmat = treelite_runtime.DMatrix(X[rbegin:rend])
    predictor.predict(dmat, pred_margin=True)
