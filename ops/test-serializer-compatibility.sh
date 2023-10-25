#!/bin/bash

set -euo pipefail

# Build XGBoost from the source
# Don't use XGBoost from pip because its use of libomp clashes with
# llvm-openmp from Conda. TODO(hcho3): Remove this once XGBoost 2.0
# is available on Conda.
echo "##[section]Building XGBoost..."
if ! conda list | grep -q xgboost
then
  git clone --recursive https://github.com/dmlc/xgboost -b v2.0.0
  cd xgboost
  pip install -vvv python-package/
  cd ..
fi

echo "##[section]Building Treelite..."
mkdir build
cd build
cmake .. -GNinja
ninja
cd ..

CURRENT_VERSION=$(cat python/treelite/VERSION)

echo "##[section]Testing serialization: 3.9 -> ${CURRENT_VERSION}"
pip install treelite==3.9.0 treelite_runtime==3.9.0
python tests/cython/compatibility_tester.py --task save --checkpoint-path checkpoint.bin \
  --model-pickle-path model.pkl --expected-treelite-version 3.9.0
PYTHONPATH=./python/ python tests/cython/compatibility_tester.py --task load \
  --checkpoint-path checkpoint.bin --model-pickle-path model.pkl \
  --expected-treelite-version ${CURRENT_VERSION}

echo "##[section]Testing serialization: ${CURRENT_VERSION} -> ${CURRENT_VERSION}"
PYTHONPATH=./python/ python tests/cython/compatibility_tester.py --task save \
  --checkpoint-path checkpoint.bin --model-pickle-path model.pkl \
  --expected-treelite-version ${CURRENT_VERSION}
PYTHONPATH=./python/ python tests/cython/compatibility_tester.py --task load \
  --checkpoint-path checkpoint.bin --model-pickle-path model.pkl \
  --expected-treelite-version ${CURRENT_VERSION}
