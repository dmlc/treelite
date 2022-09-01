#!/bin/bash

set -euo pipefail

echo "##[section]Building Treelite..."
mkdir build
cd build
cmake .. -GNinja
ninja
cd ..

echo "##[section]Setting up Python environment..."
conda create -n dev -y -q -c conda-forge python=3.9 numpy scipy pandas pytest scikit-learn awscli
source activate dev
pip install treelite==2.4.0 treelite_runtime==2.4.0

CURRENT_VERSION=$(cat python/treelite/VERSION)

echo "##[section]Testing backward compatibility of serialization: 2.4 -> ${CURRENT_VERSION}"
python tests/cython/compatibility_tester.py --task save --checkpoint-path checkpoint.bin \
  --model-pickle-path model.pkl --expected-treelite-version 2.4.0
PYTHONPATH=./python/:./runtime/python/ python tests/cython/compatibility_tester.py --task load \
  --checkpoint-path checkpoint.bin --model-pickle-path model.pkl \
  --expected-treelite-version ${CURRENT_VERSION}
