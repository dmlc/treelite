#!/bin/bash

set -euo pipefail

echo "##[section]Building Treelite..."
mkdir build
cd build
cmake .. -GNinja
ninja
cd ..

CURRENT_VERSION=$(cat python/treelite/VERSION)

echo "##[section]Testing serialization: 3.9 -> ${CURRENT_VERSION}"
pip install treelite==3.9.0 treelite_runtime==3.9.0
python tests/serializer/compatibility_tester.py --task save --checkpoint-path checkpoint.bin \
  --model-pickle-path model.pkl --expected-treelite-version 3.9.0
PYTHONPATH=./python/ python tests/serializer/compatibility_tester.py --task load \
  --checkpoint-path checkpoint.bin --model-pickle-path model.pkl \
  --expected-treelite-version ${CURRENT_VERSION}

echo "##[section]Testing serialization: ${CURRENT_VERSION} -> ${CURRENT_VERSION}"
PYTHONPATH=./python/ python tests/serializer/compatibility_tester.py --task save \
  --checkpoint-path checkpoint.bin --model-pickle-path model.pkl \
  --expected-treelite-version ${CURRENT_VERSION}
PYTHONPATH=./python/ python tests/serializer/compatibility_tester.py --task load \
  --checkpoint-path checkpoint.bin --model-pickle-path model.pkl \
  --expected-treelite-version ${CURRENT_VERSION}
