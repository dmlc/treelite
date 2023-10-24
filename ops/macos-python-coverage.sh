#!/bin/bash

set -euo pipefail

conda --version
python --version

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

# Run coverage test
echo "##[section]Building Treelite..."
set -x
rm -rf build/
mkdir build
cd build
cmake .. -DTEST_COVERAGE=ON -DBUILD_CPP_TEST=ON -GNinja
ninja -v
cd ..

./build/treelite_cpp_test
PYTHONPATH=./python python -m pytest --cov=treelite -v -rxXs \
  --fulltrace --durations=0 tests/python
lcov --directory . --capture --output-file coverage.info
lcov --remove coverage.info '*dmlccore*' --output-file coverage.info
lcov --remove coverage.info '*fmtlib*' --output-file coverage.info
lcov --remove coverage.info '*/usr/*' --output-file coverage.info
lcov --remove coverage.info '*googletest*' --output-file coverage.info
codecov
