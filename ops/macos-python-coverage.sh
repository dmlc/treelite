#!/bin/bash

set -euo pipefail

echo "##[section]Setting up Python environment..."
conda create -n python3 -y -q -c conda-forge python=3.9 numpy scipy pandas pytest pytest-cov \
  scikit-learn coverage ninja lcov cmake llvm-openmp xgboost lightgbm
python -m pip install codecov

echo "##[section]Building Treelite..."
source activate python3
conda --version
python --version

# Run coverage test
set -x
rm -rf build/
mkdir build
cd build
cmake .. -DTEST_COVERAGE=ON -DBUILD_CPP_TEST=ON -GNinja
ninja
cd ..

./build/treelite_cpp_test
PYTHONPATH=./python:./runtime/python python -m pytest --cov=treelite --cov=treelite_runtime -v -rxXs \
  --fulltrace tests/python
lcov --directory . --capture --output-file coverage.info
lcov --remove coverage.info '*dmlccore*' --output-file coverage.info
lcov --remove coverage.info '*fmtlib*' --output-file coverage.info
lcov --remove coverage.info '*/usr/*' --output-file coverage.info
lcov --remove coverage.info '*googletest*' --output-file coverage.info
codecov
