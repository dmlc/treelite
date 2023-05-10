#!/bin/bash

set -euo pipefail

echo "##[section]Installing lcov and Ninja..."
sudo apt-get install lcov ninja-build

echo "##[section]Building Treelite..."
mkdir build/
cd build/
cmake .. -DTEST_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug -DBUILD_CPP_TEST=ON -GNinja
ninja
cd ..

echo "##[section]Running Google C++ tests..."
./build/treelite_cpp_test

echo "##[section]Build Cython extension..."
cd tests/cython
python setup.py build_ext --inplace
cd ../..

echo "##[section]Running Python integration tests..."
export PYTHONPATH='./python'
python -m pytest --cov=treelite -v -rxXs --fulltrace --durations=0 tests/python tests/cython

echo "##[section]Collecting coverage data..."
lcov --directory . --capture --output-file coverage.info
lcov --remove coverage.info '*/usr/*' --output-file coverage.info
lcov --remove coverage.info '*/build/_deps/*' --output-file coverage.info

echo "##[section]Submitting code coverage data to CodeCov..."
bash <(curl -s https://codecov.io/bash) -X gcov || echo "Codecov did not collect coverage reports"
