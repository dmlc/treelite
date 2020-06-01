#!/bin/bash

set -eo pipefail

source $HOME/miniconda/bin/activate

if [ ${TASK} == "python_test" ]
then
  conda activate python3
  conda --version
  python --version

  # Build binary wheel
  set -x
  rm -rf build/
  mkdir build
  cd build
  cmake .. -DENABLE_PROTOBUF=ON
  make -j$(nproc)
  cd ..
  rm -rfv python/dist python/build
  cd python/
  python setup.py bdist_wheel --universal

  # Install Treelite into Python env
  ls -l ./dist/*.whl
  python -m pip install ./dist/*.whl
  cd ..

  # Run tests
  python -m pip install numpy scipy pandas pytest pytest-cov scikit-learn lightgbm coverage
  export GCC_PATH=gcc-7
  python -m pytest -v --fulltrace tests/python
fi
