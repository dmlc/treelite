#!/bin/bash

set -eo pipefail

if [ ${TRAVIS_OS_NAME} == "osx" ]; then
  brew update
  # Use libomp 11.1.0: https://github.com/dmlc/xgboost/issues/7039
  wget https://raw.githubusercontent.com/Homebrew/homebrew-core/679923b4eb48a8dc7ecc1f05d06063cd79b3fc00/Formula/libomp.rb -O $(find $(brew --repository) -name libomp.rb)
  brew install cmake libomp lcov ninja
  brew pin libomp

  wget -O conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
else
  wget -O conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  sudo rm -rf /usr/local/cmake-3.12.4  # Remove old CMake installed in Travis CI Linux worker
  cmake --version
fi
bash conda.sh -b -p $HOME/miniconda
source $HOME/miniconda/bin/activate
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create -n python3 python=3.7
