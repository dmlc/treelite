#!/bin/bash

set -eo pipefail

if [ ${TASK} == "python_test" ] || [ ${TASK} == "python_sdist_test" ]
then
  if [ ${TRAVIS_OS_NAME} == "osx" ]; then
    wget -O conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
  else
    wget -O conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  fi
  bash conda.sh -b -p $HOME/miniconda
  source $HOME/miniconda/bin/activate
  hash -r
  conda config --set always_yes yes --set changeps1 no
  conda update -q conda
  conda info -a
  conda create -n python3 python=3.7
fi
