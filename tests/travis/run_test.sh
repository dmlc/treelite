#!/bin/bash

if [ ${TASK} == "lint" ]; then
  make lint || exit -1
fi

if [ ${TASK} == "python_test" ]; then
  NRPOC=2 make all || exit -1
  echo "-------------------------------"
  source activate python3
  python --version
  conda install numpy scipy pandas nose scikit-learn
  pip install xgboost

  python -m pip install coverage codecov
  python -m nose tests/python --with-coverage || exit -1
  codecov
  source activate python2
  echo "-------------------------------"
  python --version
  conda install numpy scipy pandas nose scikit-learn
  pip install xgboost
  python -m nose tests/python || exit -1
  exit 0
fi
