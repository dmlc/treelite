#!/bin/bash

if [ ${TASK} == "lint" ]; then
  make lint || exit -1
  echo "Check documentations..."
  make doxygen 2>log.txt
  (cat log.txt| grep -v ENABLE_PREPROCESSING |grep -v "unsupported tag") > logclean.txt
  echo "---------Error Log----------"
  cat logclean.txt
  echo "----------------------------"
  (cat logclean.txt|grep warning) && exit -1
  (cat logclean.txt|grep error) && exit -1
fi

if [ ${TASK} == "python_test" ]; then
  NRPOC=2 make all || exit -1
  echo "-------------------------------"
  source activate python3
  python --version
  conda install numpy scipy pandas nose scikit-learn

  python -m pip install coverage codecov
  python -m nose tests/python --with-coverage || exit -1
  codecov
  source activate python2
  echo "-------------------------------"
  python --version
  conda install numpy scipy pandas nose scikit-learn
  python -m nose tests/python || exit -1
  exit 0
fi
