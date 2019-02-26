#!/bin/bash

if [ ${TASK} == "lint" ]; then
  make lint || exit -1
fi

if [ ${TASK} == "python_test" ]; then
  NPROC=2 make all || exit -1
  echo "-------------------------------"
  source activate python3
  python --version
  conda install numpy scipy pandas nose scikit-learn
  if [ ${TRAVIS_OS_NAME} == "linux" ]; then
    python -m pip install xgboost  # Run XGBoost/LightGBM integration only on Linux
    python -m pip install --no-binary :all: lightgbm
  elif [ ${TRAVIS_OS_NAME} == "osx" ]; then
    export GCC_PATH=gcc-7
  fi

  python -m pip install coverage codecov
  python -m nose tests/python --with-coverage || exit -1
  python -m codecov

  source activate python2
  echo "-------------------------------"
  python --version
  conda install numpy scipy pandas nose scikit-learn
  if [ ${TRAVIS_OS_NAME} == "linux" ]; then
    python -m pip install xgboost  # Run XGBoost/LightGBM integration only on Linux
    python -m pip install --no-binary :all: lightgbm
  fi
  python -m pip install coverage codecov
  python -m nose tests/python --with-coverage || exit -1
  python -m codecov

  exit 0
fi

if [ ${TASK} == "cpp_test" ]; then
  NPROC=2 make cpp-coverage || exit -1
  # use Python tests to get initial C++ coverage
  source activate python3
  conda install numpy scipy pandas nose scikit-learn
  if [ ${TRAVIS_OS_NAME} == "linux" ]; then
    python -m pip install xgboost  # Run XGBoost/LightGBM integration only on Linux
    python -m pip install --no-binary :all: lightgbm
  fi
  python -m nose tests/python || exit -1
  # capture coverage info
  lcov --directory . --capture --output-file coverage.info
  # filter system and 3rd-party headers
  lcov --remove coverage.info '/usr/*' --output-file coverage.info
  lcov --remove coverage.info '*3rdparty*' --output-file coverage.info
  lcov --remove coverage.info '*dmlc-core*' --output-file coverage.info
  # Uploading report to Codecov
  bash <(curl -s https://codecov.io/bash) || echo "Codecov did not collect coverage reports"
fi

if [ ${TASK} == "java_test" ]; then
  cd runtime/java/treelite4j
  mvn test -DJNI.args=cpp-coverage || exit -1
  # capture coverage info
  lcov --directory . --capture --output-file coverage.info
  # filter system and 3rd-party headers
  lcov --remove coverage.info '/usr/*' --output-file coverage.info
  lcov --remove coverage.info '*3rdparty*' --output-file coverage.info
  lcov --remove coverage.info '*dmlc-core*' --output-file coverage.info
  # Uploading report to Codecov
  bash <(curl -s https://codecov.io/bash) || echo "Codecov did not collect coverage reports"
fi
