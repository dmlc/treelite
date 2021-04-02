#!/bin/bash

set -eo pipefail

source $HOME/miniconda/bin/activate

if [ ${TASK} == "python_coverage_test" ]
then
  conda activate python3
  conda --version
  python --version

  # Run coverage test
  set -x
  rm -rf build/
  mkdir build
  cd build
  cmake .. -DTEST_COVERAGE=ON -DUSE_OPENMP=ON -DBUILD_CPP_TEST=ON -GNinja
  ninja
  cd ..
  conda install -c conda-forge numpy scipy pandas pytest pytest-cov scikit-learn coverage
  python -m pip install --pre xgboost
  python -m pip install lightgbm codecov
  ./build/treelite_cpp_test
  PYTHONPATH=./python:./runtime/python python -m pytest --cov=treelite --cov=treelite_runtime -v --fulltrace tests/python
  lcov --directory . --capture --output-file coverage.info
  lcov --remove coverage.info '*dmlccore*' --output-file coverage.info
  lcov --remove coverage.info '*fmtlib*' --output-file coverage.info
  lcov --remove coverage.info '*/usr/*' --output-file coverage.info
  lcov --remove coverage.info '*googletest*' --output-file coverage.info
  codecov
fi

if [ ${TASK} == "cmake_import_test" ]
then
  conda activate python3
  conda --version
  python --version

  # Install Treelite C++ library into the Conda env
  set -x
  rm -rf build/
  mkdir build
  cd build
  cmake .. -DUSE_OPENMP=ON -DBUILD_STATIC_LIBS=ON -GNinja
  ninja install

  # Try compiling a sample application
  cd ../tests/example_app/
  rm -rf build/
  mkdir build
  cd build
  cmake .. -GNinja
  ninja
  ./example
fi

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
  cmake .. -DUSE_OPENMP=ON -GNinja
  ninja
  cd ..
  rm -rfv python/dist python/build
  cd python/
  python setup.py bdist_wheel --universal
  TAG=macosx_10_14_x86_64.macosx_10_15_x86_64.macosx_11_0_x86_64
  python ../tests/ci_build/rename_whl.py dist/*.whl ${TRAVIS_COMMIT} ${TAG}
  cd ..
  rm -rfv runtime/python/dist runtime/python/build
  cd runtime/python/
  python setup.py bdist_wheel --universal
  python ../../tests/ci_build/rename_whl.py dist/*.whl ${TRAVIS_COMMIT} ${TAG}
  cd ../..

  # Install Treelite into Python env
  ls -l ./python/dist/*.whl
  ls -l ./runtime/python/dist/*.whl
  python -m pip install ./python/dist/treelite-*-py3-none-${TAG}.whl
  python -m pip install ./runtime/python/dist/treelite_runtime-*-py3-none-${TAG}.whl

  # Run tests
  conda install -c conda-forge numpy scipy pandas pytest scikit-learn coverage
  python -m pip install --pre xgboost
  python -m pip install lightgbm
  python -m pytest -v --fulltrace tests/python

  # Deploy binary wheel to S3
  python -m pip install awscli
  if [ "${TRAVIS_BRANCH}" == "mainline" ]
  then
    S3_DEST="s3://treelite-wheels/"
  elif [ -z "${TRAVIS_TAG}" ]
  then
    S3_DEST="s3://treelite-wheels/${TRAVIS_BRANCH}/"
  fi
  python -m awscli s3 cp python/dist/treelite-*.whl "${S3_DEST}" --acl public-read || true
  python -m awscli s3 cp runtime/python/dist/treelite_runtime*.whl "${S3_DEST}" --acl public-read || true
fi

if [ ${TASK} == "python_sdist_test" ]; then
  conda activate python3
  python --version
  conda install numpy scipy

  # Build source distribution
  make pippack

  # Install Treelite into Python env
  python -m pip install -v treelite-*.tar.gz
  python -m pip install -v treelite_runtime-*.tar.gz

  # Run tests
  conda install -c conda-forge numpy scipy pandas pytest scikit-learn coverage
  python -m pip install --pre xgboost
  python -m pip install lightgbm
  python -m pytest -v --fulltrace tests/python

  # Deploy source wheel to S3
  python -m pip install awscli
  if [ "${TRAVIS_BRANCH}" == "mainline" ]
  then
    S3_DEST="s3://treelite-wheels/"
  elif [ -z "${TRAVIS_TAG}" ]
  then
    S3_DEST="s3://treelite-wheels/${TRAVIS_BRANCH}/"
  fi
  for file in ./treelite-*.tar.gz ./treelite_runtime-*.tar.gz
  do
    mv "${file}" "${file%.tar.gz}+${TRAVIS_COMMIT}.tar.gz"
  done
  python -m awscli s3 cp treelite-*.tar.gz "${S3_DEST}" --acl public-read || true
  python -m awscli s3 cp treelite_runtime-*.tar.gz "${S3_DEST}" --acl public-read || true
fi
