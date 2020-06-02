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
  cmake .. -DENABLE_PROTOBUF=ON -DUSE_OPENMP=OFF
  make -j$(nproc)
  cd ..
  rm -rfv python/dist python/build
  cd python/
  python setup.py bdist_wheel --universal
  cd ..

  # Install Treelite into Python env
  ls -l ./python/dist/*.whl
  python -m pip install ./python/dist/*.whl

  # Run tests
  python -m pip install numpy scipy pandas pytest pytest-cov scikit-learn lightgbm coverage
  export GCC_PATH=gcc-7
  python -m pytest -v --fulltrace tests/python

  # Deploy binary wheel to S3
  python -m pip install awscli
  if [ "${TRAVIS_BRANCH}" == "master" ]
  then
    S3_DEST="s3://treelite-wheels/"
  elif [ -z "${TRAVIS_TAG}" ]
  then
    S3_DEST="s3://treelite-wheels/${TRAVIS_BRANCH}/"
  fi
  python -m awscli s3 cp python/dist/*.whl "${S3_DEST}" --acl public-read || true
fi
