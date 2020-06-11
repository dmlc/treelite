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
  cmake .. -DENABLE_PROTOBUF=ON -DUSE_OPENMP=ON
  make -j$(nproc)
  cd ..
  conda install -c conda-forge numpy scipy pandas pytest pytest-cov scikit-learn coverage
  python -m pip install xgboost lightgbm codecov
  python -m pytest --cov=treelite --cov=treelite_runtime -v --fulltrace tests/python
  codecov
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
  cmake .. -DENABLE_PROTOBUF=ON -DUSE_OPENMP=OFF
  make -j$(nproc)
  cd ..
  rm -rfv python/dist python/build
  cd python/
  python setup.py bdist_wheel --universal
  TAG=macosx_10_13_x86_64.macosx_10_14_x86_64.macosx_10_15_x86_64
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
  python -m pip install xgboost lightgbm
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
  python -m awscli s3 cp python/dist/treelite-*.whl "${S3_DEST}" --acl public-read || true
  python -m awscli s3 cp runtime/python/dist/treelite_runtime*.whl "${S3_DEST}" --acl public-read || true
fi

if [ ${TASK} == "python_sdist_test" ]; then
  conda activate python3
  python --version
  conda install numpy scipy
  if [ ${USE_SYSTEM_PROTOBUF} == "yes" ]; then
    conda install protobuf
  fi

  # Build source distribution
  make pippack

  # Install Treelite into Python env
  python -m pip install -v treelite-*.tar.gz
  python -m pip install -v treelite_runtime-*.tar.gz

  # Run tests
  conda install -c conda-forge numpy scipy pandas pytest scikit-learn coverage
  python -m pip install xgboost lightgbm
  export GCC_PATH=gcc-7
  python -m pytest -v --fulltrace tests/python

  # Deploy source wheel to S3
  if [ ${USE_SYSTEM_PROTOBUF} == "no" ]; then
    python -m pip install awscli
    if [ "${TRAVIS_BRANCH}" == "master" ]
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
fi
