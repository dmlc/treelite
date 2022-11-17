#!/bin/bash

set -euo pipefail

echo "##[section]Setting up Python environment..."
conda create -n python3 -y -q -c conda-forge python=3.9 numpy scipy pandas pytest scikit-learn coverage \
  cmake ninja awscli
source activate python3
python --version
python -m pip install xgboost lightgbm

echo "##[section]Building a source distribution..."
make pippack

echo "##[section]Testing the source distribution..."
python -m pip install -v treelite-*.tar.gz
python -m pip install -v treelite_runtime-*.tar.gz
python -m pytest -v -rxXs --fulltrace tests/python/test_basic.py

# Deploy source distribution to S3
for file in ./treelite-*.tar.gz ./treelite_runtime-*.tar.gz
do
  mv "${file}" "${file%.tar.gz}+${COMMIT_ID}.tar.gz"
done
python -m awscli s3 cp treelite-*.tar.gz s3://treelite-wheels/ --acl public-read || true
python -m awscli s3 cp treelite_runtime-*.tar.gz  s3://treelite-wheels/ --acl public-read || true
