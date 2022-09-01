#!/bin/bash

set -euo pipefail

echo "##[section]Setting up Python environment..."
python -m pip install --upgrade pip numpy scipy pandas pytest scikit-learn xgboost lightgbm awscli

echo "##[section]Installing Treelite into Python environment..."
python -m pip install main/*.whl runtime/*.whl

echo "##[section]Uploading Python wheels..."
python -m awscli s3 cp main/*.whl s3://treelite-wheels/ --acl public-read || true
python -m awscli s3 cp runtime/*.whl s3://treelite-wheels/ --acl public-read || true
