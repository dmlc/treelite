#!/bin/bash

set -euo pipefail

echo "Installing pylint and cpplint..."
python -m pip install --upgrade pip cpplint pylint numpy scipy pandas scikit-learn pytest xgboost \
  lightgbm

echo "Running pylint and cpplint..."
make lint
