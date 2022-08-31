#!/bin/bash

set -euo pipefail

echo "##[section]Installing pylint and cpplint..."
python -m pip install --upgrade pip cpplint pylint numpy scipy pandas scikit-learn pytest xgboost \
  lightgbm

echo "##[section]Running pylint and cpplint..."
make lint
