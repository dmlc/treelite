#!/bin/bash

set -euo pipefail

echo "##[section]Installing pylint and cpplint..."
conda create -n dev -y -q -c conda-forge python=3.9 numpy scipy pandas scikit-learn pytest
source activate dev
pip install cpplint pylint xgboost lightgbm

echo "##[section]Running pylint and cpplint..."
make lint
