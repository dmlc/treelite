#!/bin/bash

set -euo pipefail

echo "##[section]Installing pylint and cpplint..."
conda env create -q -f ops/conda_env/dev.yml
source activate dev

echo "##[section]Running pylint and cpplint..."
make lint
