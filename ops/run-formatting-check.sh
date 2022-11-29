#!/bin/bash

set -euo pipefail

echo "##[section]Installing pylint and cpplint..."
conda update -n base -c conda-forge -q -y conda
conda install -c conda-forge -y mamba>=1.0.0
mamba env create -q -f ops/conda_env/dev.yml
source activate dev

echo "##[section]Running pylint and cpplint..."
make lint
