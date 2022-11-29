#!/bin/bash

set -euo pipefail

echo "##[section]Installing pylint and cpplint..."
${BASH_SOURCE%/*}/update-conda.sh
conda install -c conda-forge -y mamba>=1.0.0
mamba env create -q -f ops/conda_env/dev.yml
source activate dev

echo "##[section]Running pylint and cpplint..."
make lint
