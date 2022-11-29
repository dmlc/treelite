#!/bin/bash

set -euo pipefail

echo "##[section]Setting up Python environment..."
conda install -c conda-forge -y mamba
mamba env create -q -f ops/conda_env/dev.yml
source activate dev

echo "##[section]Installing Treelite into Python environment..."
pip install wheelhouse/*.whl

echo "##[section]Running Python tests..."
python -m pytest -v -rxXs --fulltrace tests/python/test_basic.py
