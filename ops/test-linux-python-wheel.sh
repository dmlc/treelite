#!/bin/bash

set -euo pipefail

echo "##[section]Setting up Python environment..."
./update-conda.sh
conda install -c conda-forge -y mamba>=1.0.0
mamba env create -q -f ops/conda_env/dev.yml
source activate dev

echo "##[section]Installing Treelite into Python environment..."
pip install main/*.whl runtime/*.whl

echo "##[section]Running Python tests..."
python -m pytest -v -rxXs --fulltrace tests/python/test_basic.py

echo "##[section]Uploading Python wheels..."
python -m awscli s3 cp main/*.whl s3://treelite-wheels/ --acl public-read || true
python -m awscli s3 cp runtime/*.whl s3://treelite-wheels/ --acl public-read || true
