#!/bin/bash

set -euo pipefail

echo "##[section]Installing Treelite into Python environment..."
pip install main/*.whl runtime/*.whl

echo "##[section]Running Python tests..."
python -m pytest -v -rxXs --fulltrace tests/python/test_basic.py

echo "##[section]Uploading Python wheels..."
python -m awscli s3 cp main/*.whl s3://treelite-wheels/ --acl public-read || true
python -m awscli s3 cp runtime/*.whl s3://treelite-wheels/ --acl public-read || true
