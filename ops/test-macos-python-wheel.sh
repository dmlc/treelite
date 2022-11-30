#!/bin/bash

set -euo pipefail

echo "##[section]Installing Treelite into Python environment..."
pip install wheelhouse/*.whl

echo "##[section]Running Python tests..."
python -m pytest -v -rxXs --fulltrace tests/python/test_basic.py
