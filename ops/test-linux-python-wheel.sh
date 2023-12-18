#!/bin/bash

set -euo pipefail

echo "##[section]Installing Treelite into Python environment..."
pip install python/dist/*.whl

echo "##[section]Running Python tests..."
python -m pytest -v -rxXs --fulltrace --durations=0 tests/python/test_sklearn_integration.py
