#!/bin/bash

set -euo pipefail

echo "##[section]Testing the wheel inside a minimal container..."
PYTHONBIN=/opt/python/cp38-cp38/bin/python
docker run --rm -it --pull=always -v $PWD/wheelhouse:/workspace -w /workspace \
    quay.io/pypa/manylinux2014_x86_64:latest \
    bash -c "${PYTHONBIN} -m pip install /workspace/*.whl && ${PYTHONBIN} -c 'import treelite'"

echo "##[section]Installing Treelite into Python environment..."
pip install wheelhouse/*.whl

echo "##[section]Running Python tests..."
python -m pytest -v -rxXs --fulltrace --durations=0 tests/python/test_sklearn_integration.py
