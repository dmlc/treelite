#!/bin/bash

set -euo pipefail

echo "##[section]Setting up Python environment..."
conda create -n python3 -y -q -c conda-forge python=3.9
source activate python3
pip install cibuildwheel
source deactivate

echo "##[section]Building MacOS Python wheels..."
tests/ci_build/build_python_wheels.sh ${CIBW_PLATFORM_ID} ${COMMIT_ID}

echo "##[section]Uploading MacOS Python wheels to S3..."
source activate python3
python --version
python -m pip install awscli
python -m awscli s3 cp wheelhouse/treelite-*.whl s3://treelite-wheels/ --acl public-read || true
python -m awscli s3 cp wheelhouse/treelite_runtime-*.whl s3://treelite-wheels/ --acl public-read || true
