#!/bin/bash

set -euo pipefail

echo "##[section]Building MacOS Python wheels..."
tests/ci_build/build_python_wheels.sh ${CIBW_PLATFORM_ID} ${COMMIT_ID}

echo "##[section]Uploading MacOS Python wheels to S3..."
source activate python3
python --version
python -m pip install awscli
python -m awscli s3 cp treelite-*.whl s3://treelite-wheels/ --acl public-read || true
python -m awscli s3 cp treelite_runtime-*.whl s3://treelite-wheels/ --acl public-read || true
