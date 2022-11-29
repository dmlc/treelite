#!/bin/bash

set -euo pipefail

echo "##[section]Building MacOS Python wheels..."
tests/ci_build/build_macos_python_wheels.sh ${CIBW_PLATFORM_ID} ${COMMIT_ID}

echo "##[section]Uploading MacOS Python wheels to S3..."
python -m awscli s3 cp wheelhouse/treelite-*.whl s3://treelite-wheels/ --acl public-read || true
python -m awscli s3 cp wheelhouse/treelite_runtime-*.whl s3://treelite-wheels/ --acl public-read || true
