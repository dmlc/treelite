#!/bin/bash

set -euo pipefail

TAG=manylinux2014_x86_64

export CIBW_BUILD=cp38-manylinux_x86_64
export CIBW_ARCHS=x86_64
export CIBW_BUILD_VERBOSITY=3
export CIBW_MANYLINUX_X86_64_IMAGE=manylinux2014

echo "##[section]Building Python wheel (amd64) for Treelite..."
python -m cibuildwheel python --output-dir wheelhouse
mv -v wheelhouse/*.whl python/dist/
python tests/ci_build/rename_whl.py python/dist ${COMMIT_ID} ${TAG}

echo "##[section]Uploading Python wheel (amd64)..."
python -m awscli s3 cp python/dist/*.whl s3://treelite-wheels/ --acl public-read --region us-west-2 || true
