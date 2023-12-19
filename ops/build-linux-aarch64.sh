#!/bin/bash

set -euo pipefail

TAG=manylinux2014_aarch64

export CIBW_BUILD=cp38-manylinux_aarch64
export CIBW_ARCHS=aarch64
export CIBW_BUILD_VERBOSITY=3
export CIBW_MANYLINUX_AARCH64_IMAGE=manylinux2014

echo "##[section]Building Python wheel (aarch64) for Treelite..."
python -m cibuildwheel python --output-dir wheelhouse
python tests/ci_build/rename_whl.py wheelhouse ${COMMIT_ID} ${TAG}

echo "##[section]Uploading Python wheel (aarch64)..."
python -m awscli s3 cp wheelhouse/*.whl s3://treelite-wheels/ --acl public-read --region us-west-2 || true
