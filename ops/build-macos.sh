#!/bin/bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 [platform_id] [commit ID]"
  exit 1
fi

platform_id=$1
commit_id=$2

echo "##[section]Building MacOS Python wheels..."
tests/ci_build/build_macos_python_wheels.sh ${platform_id} ${commit_id}

echo "##[section]Uploading MacOS Python wheels to S3..."
python -m awscli s3 cp wheelhouse/treelite-*.whl s3://treelite-wheels/ --acl public-read --region us-west-2 || true
