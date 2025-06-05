#!/bin/bash

set -euo pipefail

if [[ -z "${COMMIT_ID:-}" ]]
then
  echo "Make sure to set environment variable COMMIT_ID"
  exit 1
fi

if [[ "$#" -lt 1 ]]
then
  echo "Usage: $0 {amd64,aarch64}"
  exit 2
fi

arch="$1"

echo "##[section] Building Treelite for ${arch}..."
tests/ci_build/ci_build.sh ubuntu20_${arch} tests/ci_build/build_via_cmake.sh

echo "##[section] Packing CPack for ${arch}..."
tests/ci_build/ci_build.sh ubuntu20_${arch} bash -c "cd build/ && cpack -G TGZ"
for tgz in build/treelite-*-Linux.tar.gz
do
  mv -v "${tgz}" "${tgz%-Linux.tar.gz}+${COMMIT_ID}-Linux-${arch}.tar.gz"
done

echo "##[section]Uploading CPack for ${arch}..."
python -m awscli s3 cp build/*.tar.gz s3://treelite-cpack/ --acl public-read --region us-west-2 || true
