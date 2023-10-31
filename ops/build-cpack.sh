#!/bin/bash

set -euo pipefail

echo "##[section] Building Treelite for amd64..."
tests/ci_build/ci_build.sh ubuntu20 tests/ci_build/build_via_cmake.sh

echo "##[section] Packing CPack for amd64..."
tests/ci_build/ci_build.sh ubuntu20 bash -c "cd build/ && cpack -G TGZ"
for tgz in build/treelite-*-Linux.tar.gz
do
  mv -v "${tgz}" "${tgz%-Linux.tar.gz}+${COMMIT_ID}-Linux-amd64.tar.gz"
done

echo "##[section]Uploading CPack for amd64..."
python -m awscli s3 cp build/*.tar.gz s3://treelite-cpack/ --acl public-read

rm -rf build/

echo "##[section] Building Treelite for aarch64..."
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
tests/ci_build/ci_build.sh ubuntu20_aarch64 tests/ci_build/build_via_cmake.sh

echo "##[section] Packing CPack for aarch64..."
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
tests/ci_build/ci_build.sh ubuntu20_aarch64 bash -c "cd build/ && cpack -G TGZ"
for tgz in build/treelite-*-Linux.tar.gz
do
  mv -v "${tgz}" "${tgz%-Linux.tar.gz}+${COMMIT_ID}-Linux-aarch64.tar.gz"
done

echo "##[section]Uploading CPack for aarch64..."
python -m awscli s3 cp build/*.tar.gz s3://treelite-cpack/ --acl public-read
