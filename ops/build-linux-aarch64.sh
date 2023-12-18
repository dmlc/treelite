#!/bin/bash

set -euo pipefail

TAG=manylinux2014_aarch64

echo "##[section]Building Python wheel (aarch64) for Treelite..."
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
tests/ci_build/ci_build.sh centos7_amd64 bash -c "cd python/ && pip wheel --no-deps -v . --wheel-dir dist/"
tests/ci_build/ci_build.sh centos7_amd64 auditwheel repair --only-plat --plat ${TAG} python/dist/*.whl
rm -v python/dist/*.whl
mv -v wheelhouse/*.whl python/dist/
python tests/ci_build/rename_whl.py python/dist ${COMMIT_ID} ${TAG}

echo "##[section]Uploading Python wheel (aarch64)..."
python -m awscli s3 cp python/dist/*.whl s3://treelite-wheels/ --acl public-read --region us-west-2
