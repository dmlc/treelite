#!/bin/bash

set -euo pipefail

TAG=manylinux2014_x86_64

echo "##[section]Building Treelite..."
tests/ci_build/ci_build.sh centos7 tests/ci_build/build_via_cmake.sh

echo "##[section]Packaging Python wheel for Treelite..."
tests/ci_build/ci_build.sh centos7 bash -c "cd python/ && pip wheel --no-deps -v . --wheel-dir dist/"
tests/ci_build/ci_build.sh auditwheel_x86_64 auditwheel repair --only-plat --plat ${TAG} python/dist/*.whl
rm -v python/dist/*.whl
mv -v wheelhouse/*.whl python/dist/
tests/ci_build/ci_build.sh centos7 python tests/ci_build/rename_whl.py python/dist ${COMMIT_ID} ${TAG}
