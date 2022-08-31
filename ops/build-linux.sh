#!/bin/bash

set -euo pipefail

TAG=manylinux2014_x86_64

echo "Building Treelite..."
tests/ci_build/ci_build.sh cpu tests/ci_build/build_via_cmake.sh

echo "Packaging Python wheel for Treelite..."
tests/ci_build/ci_build.sh cpu bash -c "cd python/ && python setup.py bdist_wheel --universal"
tests/ci_build/ci_build.sh auditwheel_x86_64 auditwheel repair --only-plat --plat ${TAG} python/dist/*.whl
rm -v python/dist/*.whl
mv -v wheelhouse/*.whl python/dist/
tests/ci_build/ci_build.sh cpu python tests/ci_build/rename_whl.py python/dist/*.whl ${COMMIT_ID} ${TAG}

echo "Packaging Python wheel for Treelite runtime..."
tests/ci_build/ci_build.sh cpu bash -c "cd runtime/python/ && python setup.py bdist_wheel --universal"
tests/ci_build/ci_build.sh auditwheel_x86_64 auditwheel repair --only-plat --plat ${TAG} runtime/python/dist/*.whl
rm -v runtime/python/dist/*.whl
mv -v wheelhouse/*.whl runtime/python/dist/
tests/ci_build/ci_build.sh cpu python tests/ci_build/rename_whl.py runtime/python/dist/*.whl ${COMMIT_ID} ${TAG}
