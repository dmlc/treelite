#!/bin/bash

set -e
set -x

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 [platform_id] [commit ID]"
  exit 1
fi

platform_id=$1
shift
commit_id=$1
shift

if [[ "$platform_id" == macosx_* ]]; then
    if [[ "$platform_id" == macosx_arm64 ]]; then
        # MacOS, Apple Silicon
        wheel_tag=macosx_12_0_arm64
        cpython_ver=312
        cibw_archs=arm64
        export MACOSX_DEPLOYMENT_TARGET=12.0
    elif [[ "$platform_id" == macosx_x86_64 ]]; then
        # MacOS, Intel
        wheel_tag=macosx_10_15_x86_64.macosx_11_0_x86_64.macosx_12_0_x86_64
        cpython_ver=312
        cibw_archs=x86_64
        export MACOSX_DEPLOYMENT_TARGET=10.15
    else
        echo "Platform not supported: $platform_id"
        exit 3
    fi
    # Set up environment variables to configure cibuildwheel
    export CIBW_BUILD=cp${cpython_ver}-${platform_id}
    export CIBW_ARCHS=${cibw_archs}
    export CIBW_TEST_SKIP='*-macosx_arm64'
    export CIBW_BUILD_VERBOSITY=3
else
    echo "Platform not supported: $platform_id"
    exit 2
fi

# Tell delocate-wheel to not vendor libomp.dylib into the wheel
export CIBW_REPAIR_WHEEL_COMMAND_MACOS="delocate-wheel --require-archs {delocate_archs} -w {dest_dir} -v {wheel} --exclude libomp.dylib"

python -m cibuildwheel python --output-dir wheelhouse
python tests/ci_build/rename_whl.py wheelhouse ${commit_id} ${wheel_tag}
