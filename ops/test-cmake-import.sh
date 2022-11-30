#!/bin/bash

set -euo pipefail

# Install Treelite C++ library into the Conda env
set -x
rm -rf build/
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -DCMAKE_INSTALL_LIBDIR="lib" -GNinja
ninja install

# Try compiling a sample application
cd ../tests/example_app/
rm -rf build/
mkdir build
cd build
cmake .. -GNinja
ninja
./cpp_example
./c_example
