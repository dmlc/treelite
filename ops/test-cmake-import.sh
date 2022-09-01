#!/bin/bash

set -euo pipefail

echo "##[section]Setting up Python environment..."
conda create -n python3 -y -q -c conda-forge python=3.9 ninja cmake rapidjson fmt llvm-openmp
source activate python3
conda --version
python --version

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
