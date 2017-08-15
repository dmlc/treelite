# tree-lite
fast tree inference

## How to install
```bash
git clone --recursive https://github.com/dmlc/tree-lite.git
cd tree-lite
mkdir build
cd build
cmake ..
make
```
If your system has Protocol Buffers library, tree-lite will be able to read
models from Protocol Buffers format. If not, tree-lite will throw an error when
presented with a Protocol Buffers model file.
