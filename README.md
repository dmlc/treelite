# tree-lite

fast tree inference

## =======UNDER CONSTRUCTION: to be released soon=======

## [To-do list](https://github.com/dmlc/tree-lite/issues/1)

## How to install (UNIX-like systems)
```bash
git clone --recursive https://github.com/dmlc/tree-lite.git
cd tree-lite
mkdir build
cd build
cmake ..
make
```

## How to install (Windows with Visual Studio)
On the command-line:
```cmd
git clone --recursive https://github.com/dmlc/tree-lite.git
cd tree-lite
mkdir build
cd build

:: if using Protobuf, specify its location; otherwise remove the following line
set CMAKE_PREFIX_PATH=C:\path\to\protobuf

:: be sure to specify "Win64" at the end
cmake .. -G"Visual Studio 15 2017 Win64"
```
Now the `build` folder should have the solution file `treelite.sln`.

## Optional Protobuf support
If your system has Protobuf
([google/protobuf](https://github.com/google/protobuf)) library installed,
tree-lite will be compiled with Protobuf support. It can be compiled without
Protobuf, but in this case you won't be able to read models from Protobuf
files.
