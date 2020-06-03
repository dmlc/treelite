#!/bin/sh

set -e
set -x

if which "cmake"
then
  echo "Found CMake"
else
  echo "Please install CMake first"
  exit 1
fi

if which "ninja"
then
  echo "Found Ninja"
  GENERATOR="Ninja"
  BUILD_CMD="ninja"
else
  echo "Did not find Ninja. Using GNU Make."
  if which "make"
  then
    echo "Found GNU Make"
    GENERATOR="Unix Makefiles"
    if which nproc
    then
      BUILD_CMD="make -j$(nproc)"
    else
      BUILD_CMD="make"
    fi
  else
    echo "Please install GNU Make first"
    exit 2
  fi
fi

oldpath=`pwd`
cd ./treelite/

mkdir -p build
cd build
cmake .. -G"$GENERATOR" -DENABLE_PROTOBUF=ON
${BUILD_CMD}

cd $oldpath
set +x
