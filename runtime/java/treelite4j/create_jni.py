#!/usr/bin/env python
# coding: utf-8
import errno
import glob
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager

# Monkey-patch the API inconsistency between Python2.X and 3.X.
if sys.platform.startswith("linux"):
  sys.platform = "linux"

@contextmanager
def cd(path):
  path = normpath(path)
  cwd = os.getcwd()
  os.chdir(path)
  print("cd " + path)
  try:
    yield path
  finally:
    os.chdir(cwd)


def maybe_makedirs(path):
  path = normpath(path)
  print("mkdir -p " + path)
  try:
    os.makedirs(path)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise


def run(command, **kwargs):
  print(command)
  subprocess.check_call(command, shell=True, **kwargs)


def cp(source, target):
  source = normpath(source)
  target = normpath(target)
  print("cp {0} {1}".format(source, target))
  shutil.copy(source, target)


def normpath(path):
  """Normalize UNIX path to a native path."""
  normalized = os.path.join(*path.split("/"))
  if os.path.isabs(path):
    return os.path.abspath("/") + normalized
  else:
    return normalized


if __name__ == "__main__":
  if sys.platform == "darwin":
    # Enable of your compiler supports OpenMP.
    os.environ["JAVA_HOME"] = subprocess.check_output(
        "/usr/libexec/java_home").strip().decode()

  print("building Java wrapper")
  with cd(".."):
    maybe_makedirs("build")
    with cd("build"):
      if sys.platform == "win32":
        # Force x64 build on Windows.
        maybe_generator = ' -G"Visual Studio 14 Win64"'
      else:
        maybe_generator = ""
      if 'cpp-coverage' in sys.argv:
        maybe_generator += ' -DTEST_COVERAGE=ON'
      run("cmake .. " + maybe_generator)
      run("cmake --build . --config Release")

  print("copying native library")
  library_name = {
    "win32": "treelite4j.dll",
    "darwin": "libtreelite4j.dylib",
    "linux": "libtreelite4j.so"
  }[sys.platform]
  maybe_makedirs("src/main/resources/lib")
  cp("../build/lib/" + library_name, "src/main/resources/lib")

  print("building mushroom example")
  with cd("src/test/resources/mushroom_example"):
    run("cmake . " + maybe_generator)
    run("cmake --build . --config Release")
