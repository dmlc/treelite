"""Utility functions for tests"""
import numpy as np
from sys import platform as _platform
from treelite.contrib import _libext

def load_txt(filename):
  """Get 1D array from text file"""
  content = []
  with open(filename, 'r') as f:
    for line in f:
      content.append(float(line))
  return np.array(content)

def os_compatible_toolchains():
  if _platform == 'darwin':
    toolchains = ['gcc-7', 'clang']
  elif _platform == 'win32':
    toolchains = ['msvc']
  else:
    toolchains = ['gcc', 'clang']
  return toolchains

def os_platform():
  if _platform == 'darwin':
    return 'osx'
  elif _platform == 'win32' or _platform == 'cygwin':
    return 'windows'
  else:
    return 'unix'

def libname(fmt):
  return fmt.format(_libext())
