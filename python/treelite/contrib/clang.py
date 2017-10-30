# coding: utf-8
"""
Tools to interact with clang toolchain
"""

from __future__ import absolute_import as _abs
import os
import subprocess

from ..common.util import TemporaryDirectory
from .util import _create_shared_base, _libext, _shell

LIBEXT = _libext()

def _openmp_supported():
  # make temporary folder
  with TemporaryDirectory() as temp_dir:
    filename = os.path.join(temp_dir, 'test.c')
    with open(filename, 'w') as f:
      f.write('int main() { return 0; }\n')
    retcode = subprocess.call('clang {} -fopenmp'.format(filename), shell=True,
                              stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
  return retcode == 0

def _obj_ext():
  return '.o'

def _obj_cmd(source, options):
  obj_ext = _obj_ext()
  return 'clang -c -O3 -o {} {} -fPIC -std=c99 -flto {}'\
          .format(source + obj_ext, source + '.c', ' '.join(options))

def _lib_cmd(sources, target, lib_ext, options):
  obj_ext = _obj_ext()
  return 'clang -shared -O3 -o {} {} -std=c99 -flto {}'\
          .format(target + lib_ext,
                  ' '.join([x['name'] + obj_ext for x in sources]),
                  ' '.join(options))

def _create_shared(dirpath, recipe, nthread, options, verbose):
  if _openmp_supported():  # clang may not support OpenMP, so make it optional
    options += ['-fopenmp']

  # Specify command to compile an object file
  recipe['object_ext'] = _obj_ext()
  recipe['library_ext'] = LIBEXT
  recipe['shell'] = _shell()
  # pylint: disable=C0111
  def obj_cmd(source):
    return _obj_cmd(source, options)
  def lib_cmd(sources, target):
    return _lib_cmd(sources, target, LIBEXT, options)
  recipe['create_object_cmd'] = obj_cmd
  recipe['create_library_cmd'] = lib_cmd
  recipe['initial_cmd'] = ''
  return _create_shared_base(dirpath, recipe, nthread, verbose)

def _check_ext(dllpath):
  fileext = os.path.splitext(dllpath)[1]
  if fileext != LIBEXT:
    raise ValueError('Library file should have {} extension'.format(LIBEXT))

__all__ = []
