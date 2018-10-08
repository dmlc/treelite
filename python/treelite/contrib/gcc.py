# coding: utf-8
"""
Tools to interact with toolchains GCC, Clang, and other UNIX compilers
"""

from __future__ import absolute_import as _abs
import os
import subprocess

from ..common.compat import DEVNULL
from ..common.util import TemporaryDirectory
from .util import _create_shared_base, _libext

LIBEXT = _libext()

def _obj_ext():
  return '.o'

def _obj_cmd(source, toolchain, options):
  obj_ext = _obj_ext()
  return '{} -c -O3 -o {} {} -fPIC -std=c99 {}'\
         .format(toolchain, source + obj_ext, source + '.c', ' '.join(options))

def _lib_cmd(sources, target, lib_ext, toolchain, options):
  obj_ext = _obj_ext()
  return '{} -shared -O3 -o {} {} -std=c99 {}'\
          .format(toolchain,
                  target + lib_ext,
                  ' '.join([x['name'] + obj_ext for x in sources]),
                  ' '.join(options))

def _create_shared(dirpath, toolchain, recipe, nthread, options, verbose):
  options += ['-lm']
  # Specify command to compile an object file
  recipe['object_ext'] = _obj_ext()
  recipe['library_ext'] = LIBEXT
  # pylint: disable=C0111
  def obj_cmd(source):
    return _obj_cmd(source, toolchain, options)
  def lib_cmd(sources, target):
    return _lib_cmd(sources, target, LIBEXT, toolchain, options)
  recipe['create_object_cmd'] = obj_cmd
  recipe['create_library_cmd'] = lib_cmd
  recipe['initial_cmd'] = ''
  return _create_shared_base(dirpath, recipe, nthread, verbose)

__all__ = []
