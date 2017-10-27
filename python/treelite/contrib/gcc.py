# coding: utf-8
"""
Tools to interact with gcc toolchain
"""

from __future__ import absolute_import as _abs
import os
from .util import _create_shared_base, _libext, _shell

LIBEXT = _libext()

def _create_shared(dirpath, recipe, nthread, options, verbose):
  # Specify command to compile an object file
  recipe['object_ext'] = '.o'
  recipe['library_ext'] = LIBEXT
  recipe['shell'] = _shell()
  # pylint: disable=C0111
  def obj_cmd(source):
    return 'gcc -c -O3 -o {} {} -fPIC -std=c99 -flto -fopenmp {}'\
           .format(source + '.o', source + '.c', ' '.join(options))
  def lib_cmd(sources, target):
    return 'gcc -shared -O3 -o {} {} -std=c99 -flto -fopenmp {}'\
           .format(target + LIBEXT,
                   ' '.join([x[0] + '.o' for x in sources]),
                   ' '.join(options))
  recipe['create_object_cmd'] = obj_cmd
  recipe['create_library_cmd'] = lib_cmd
  recipe['initial_cmd'] = ''
  return _create_shared_base(dirpath, recipe, nthread, verbose)

def _check_ext(dllpath):
  fileext = os.path.splitext(dllpath)[1]
  if fileext != LIBEXT:
    raise ValueError('Library file should have {} extension'.format(LIBEXT))

__all__ = []
