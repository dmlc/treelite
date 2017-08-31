# coding: utf-8
"""Contrib APIs of tree-lite python package.

Contrib API provides ways to interact with third-party libraries and tools.
"""

import os
import json
from ..core import TreeliteError

def create_shared(compiler, dirpath, nthread=None, options=None):
  """Create shared library.

  Parameters
  ----------
  compiler : string
      which compiler to use
  dirpath : string
      directory containing the header and source files previously generated
      by compiler.Compiler.compile(). The directory must contain recipe.json
      which specifies build dependencies.

  nthread : int, optional
      number of threads to use while compiling source files in parallel.
      Defaults to the number of cores in the system.

  options : str, optional (default: None)
      Additional options
  """

  if nthread is not None and nthread <= 0:
    raise TreeliteError('nthread must be positive integer')
  if not os.path.isdir(dirpath):
    raise TreeliteError('Directory {} does not exist'.format(dirpath))
  try:
    with open(os.path.join(dirpath, 'recipe.json')) as f:
      recipe = json.load(f)
  except IOError:
    raise TreeliteError('Fail to open recipe.json')

  if 'sources' not in recipe or 'target' not in recipe:
    raise TreeliteError('Malformed recipe.json')

  if compiler == 'msvc':
    from .msvc import _create_shared
    _create_shared(dirpath, recipe, nthread, options)
  else:
    raise NotImplementedError('compiler {} not implemented yet'.format(compiler))

__all__ = ['create_shared']
