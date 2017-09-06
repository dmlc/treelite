# coding: utf-8
"""Contrib APIs of tree-lite python package.

Contrib API provides ways to interact with third-party libraries and tools.
"""

import os
import json
import time
from ..core import TreeliteError
from .util import lineno, log_info

def _check_ext(toolchain, dllpath):
  if toolchain == 'msvc':
    from .msvc import _check_ext
  elif toolchain == 'gcc':
    from .gcc import _check_ext
  elif toolchain == 'clang':
    from .clang import _check_ext
  else:
    raise ValueError('toolchain {} not supported'.format(toolchain))
  _check_ext(dllpath)

def create_shared(toolchain, dirpath, nthread=None, verbose=False, options=None):
  """Create shared library.

  Usage
  -----
  model.compile(dirpath='./my/model', params={}, verbose=True)
  create_shared(toolchain='msvc', dirpath='./my/model', verbose=True)
  # resulting shared library is ./my/model/model.dll (assuming Windows platform)

  Parameters
  ----------
  toolchain : string
      which toolchain to use (e.g. msvc, clang, gcc)
  dirpath : string
      directory containing the header and source files previously generated
      by model.compile(). The directory must contain recipe.json
      which specifies build dependencies.
  nthread : int, optional
      number of threads to use while compiling source files in parallel.
      Defaults to the number of cores in the system.
  verbose : boolean, optional (defaults to False)
      whether to produce extra messages
  options : str, optional (defaults to None)
      Additional options to pass to toolchain

  Returns
  -------
  absolute path of created shared library
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
  if options is not None:
    try:
      iterator = iter(options)
      options = [str(x) for x in options]
    except TypeError:
      raise TreeliteError('options must be a list of string')
  else:
    options = []

  # write warning for potentially long compile time
  long_time_warning = False
  for source in recipe['sources']:
    if source[1] > 10000:
      long_time_warning = True
  if long_time_warning:
    log_info(__file__, lineno(),
             '\033[1;31mWARNING: some of the source files are long. ' +\
             'Expect long compilation time.\u001B[0m '+\
             'You may want to adjust the parameter ' +\
             '\x1B[33mparallel_comp\u001B[0m.\n')

  tstart = time.time()
  if toolchain == 'msvc':
    from .msvc import _create_shared
  elif toolchain == 'gcc':
    from .gcc import _create_shared
  elif toolchain == 'clang':
    from .clang import _create_shared
  else:
    raise ValueError('toolchain {} not supported'.format(toolchain))
  libpath = _create_shared(dirpath, recipe, nthread, options, verbose)
  if verbose:
    log_info(__file__, lineno(),
             'Generated shared library in '+\
             '{0:.2f} seconds'.format(time.time() - tstart))
  return libpath

__all__ = ['create_shared']
