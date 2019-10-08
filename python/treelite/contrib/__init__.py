# coding: utf-8
"""Contrib APIs of treelite python package.

Contrib API provides ways to interact with third-party libraries and tools.
"""

import sys
import os
import json
import time
import shutil
import ctypes
from ..common.util import TreeliteError, lineno, log_info
from ..libpath import find_lib_path
from .util import _libext, _toolchain_exist_check

def expand_windows_path(dirpath):
  """
  Expand a short path to full path (only applicable for Windows)

  Parameters
  ----------
  dirpath : :py:class:`str <python:str>`
      Path to expand

  Returns
  -------
  fullpath : :py:class:`str <python:str>`
      Expanded path
  """
  if sys.platform == 'win32':
    BUFFER_SIZE = 500
    buffer = ctypes.create_unicode_buffer(BUFFER_SIZE)
    get_long_path_name = ctypes.windll.kernel32.GetLongPathNameW
    get_long_path_name.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
    get_long_path_name(dirpath, buffer, BUFFER_SIZE)
    return buffer.value
  return dirpath

def generate_makefile(dirpath, platform, toolchain, options=None):
  """
  Generate a Makefile for a given directory of headers and sources. The
  resulting Makefile will be stored in the directory. This function is useful
  for deploying a model on a different machine.

  Parameters
  ----------
  dirpath : :py:class:`str <python:str>`
      directory containing the header and source files previously generated
      by :py:meth:`Model.compile`. The directory must contain recipe.json
      which specifies build dependencies.
  platform : :py:class:`str <python:str>`
      name of the operating system on which the headers and sources shall be
      compiled. Must be one of the following: 'windows' (Microsoft Windows),
      'osx' (Mac OS X), 'unix' (Linux and other UNIX-like systems)
  toolchain : :py:class:`str <python:str>`
      which toolchain to use. You may choose one of 'msvc', 'clang', and 'gcc'.
      You may also specify a specific variation of clang or gcc (e.g. 'gcc-7')
  options : :py:class:`list <python:list>` of :py:class:`str <python:str>`, \
            optional
      Additional options to pass to toolchain
  """
  if not os.path.isdir(dirpath):
    raise TreeliteError('Directory {} does not exist'.format(dirpath))
  try:
    with open(os.path.join(dirpath, 'recipe.json')) as f:
      recipe = json.load(f)
  except IOError:
    raise TreeliteError('Failed to open recipe.json')

  if 'sources' not in recipe or 'target' not in recipe:
    raise TreeliteError('Malformed recipe.json')
  if options is not None:
    try:
      _ = iter(options)
      options = [str(x) for x in options]
    except TypeError:
      raise TreeliteError('options must be a list of string')
  else:
    options = []

  # Determine file extensions for object and library files
  if platform == 'windows':
    lib_ext = '.dll'
  elif platform == 'osx':
    lib_ext = '.dylib'
  elif platform == 'unix':
    lib_ext = '.so'
  else:
    raise ValueError('Unknown platform: must be one of {windows, osx, unix}')

  _toolchain_exist_check(toolchain)
  if toolchain == 'msvc':
    if platform != 'windows':
      raise ValueError('Visual C++ is compatible only with Windows; ' + \
                       'set platform=\'windows\'')
    from .msvc import _obj_ext, _obj_cmd, _lib_cmd
  else:
    from .gcc import _obj_ext, _obj_cmd, _lib_cmd
  obj_ext = _obj_ext()

  with open(os.path.join(dirpath, 'Makefile'), 'w') as f:
    objects = [x['name'] + obj_ext for x in recipe['sources']] \
              + recipe.get('extra', [])
    f.write('{}: {}\n'.format(recipe['target'] + lib_ext, ' '.join(objects)))
    f.write('\t{}\n'.format(_lib_cmd(objects=objects,
                                     target=recipe['target'],
                                     lib_ext=lib_ext,
                                     toolchain=toolchain,
                                     options=options)))
    for source in recipe['sources']:
      f.write('{}: {}\n'.format(source['name'] + obj_ext,
                                source['name'] + '.c'))
      f.write('\t{}\n'.format(_obj_cmd(source=source['name'],
                                       toolchain=toolchain,
                                       options=options)))

def save_runtime_package(destdir):
  """
  Save a copy of the (zipped) runtime package, containing all glue code
  necessary to deploy compiled models into the wild

  Parameters
  ----------
  destdir : :py:class:`str <python:str>`
      directory to save the zipped package
  """
  pkg_path = find_lib_path(basename='treelite_runtime.zip',
                           libformat=False)
  shutil.copy(pkg_path[0], os.path.join(destdir, 'treelite_runtime.zip'))

def create_shared(toolchain, dirpath, nthread=None, verbose=False, options=None):
  """Create shared library.

  Parameters
  ----------
  toolchain : :py:class:`str <python:str>`
      which toolchain to use. You may choose one of 'msvc', 'clang', and 'gcc'.
      You may also specify a specific variation of clang or gcc (e.g. 'gcc-7')
  dirpath : :py:class:`str <python:str>`
      directory containing the header and source files previously generated
      by :py:meth:`Model.compile`. The directory must contain recipe.json
      which specifies build dependencies.
  nthread : :py:class:`int <python:int>`, optional
      number of threads to use in creating the shared library.
      Defaults to the number of cores in the system.
  verbose : :py:class:`bool <python:bool>`, optional
      whether to produce extra messages
  options : :py:class:`list <python:list>` of :py:class:`str <python:str>`, \
            optional
      Additional options to pass to toolchain

  Returns
  -------
  libpath : :py:class:`str <python:str>`
      absolute path of created shared library

  Example
  -------
  The following command uses Visual C++ toolchain to generate
  ``./my/model/model.dll``:

  .. code-block:: python

     model.compile(dirpath='./my/model', params={}, verbose=True)
     create_shared(toolchain='msvc', dirpath='./my/model', verbose=True)

  Later, the shared library can be referred to by its directory name:

  .. code-block:: python

     predictor = Predictor(libpath='./my/model', verbose=True)
     # looks for ./my/model/model.dll

  Alternatively, one may specify the library down to its file name:

  .. code-block:: python

     predictor = Predictor(libpath='./my/model/model.dll', verbose=True)
  """

  # pylint: disable=R0912

  if nthread is not None and nthread <= 0:
    raise TreeliteError('nthread must be positive integer')
  dirpath = expand_windows_path(dirpath)
  if not os.path.isdir(dirpath):
    raise TreeliteError('Directory {} does not exist'.format(dirpath))
  try:
    with open(os.path.join(dirpath, 'recipe.json')) as f:
      recipe = json.load(f)
  except IOError:
    raise TreeliteError('Failed to open recipe.json')

  if 'sources' not in recipe or 'target' not in recipe:
    raise TreeliteError('Malformed recipe.json')
  if options is not None:
    try:
      _ = iter(options)
      options = [str(x) for x in options]
    except TypeError:
      raise TreeliteError('options must be a list of string')
  else:
    options = []

  # write warning for potentially long compile time
  long_time_warning = False
  for source in recipe['sources']:
    if int(source['length']) > 10000:
      long_time_warning = True
      break
  if long_time_warning:
    log_info(__file__, lineno(),
             '\033[1;31mWARNING: some of the source files are long. ' +\
             'Expect long compilation time.\u001B[0m '+\
             'You may want to adjust the parameter ' +\
             '\x1B[33mparallel_comp\u001B[0m.\n')

  tstart = time.time()
  _toolchain_exist_check(toolchain)
  if toolchain == 'msvc':
    from .msvc import _create_shared
  else:
    from .gcc import _create_shared
  libpath = \
    _create_shared(dirpath, toolchain, recipe, nthread, options, verbose)
  if verbose:
    log_info(__file__, lineno(),
             'Generated shared library in '+\
             '{0:.2f} seconds'.format(time.time() - tstart))
  return libpath

__all__ = ['create_shared', 'save_runtime_package', 'generate_makefile']
