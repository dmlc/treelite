# coding: utf-8
"""Contrib APIs of Treelite python package.

Contrib API provides ways to interact with third-party libraries and tools.
"""

import sys
import os
import json
import time
import ctypes
from ..util import TreeliteError, lineno, log_info
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
        buffer_size = 500
        buffer = ctypes.create_unicode_buffer(buffer_size)
        get_long_path_name = ctypes.windll.kernel32.GetLongPathNameW
        get_long_path_name.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        get_long_path_name(dirpath, buffer, buffer_size)
        return buffer.value
    return dirpath


def generate_makefile(dirpath, platform, toolchain, options=None):  # pylint: disable=R0912
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
        with open(os.path.join(dirpath, 'recipe.json'), 'r', encoding='UTF-8') as f:
            recipe = json.load(f)
    except IOError as e:
        raise TreeliteError('Failed to open recipe.json') from e

    if 'sources' not in recipe or 'target' not in recipe:
        raise TreeliteError('Malformed recipe.json')
    if options is not None:
        try:
            _ = iter(options)
            options = [str(x) for x in options]
        except TypeError as e:
            raise TreeliteError('options must be a list of string') from e
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

    with open(os.path.join(dirpath, 'Makefile'), 'w', encoding='UTF-8') as f:
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


def generate_cmakelists(dirpath, options=None):
    """
    Generate a CMakeLists.txt for a given directory of headers and sources. The
    resulting CMakeLists.txt will be stored in the directory. This function is useful
    for deploying a model on a different machine.

    Parameters
    ----------
    dirpath : :py:class:`str <python:str>`
        directory containing the header and source files previously generated
        by :py:meth:`Model.compile`. The directory must contain recipe.json
        which specifies build dependencies.
    options : :py:class:`list <python:list>` of :py:class:`str <python:str>`, \
              optional
        Additional options to pass to toolchain
    """
    if not os.path.isdir(dirpath):
        raise TreeliteError(f'Directory {dirpath} does not exist')
    try:
        with open(os.path.join(dirpath, 'recipe.json'), 'r', encoding='UTF-8') as f:
            recipe = json.load(f)
    except IOError as e:
        raise TreeliteError('Failed to open recipe.json') from e

    if 'sources' not in recipe or 'target' not in recipe:
        raise TreeliteError('Malformed recipe.json')
    if options is not None:
        try:
            _ = iter(options)
            options = [str(x) for x in options]
        except TypeError as e:
            raise TreeliteError('options must be a list of string') from e
    else:
        options = []

    target = recipe['target']
    sources = ' '.join([x['name'] + '.c' for x in recipe['sources']])
    options = ' '.join(options)
    with open(os.path.join(dirpath, 'CMakeLists.txt'), 'w', encoding='UTF-8') as f:
        print('cmake_minimum_required(VERSION 3.13)', file=f)
        print('project(mushroom LANGUAGES C)\n', file=f)
        print(f'add_library({target} SHARED)', file=f)
        print(f'target_sources({target} PRIVATE header.h {sources})', file=f)
        print(f'target_compile_options({target} PRIVATE {options})', file=f)
        print(f'target_include_directories({target} PRIVATE "${{PROJECT_BINARY_DIR}}")', file=f)
        print(f'set_target_properties({target} PROPERTIES', file=f)
        print('''POSITION_INDEPENDENT_CODE ON
            C_STANDARD 99
            C_STANDARD_REQUIRED ON
            PREFIX ""
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_BINARY_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_BINARY_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${PROJECT_BINARY_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${PROJECT_BINARY_DIR}"
            LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
            LIBRARY_OUTPUT_DIRECTORY_DEBUG "${PROJECT_BINARY_DIR}"
            LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PROJECT_BINARY_DIR}"
            LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${PROJECT_BINARY_DIR}"
            LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${PROJECT_BINARY_DIR}")
        ''', file=f)


def create_shared(toolchain, dirpath, *, nthread=None, verbose=False, options=None,
                  long_build_time_warning=True):
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
    long_build_time_warning : :py:class:`bool <python:bool>`, optional
        If set to False, suppress the warning about potentially long build time

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
        with open(os.path.join(dirpath, 'recipe.json'), 'r', encoding='UTF-8') as f:
            recipe = json.load(f)
    except IOError as e:
        raise TreeliteError('Failed to open recipe.json') from e

    if 'sources' not in recipe or 'target' not in recipe:
        raise TreeliteError('Malformed recipe.json')
    if options is not None:
        try:
            _ = iter(options)
            options = [str(x) for x in options]
        except TypeError as e:
            raise TreeliteError('options must be a list of string') from e
    else:
        options = []

    # Write warning for potentially long compile time
    if long_build_time_warning:
        warn = False
        for source in recipe['sources']:
            if int(source['length']) > 10000:
                warn = True
                break
        if warn:
            log_info(__file__, lineno(),
                     '\033[1;31mWARNING: some of the source files are long. ' + \
                     'Expect long build time.\u001B[0m ' + \
                     'You may want to adjust the parameter ' + \
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
                 'Generated shared library in ' + \
                 '{0:.2f} seconds'.format(time.time() - tstart))
    return libpath


__all__ = ['create_shared', 'generate_makefile', 'generate_cmakelists']
