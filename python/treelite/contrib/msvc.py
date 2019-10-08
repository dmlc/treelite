# coding: utf-8
"""
Tools to interact with Microsoft Visual C++ (MSVC) toolchain
"""

from __future__ import absolute_import as _abs
import os
import glob
import re
from distutils.version import StrictVersion
from .util import _create_shared_base, _libext

LIBEXT = _libext()

def _is_64bit_windows():
  return 'PROGRAMFILES(X86)' in os.environ

def _varsall_bat_path():
  # if a custom location is given, try that first
  if 'TREELITE_VCVARSALL' in os.environ:
    candidate = os.environ['TREELITE_VCVARSALL']
    if os.path.basename(candidate).lower() != 'vcvarsall.bat':
      raise OSError('Environment variable TREELITE_VCVARSALL must point to '+\
                    'file vcvarsall.bat')
    if os.path.isfile(candidate):
      return candidate
    raise OSError('Environment variable TREELITE_VCVARSALL does not refer '+\
                  'to existing vcvarsall.bat')

  ## Bunch of heuristics to locate vcvarsall.bat
  candidate_paths = []     # List of possible paths to vcvarsall.bat
  try:
    import winreg                 # pylint: disable=E0401
    if _is_64bit_windows():
      key_name = 'SOFTWARE\\Wow6432Node\\Microsoft\\VisualStudio\\SxS\\VS7'
    else:
      key_name = 'SOFTWARE\\Microsoft\\VisualStudio\\SxS\\VC7'
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_name)
    i = 0
    while True:
      try:
        version, vcroot, _ = winreg.EnumValue(key, i)
        if StrictVersion(version) >= StrictVersion('15.0'):
          # Visual Studio 2017 revamped directory structure
          candidate_paths.append(os.path.join(vcroot, 'VC\\Auxiliary\\Build\\vcvarsall.bat'))
        else:
          candidate_paths.append(os.path.join(vcroot, 'VC\\vcvarsall.bat'))
      except WindowsError:   # pylint: disable=E0602
        break
      i += 1
  except FileNotFoundError:
    pass   # No registry key found
  except ImportError:
    pass   # No winreg module

  for candidate in candidate_paths:
    if os.path.isfile(candidate):
      return candidate

  # If registry method fails, try a bunch of pre-defined paths

  # Visual Studio 2017 and higher
  for vcroot in glob.glob('C:\\Program Files (x86)\\Microsoft Visual Studio\\*') + \
                glob.glob('C:\\Program Files\\Microsoft Visual Studio\\*'):
    if re.fullmatch(r'[0-9]+', os.path.basename(vcroot)):
      for candidate in glob.glob(vcroot + '\\*\\VC\\Auxiliary\\Build\\vcvarsall.bat'):
        if os.path.isfile(candidate):
          return candidate
  # Previous versions of Visual Studio
  pattern = '\\Microsoft Visual Studio*\\VC\\vcvarsall.bat'
  for candidate in glob.glob('C:\\Program Files (x86)' + pattern) + \
                   glob.glob('C:\\Program Files' + pattern):
    if os.path.isfile(candidate):
      return candidate

  raise OSError('vcvarsall.bat not found; please specify its full path in '+\
                'the environment variable TREELITE_VCVARSALL')

def _obj_ext():
  return '.obj'

# pylint: disable=W0613
def _obj_cmd(source, toolchain, options):
  return 'cl.exe /c /openmp /Ox {} {}'\
          .format(source + '.c', ' '.join(options))

# pylint: disable=W0613
def _lib_cmd(objects, target, lib_ext, toolchain, options):
  return 'cl.exe /LD /Fe{} /openmp {} {}'\
          .format(target, ' '.join(objects), ' '.join(options))

# pylint: disable=R0913
def _create_shared(dirpath, toolchain, recipe, nthread, options, verbose):
  # Specify command to compile an object file
  recipe['object_ext'] = _obj_ext()
  recipe['library_ext'] = LIBEXT
  # pylint: disable=C0111
  def obj_cmd(source):
    return _obj_cmd(source, toolchain, options)
  def lib_cmd(objects, target):
    return _lib_cmd(objects, target, LIBEXT, toolchain, options)
  recipe['create_object_cmd'] = obj_cmd
  recipe['create_library_cmd'] = lib_cmd
  recipe['initial_cmd'] = '\"{}\" {}\n'\
                          .format(_varsall_bat_path(),
                                  'amd64' if _is_64bit_windows() else 'x86')
  return _create_shared_base(dirpath, recipe, nthread, verbose)

__all__ = []
