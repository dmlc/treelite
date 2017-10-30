# coding: utf-8
"""
Tools to interact with Microsoft Visual C++ (MSVC) toolchain
"""

from __future__ import absolute_import as _abs
import os
from ..common.compat import PY3
from .util import _create_shared_base, _libext, _shell

LIBEXT = _libext()

def _is_64bit_windows():
  return 'PROGRAMFILES(X86)' in os.environ

def _varsall_bat_path():
  if PY3:
    import winreg                 # pylint: disable=E0401
  else:
    import _winreg as winreg      # pylint: disable=E0401
  if _is_64bit_windows():
    key_name = 'SOFTWARE\\Wow6432Node\\Microsoft\\VisualStudio\\SxS\\VS7'
  else:
    key_name = 'SOFTWARE\\Microsoft\\VisualStudio\\SxS\\VC7'
  key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_name)
  i = 0
  vs_installs = []         # list of all Visual Studio installations
  while True:
    try:
      version, location, _ = winreg.EnumValue(key, i)
      vs_installs.append((version, location))
    except WindowsError:   # pylint: disable=E0602
      break
    i += 1

  # if a custom location is given, try that first
  if 'TREELITE_VCVARSALL' in os.environ:
    candidate = os.environ['TREELITE_VCVARSALL']
    if os.path.basename(candidate).lower() != 'vcvarsall.bat':
      raise OSError('Environment variable TREELITE_VCVARSALL must point to '+\
                    'file vcvarsall.bat')
    if os.path.isfile(candidate):
      return candidate
    else:
      raise OSError('Environment variable TREELITE_VCVARSALL does not refer '+\
                    'to existing vcvarsall.bat')

  # scan all detected Visual Studio installations, with most recent first
  for version, vcroot in sorted(vs_installs, key=lambda x: x[0], reverse=True):
    if version == '15.0':   # Visual Studio 2017 revamped directory structure
      candidate = os.path.join(vcroot, 'VC\\Auxiliary\\Build\\vcvarsall.bat')
    else:
      candidate = os.path.join(vcroot, 'VC\\vcvarsall.bat')
    if os.path.isfile(candidate):
      return candidate
  raise OSError('vcvarsall.bat not found; please specify its full path in '+\
                'the environment variable TREELITE_VCVARSALL')

def _obj_ext():
  return '.obj'

def _obj_cmd(source, options):
  return 'cl.exe /c /openmp /Ox {} {}'\
          .format(source + '.c', ' '.join(options))

# pylint: disable=W0613
def _lib_cmd(sources, target, lib_ext, options):
  obj_ext = _obj_ext()
  return 'cl.exe /LD /Fe{} /openmp {} {}'\
          .format(target,
                  ' '.join([x['name'] + obj_ext for x in sources]),
                  ' '.join(options))

def _create_shared(dirpath, recipe, nthread, options, verbose):
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
  recipe['initial_cmd'] = '\"{}\" {}\n'\
                          .format(_varsall_bat_path(),
                                  'amd64' if _is_64bit_windows() else 'x86')
  return _create_shared_base(dirpath, recipe, nthread, verbose)

def _check_ext(dllpath):
  fileext = os.path.splitext(dllpath)[1]
  if fileext != '.dll':
    raise ValueError('Library file should have .dll extension')

__all__ = []
