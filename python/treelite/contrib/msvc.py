# coding: utf-8
"""
Tools to interact with Microsoft Visual C++ (MSVC) toolchain
"""

from __future__ import absolute_import as _abs
import os
import subprocess
from multiprocessing import cpu_count
from ..core import TreeliteError
from ..compat import _str_decode, _str_encode, PY3
from .util import lineno, log_info

if PY3:
  import winreg                 # pylint: disable=E0401
else:
  import _winreg as winreg      # pylint: disable=E0401

def _is_64bit_windows():
  return 'PROGRAMFILES(X86)' in os.environ

def _varsall_bat_path():
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

def _enqueue(args):
  queue = args[0]
  tid = args[1]
  dirpath = args[2]
  options = args[3]
  proc = subprocess.Popen('cmd.exe', shell=True, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
  proc.stdin.write(_str_encode('\"{}\" amd64\n'.format(_varsall_bat_path())))
  proc.stdin.write(_str_encode('cd {}\n'.format(dirpath)))
  proc.stdin.write(_str_encode('type NUL > retcode_cpu{}.txt\n'.format(tid)))
  for source in queue:
    proc.stdin.write(_str_encode('cl.exe /c /openmp /Ox {} {}\n'
                                 .format(source + '.c', ' '.join(options))))
    proc.stdin.write(_str_encode('echo %errorlevel% >> retcode_cpu{}.txt\n'
                                 .format(tid)))
  proc.stdin.flush()

  return proc

def _wait(proc, args):
  tid = args[1]
  dirpath = args[2]
  stdout, _ = proc.communicate()
  with open(os.path.join(dirpath, 'retcode_cpu{}.txt'.format(tid)), 'r') as f:
    retcode = [int(line) for line in f]
  return {'stdout':_str_decode(stdout), 'retcode':retcode}

def _create_dll(dirpath, target, sources, options):
  proc = subprocess.Popen('cmd.exe', shell=True, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
  if _is_64bit_windows():
    proc.stdin.write(_str_encode('\"{}\" x64\n'.format(_varsall_bat_path())))
  else:
    proc.stdin.write(_str_encode('\"{}\" x86\n'.format(_varsall_bat_path())))
  proc.stdin.write(_str_encode('cd {}\n'.format(dirpath)))
  proc.stdin.write(_str_encode(
      'cl.exe /LD /Fe{} /openmp {} {}\n'
      .format(target,
              ' '.join([x[0] + '.obj' for x in sources]),
              ' '.join(options))))
  proc.stdin.write(_str_encode('echo %errorlevel% > retcode_dll.txt\n'))
  proc.stdin.flush()
  stdout, _ = proc.communicate()
  with open(os.path.join(dirpath, 'retcode_dll.txt'), 'r') as f:
    retcode = int(f.readline())
  return {'stdout':_str_decode(stdout), 'retcode':retcode}

def _create_shared(dirpath, recipe, nthread, options, verbose):
  # 1. Compile sources in parallel
  if verbose:
    log_info(__file__, lineno(),
             'Compiling sources files in directory {} '.format(dirpath) +\
             'into object files (*.obj)...')
  ncore = cpu_count()
  ncpu = min(ncore, nthread) if nthread is not None else ncore
  workqueues = [([], tid, os.path.abspath(dirpath), options) \
                for tid in range(ncpu)]
  for i, source in enumerate(recipe['sources']):
    workqueues[i % ncpu][0].append(source[0])

  procs = [_enqueue(workqueues[tid]) for tid in range(ncpu)]
  result = []
  for tid in range(ncpu):
    result.append(_wait(procs[tid], workqueues[tid]))

  for tid in range(ncpu):
    if not all(x == 0 for x in result[tid]['retcode']):
      with open(os.path.join(dirpath, 'log_cpu{}.txt'.format(tid)), 'w') as f:
        f.write(result[tid]['stdout'] + '\n')
      raise TreeliteError('Error occured in worker #{}: '.format(tid) +\
                          '{}'.format(result[tid]['stdout']))

  # 2. Package objects into a dynamic shared library (.dll)
  if verbose:
    log_info(__file__, lineno(),
             'Generating dynamic shared library {}...'\
                     .format(os.path.join(dirpath, recipe['target'] + '.dll')))
  result = _create_dll(os.path.abspath(dirpath),
                       recipe['target'], recipe['sources'], options)
  if result['retcode'] != 0:
    with open(os.path.join(dirpath, 'log_dll.txt'), 'w') as f:
      f.write(result['stdout'] + '\n')
    raise TreeliteError('Error occured while creating DLL: ' +\
                        '{}'.format(result['stdout']))

  # 3. Clean up
  for tid in range(ncpu):
    os.remove(os.path.join(dirpath, 'retcode_cpu{}.txt').format(tid))
  os.remove(os.path.join(dirpath, 'retcode_dll.txt'))

  # Return full path of shared library
  return os.path.join(os.path.abspath(dirpath), recipe['target'] + '.dll')

def _check_ext(dllpath):
  fileext = os.path.splitext(dllpath)[1]
  if fileext != '.dll':
    raise ValueError('Library file should have .dll extension')

__all__ = []
