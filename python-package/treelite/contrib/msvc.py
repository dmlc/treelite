# coding: utf-8
"""
Tools to interact with Microsoft Visual C++ (MSVC) compiler
"""

from __future__ import absolute_import as _abs
from ..core import _LIB, _check_call, TreeliteError
from ..compat import PY3
from .util import _str_decode, _str_encode

import os
import subprocess
import json
import ctypes
from multiprocessing import cpu_count

def _varsall_bat_path():
  _LIB.TreeliteVarsallBatPath.restype = ctypes.c_char_p
  return _str_decode(_LIB.TreeliteVarsallBatPath())

def _enqueue(args):
  queue = args[0]
  id = args[1]
  dirpath = args[2]
  proc = subprocess.Popen('cmd.exe', shell=True, stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
  proc.stdin.write(_str_encode('\"{}\" x64\n'.format(_varsall_bat_path())))
  proc.stdin.write(_str_encode('cd {}\n'.format(dirpath)))
  proc.stdin.write(_str_encode('type NUL > retcode_cpu{}.txt\n'.format(id)))
  for source in queue:
    proc.stdin.write(_str_encode('cl.exe /c /openmp /Ox {}\n'
                                 .format(source + '.c')))
    proc.stdin.write(_str_encode('echo %errorlevel% >> retcode_cpu{}.txt\n'
                                 .format(id)))
  proc.stdin.flush()

  return proc

def _wait(proc, args):
  id = args[1]
  dirpath = args[2]
  stdout, _ = proc.communicate()
  with open(os.path.join(dirpath, 'retcode_cpu{}.txt'.format(id)), 'r') as f:
    retcode = [int(line) for line in f]
  return {'stdout':_str_decode(stdout), 'retcode':retcode}

def _create_dll(dirpath, target, sources):
  proc = subprocess.Popen('cmd.exe', shell=True, stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
  proc.stdin.write(_str_encode('\"{}\" x64\n'.format(_varsall_bat_path())))
  proc.stdin.write(_str_encode('cd {}\n'.format(dirpath)))
  proc.stdin.write(_str_encode('cl.exe /LD /Fe{} /openmp {}\n'
                  .format(target, ' '.join([x[0] + '.obj' for x in sources]))))
  proc.stdin.write(_str_encode('echo %errorlevel% > retcode_dll.txt\n'))
  proc.stdin.flush()
  stdout, _ = proc.communicate()
  with open(os.path.join(dirpath, 'retcode_dll.txt'), 'r') as f:
    retcode = int(f.readline())
  return {'stdout':_str_decode(stdout), 'retcode':retcode}

def _create_shared(dirpath, recipe, nthread, options):
  # 1. Compile sources in parallel
  ncore = cpu_count()
  ncpu = min(ncore, nthread) if nthread is not None else ncore
  workqueues = [([], id, os.path.abspath(dirpath)) for id in range(ncpu)]
  long_time_warning = False
  for i, source in enumerate(recipe['sources']):
    workqueues[i % ncpu][0].append(source[0])
    if source[1] > 10000:
      long_time_warning = True

  if long_time_warning:
    print('\033[1;31mWARNING: some of the source files are long. ' +\
          'Expect long compilation time.\u001B[0m\n'+\
          '         You may want to adjust the parameter ' +\
          '\x1B[33mparallel_comp\u001B[0m.\n')

  procs = [_enqueue(workqueues[id]) for id in range(ncpu)]
  result = []
  for id in range(ncpu):
    result.append(_wait(procs[id], workqueues[id]))

  for id in range(ncpu):
    if not all(x == 0 for x in result[id]['retcode']):
      with open(os.path.join(dirpath, 'log_cpu{}.txt'.format(id)), 'w') as f:
        f.write(result[id]['stdout'] + '\n')
      raise TreeliteError('Error occured in worker #{}. '.format(id) +\
                          'See log_cpu{}.txt for details'.format(id))

  # 2. Package objects into a dynamic shared library (.dll)
  result = _create_dll(os.path.abspath(dirpath),
                        recipe['target'], recipe['sources'])
  if result['retcode'] != 0:
    with open(os.path.join(dirpath, 'log_dll.txt'), 'w') as f:
        f.write(result['stdout'] + '\n')
    raise TreeliteError('Error occured while creating DLL.' +\
                        'See log_dll.txt for details.')

  # 3. Clean up
  for id in range(ncpu):
    os.remove(os.path.join(dirpath, 'retcode_cpu{}.txt').format(id))
  os.remove(os.path.join(dirpath, 'retcode_dll.txt'))

__all__ = ['']
