# coding: utf-8
"""
Tools to interact with clang toolchain
"""

from __future__ import absolute_import as _abs
import os
import subprocess
from multiprocessing import cpu_count
from sys import platform as _platform
from ..common.compat import _str_decode, _str_encode
from ..common.util import TreeliteError, lineno, log_info, TemporaryDirectory

if _platform == 'darwin':
  LIBEXT = '.dylib'
else:
  LIBEXT = '.so'

def _openmp_supported():
  # make temporary folder
  with TemporaryDirectory() as temp_dir:
    filename = os.path.join(temp_dir, 'test.c')
    with open(filename, 'w') as f:
      f.write('int main() { return 0; }\n')
    retcode = subprocess.call('clang {} -fopenmp'.format(filename), shell=True,
                              stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
  return retcode == 0

def _enqueue(args):
  queue = args[0]
  tid = args[1]
  dirpath = args[2]
  options = args[3]
  proc = subprocess.Popen(os.environ['SHELL'], shell=True,
                          stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
  proc.stdin.write(_str_encode('cd {}\n'.format(dirpath)))
  proc.stdin.write(_str_encode(' > retcode_cpu{}.txt\n'.format(tid)))
  for source in queue:
    proc.stdin.write(_str_encode('clang -c -O3 -o {} {} '\
                                 .format(source + '.o', source + '.c') +\
                                 '-fPIC -std=c99 -flto {}\n'\
                                 .format(' '.join(options))))
    proc.stdin.write(_str_encode('echo $? >> retcode_cpu{}.txt\n'.format(tid)))
  proc.stdin.flush()

  return proc

def _wait(proc, args):
  tid = args[1]
  dirpath = args[2]
  stdout, _ = proc.communicate()
  with open(os.path.join(dirpath, 'retcode_cpu{}.txt'.format(tid)), 'r') as f:
    retcode = [int(line) for line in f]
  return {'stdout':_str_decode(stdout), 'retcode':retcode}

def _create_lib(dirpath, target, sources, options):
  proc = subprocess.Popen(os.environ['SHELL'], shell=True,
                          stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
  proc.stdin.write(_str_encode('cd {}\n'.format(dirpath)))
  proc.stdin.write(_str_encode('clang -shared -O3 -o {} {} '\
                               .format(
                                   target + LIBEXT,
                                   ' '.join([x[0] + '.o' for x in sources]))+\
                                     '-std=c99 -flto {}\n'\
                                     .format(' '.join(options))))
  proc.stdin.write(_str_encode('echo $? > retcode_lib.txt\n'))
  proc.stdin.flush()
  stdout, _ = proc.communicate()
  with open(os.path.join(dirpath, 'retcode_lib.txt'), 'r') as f:
    retcode = int(f.readline())
  return {'stdout':_str_decode(stdout), 'retcode':retcode}

def _create_shared(dirpath, recipe, nthread, options, verbose):
  if _openmp_supported():  # clang may not support OpenMP, so make it optional
    options += ['-fopenmp']

  # 1. Compile sources in parallel
  if verbose:
    log_info(__file__, lineno(),
             'Compiling sources files in directory {} '.format(dirpath) +\
             'into object files (*.o)...')
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

  # 2. Package objects into a dynamic shared library (.so/.dylib)
  if verbose:
    log_info(__file__, lineno(),
             'Generating dynamic shared library {}...'\
                     .format(os.path.join(dirpath, recipe['target'] + LIBEXT)))
  result = _create_lib(os.path.abspath(dirpath),
                       recipe['target'], recipe['sources'], options)
  if result['retcode'] != 0:
    with open(os.path.join(dirpath, 'log_lib.txt'), 'w') as f:
      f.write(result['stdout'] + '\n')
    raise TreeliteError('Error occured while creating dynamic library: ' +\
                        '{}'.format(result['stdout']))

  # 3. Clean up
  for tid in range(ncpu):
    os.remove(os.path.join(dirpath, 'retcode_cpu{}.txt').format(tid))
  os.remove(os.path.join(dirpath, 'retcode_lib.txt'))

  # Return full path of shared library
  return os.path.join(os.path.abspath(dirpath), recipe['target'] + LIBEXT)

def _check_ext(dllpath):
  fileext = os.path.splitext(dllpath)[1]
  if fileext != LIBEXT:
    raise ValueError('Library file should have {} extension'.format(LIBEXT))

__all__ = []
