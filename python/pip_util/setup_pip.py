# coding: utf-8
"""Setup script"""
from __future__ import print_function
import os
import shutil
import sys
import tempfile
from setuptools import setup, find_packages, Distribution

class BinaryDistribution(Distribution):	
  """Auxilliary class necessary to inform setuptools that this is a	
  non-generic, platform-specific package."""	
  def has_ext_modules(self):	
    return True

if len(sys.argv) < 2 or sys.argv[1] != 'sdist':
  os.system('sh ./treelite/build-python.sh')

  LIBPATH_PY = os.path.abspath('./treelite/libpath.py')
  LIBPATH = {'__file__': LIBPATH_PY}
  # pylint: disable=W0122
  exec(compile(open(LIBPATH_PY, "rb").read(), LIBPATH_PY, 'exec'),
       LIBPATH, LIBPATH)
  
  # Paths for C/C++ libraries
  LIB_PATH = LIBPATH['find_lib_path'](basename='treelite')
  RT_PATH = LIBPATH['find_lib_path'](basename='treelite_runtime')
  
  if (not LIB_PATH) or (not RT_PATH):
    raise RuntimeError('Please compile the C++ package first')
  
  # ignore libraries already in python/treelite; only use ones in ../lib
  if os.path.abspath(os.path.dirname(LIB_PATH[0])) == os.path.abspath('./treelite'):
    del LIB_PATH[0]
    del RT_PATH[0]
  
  LIB_BASENAME = os.path.basename(LIB_PATH[0])
  LIB_DEST = os.path.join('./treelite', LIB_BASENAME)
  RT_BASENAME = os.path.basename(RT_PATH[0])
  RT_DEST = os.path.join('./treelite', RT_BASENAME)
  
  # remove stale copies of library
  if os.path.exists(LIB_DEST):
    os.remove(LIB_DEST)
  if os.path.exists(RT_DEST):
    os.remove(RT_DEST)
  shutil.copy(LIB_PATH[0], LIB_DEST)
  shutil.copy(RT_PATH[0], RT_DEST)
  
  # Create a zipped package containing glue code for deployment
  with tempfile.TemporaryDirectory() as tempdir:
    shutil.copytree('./treelite/runtime/native/', os.path.abspath(os.path.join(tempdir, 'runtime')))
    libpath = os.path.abspath(os.path.join(tempdir, 'runtime', 'lib'))
    filelist = os.path.abspath(os.path.join(tempdir, 'runtime', 'FILELIST'))
    if os.path.exists(libpath):  # remove compiled lib
      shutil.rmtree(libpath)
    if os.path.exists(filelist):
      os.remove(filelist)
    shutil.make_archive(base_name='./treelite/treelite_runtime',
                        format='zip',
                        root_dir=os.path.abspath(tempdir),
                        base_dir='runtime/')
    
  # copy treelite.runtime
  PY_RT_SRC = './treelite/runtime/native/python/treelite_runtime'
  PY_RT_DEST = './treelite/runtime/treelite_runtime'
  if os.path.exists(PY_RT_DEST):
    shutil.rmtree(PY_RT_DEST)
  shutil.copytree(PY_RT_SRC, PY_RT_DEST)
else:
  assert sys.argv[1] == 'sdist'

with open('./treelite/VERSION', 'r') as f:
  VERSION = f.readlines()[0].rstrip('\n')

setup(
    name='treelite',
    version=VERSION,
    description='treelite: toolbox for decision tree deployment',
    url='http://treelite.io',
    author='DMLC',
    maintainer='Hyunsu Cho',
    maintainer_email='chohyu01@cs.washington.edu',
    zip_safe=False,
    include_package_data=True,
    packages=find_packages(),
    install_requires=['numpy', 'scipy'],
    distclass=BinaryDistribution,
    license='Apache-2.0',
    python_requires='>=3.5'
)
