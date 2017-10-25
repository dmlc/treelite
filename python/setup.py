"""Setup script"""
import os
import shutil
from setuptools import setup, Distribution, find_packages

class BinaryDistribution(Distribution):
  """Overrides Distribution class to bundle platform-specific binaries"""
  # pylint: disable=R0201
  def has_ext_modules(self):
    """Has an extension module"""
    return True

LIBPATH_PY = os.path.abspath('./treelite/common/libpath.py')
LIBPATH = {'__file__': LIBPATH_PY}
# pylint: disable=W0122
exec(compile(open(LIBPATH_PY, "rb").read(), LIBPATH_PY, 'exec'),
     LIBPATH, LIBPATH)
LIB_PATH = LIBPATH['find_lib_path']()              # main lib path
RT_PATH = LIBPATH['find_lib_path'](runtime=True)   # runtime lib path
if (not LIB_PATH) or (not RT_PATH):
  raise RuntimeError('Please compile the C++ package first')

if os.path.abspath(os.path.dirname(LIB_PATH[0])) == os.path.abspath('./treelite'):
  # remove stale copies of library
  del LIB_PATH[0]
  del RT_PATH[0]

LIB_BASENAME = os.path.basename(LIB_PATH[0])
LIB_DEST = os.path.join('./treelite', LIB_BASENAME)
RT_BASENAME = os.path.basename(RT_PATH[0])
RT_DEST = os.path.join('./treelite', RT_BASENAME)

if os.path.exists(LIB_DEST):
  os.remove(LIB_DEST)
if os.path.exists(RT_DEST):
  os.remove(RT_DEST)
shutil.copy(LIB_PATH[0], LIB_DEST)
shutil.copy(RT_PATH[0], RT_DEST)

with open('../VERSION', 'r') as f:
  VERSION = f.readlines()[0].rstrip('\n')

setup(
    name='treelite',
    version=VERSION,
    description='treelite: fast tree prediction',
    url='http://treelite.io',
    author='DMLC',
    maintainer='Hyunsu Cho',
    maintainer_email='chohyu01@cs.washington.edu',
    packages=find_packages(),
    package_data={
        'treelite': [LIB_BASENAME, RT_BASENAME],
    },
    distclass=BinaryDistribution
)
