from setuptools import setup, Distribution, find_packages
import os
import shutil

class BinaryDistribution(Distribution):
  def has_ext_modules(foo):
    return True

lib_name = 'treelite'

libpath_py = os.path.abspath('./treelite/libpath.py')
libpath = {'__file__': libpath_py}
exec(compile(open(libpath_py, "rb").read(), libpath_py, 'exec'),
     libpath, libpath)
LIB_PATH = libpath['find_lib_path']()              # main lib path
RT_PATH = libpath['find_lib_path'](runtime=True)   # runtime lib path
if len(LIB_PATH) == 0 or len(RT_PATH) == 0:
  raise RuntimeError('Please compile the C++ package first')

if os.path.abspath(os.path.dirname(LIB_PATH[0])) == os.path.abspath('./treelite'):
  # remove stale copies of library
  del LIB_PATH[0]
  del RT_PATH[0]

lib_basename = os.path.basename(LIB_PATH[0])
lib_dest = os.path.join('./treelite', lib_basename)
rt_basename = os.path.basename(RT_PATH[0])
rt_dest = os.path.join('./treelite', rt_basename)

if os.path.exists(lib_dest):
  os.remove(lib_dest)
if os.path.exists(rt_dest):
  os.remove(rt_dest)
shutil.copy(LIB_PATH[0], lib_dest)
shutil.copy(RT_PATH[0], rt_dest)

with open('../VERSION', 'r') as f:
  VERSION = f.readlines()[0]

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
        'treelite': [lib_basename, rt_basename],
    },
    distclass=BinaryDistribution
)
