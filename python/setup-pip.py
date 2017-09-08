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

lib_basename = os.path.basename(LIB_PATH[0])
lib_dest = os.path.join('./treelite', lib_basename)
rt_basename = os.path.basename(RT_PATH[0])
rt_dest = os.path.join('./treelite', rt_basename)

if not os.path.exists(lib_dest):
  shutil.copy(LIB_PATH[0], lib_dest)
if not os.path.exists(rt_dest):
  shutil.copy(RT_PATH[0], rt_dest)

setup(
    name='treelite',
    version='0.1',
    description='treelite: fast tree prediction',
    packages=find_packages(),
    package_data={
        'treelite': [lib_basename, rt_basename],
    },
    distclass=BinaryDistribution
)
