import os
from distutils.sysconfig import get_python_lib
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Distutils.build_ext import new_build_ext as build_ext

extensions = [
    Extension('*',
              sources=['*.pyx'],
              include_dirs=['../../include', '../../build/include'],
              library_dirs=[get_python_lib(), '../../build/'],
              #runtime_library_dirs=[os.path.join(os.sys.prefix, 'lib')],
              libraries=['treelite'],
              language='c++',
              extra_compile_args=['--std=c++14'])
]

setup(
    name='cython_test',
    version='0.0.1',
    setup_requires=['cython'],
    ext_modules=extensions,
    packages=find_packages(),
    install_requires=['cython'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False
)
