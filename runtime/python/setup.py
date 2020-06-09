# coding: utf-8
"""Setup Treelite runtime package."""
import os
import shutil
import collections
import logging
import distutils
from platform import system
from setuptools import setup, find_packages
from setuptools.command import sdist, install_lib, install

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

UserOption = collections.namedtuple('UserOption', 'description is_boolean value')

USER_OPTIONS = {
    'cmake-build-dir': UserOption(description='Build directory used for CMake build',
                                  value='build', is_boolean=False)
}

NEED_CLEAN_TREE = set()
NEED_CLEAN_FILE = set()
BUILD_TEMP_DIR = None


def lib_name():
    """Return platform dependent shared object name."""
    if system() == 'Linux' or system().upper().endswith('BSD'):
        name = 'libtreelite_runtime.so'
    elif system() == 'Darwin':
        name = 'libtreelite_runtime.dylib'
    elif system() == 'Windows':
        name = 'treelite_runtime.dll'
    else:
        raise RuntimeError('Unsupported operating system')
    return name


def clean_up():
    """Removed copied files."""
    for path in NEED_CLEAN_TREE:
        shutil.rmtree(path)
    for path in NEED_CLEAN_FILE:
        os.remove(path)


class DefunctSDist(sdist.sdist):  # pylint: disable=too-many-ancestors
    """Explicitly disallow sdist"""

    def run(self):
        raise NotImplementedError(
            '"python setup.py sdist" command is not available. ' +
            'Refer to official build doc at https://treelite.readthedocs.io/')


class InstallLib(install_lib.install_lib):
    logger = logging.getLogger('Treelite runtime install_lib')

    def install(self):

        outfiles = super().install()

        global BUILD_TEMP_DIR  # pylint: disable=global-statement

        # Copy shared library
        libruntime_name = lib_name()
        dst_dir = os.path.join(self.install_dir, 'treelite_runtime', 'lib')
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        dst = os.path.join(dst_dir, libruntime_name)
        # CMake build dir is specified relative to the project root directory
        src_dir = os.path.join(
            CURRENT_DIR, os.path.pardir, os.path.pardir, USER_OPTIONS['cmake-build-dir'].value)
        assert os.path.isdir(src_dir)
        src = os.path.join(src_dir, libruntime_name)
        if not os.path.exists(src):
            raise Exception(
                f'Did not find {libruntime_name} from directory {os.path.normpath(src_dir)}. ' +
                f'Run CMake first to build shard lib {libruntime_name}.'
            )
        self.logger.info(f'Installing shared library: {src} -> {dst}')
        dst, _ = self.copy_file(src, dst)
        outfiles.append(dst)

        # Copy VERSION
        dst_dir = os.path.join(self.install_dir, 'treelite_runtime')
        assert os.path.isdir(dst_dir)
        dst = os.path.join(dst_dir, 'VERSION')
        src = os.path.join(CURRENT_DIR, 'treelite_runtime', 'VERSION')
        assert os.path.exists(src)
        self.logger.info(f'Installing VERSION: {src} -> {dst}')
        dst, _ = self.copy_file(src, dst)
        outfiles.append(dst)

        return outfiles


class Install(install.install):  # pylint: disable=too-many-instance-attributes
    logger = logging.getLogger('Treelite runtime install')
    user_options = install.install.user_options + list(
        (k + ('' if v.is_boolean else '='), None, v.description) for k, v in USER_OPTIONS.items())

    def initialize_options(self):
        super().initialize_options()
        for k, v in USER_OPTIONS.items():
            arg = k.replace('-', '_')
            setattr(self, arg, v.value)

    def finalize_options(self):
        super().finalize_options()
        # CMake build dir is specified relative to the project root directory
        x = os.path.normpath(os.path.join(
            CURRENT_DIR, os.path.pardir, os.path.pardir, self.cmake_build_dir))
        assert os.path.exists(x), (
                f'Build directory "{self.cmake_build_dir}" (Full path: {x}) does not exist. ' +
                f'Specify the directory in which you ran CMake and built {lib_name()}, by ' +
                f'modifying the install command as follows: \n' +
                f'    python setup.py install --cmake-build-dir=<CMake build directory>\n'
        )
        self.logger.info(f'Looking for {lib_name()} from directory {x}')

    def run(self):
        for k, v in USER_OPTIONS.items():
            arg = k.replace('-', '_')
            if hasattr(self, arg):
                USER_OPTIONS[k] = USER_OPTIONS[k]._replace(value=getattr(self, arg))
        super().run()


if __name__ == '__main__':
    # Supported commands:
    # From PyPI:
    # - pip install treelite_runtime
    # From source tree `treelite/runtime/python`:
    # - python setup.py install
    # - python setup.py bdist_wheel && pip install <wheel-name>
    logging.basicConfig(level=logging.INFO)
    setup(name='treelite_runtime',
          version=open(os.path.join(CURRENT_DIR, 'treelite_runtime/VERSION')).read().strip(),
          description='Treelite runtime',
          install_requires=['numpy', 'scipy'],
          cmdclass={
              'sdist': DefunctSDist,
              'install_lib': InstallLib,
              'install': Install
          },
          author='DMLC',
          maintainer='Hyunsu Cho',
          maintainer_email='chohyu01@cs.washington.edu',
          zip_safe=False,
          packages=find_packages(),
          include_package_data=True,
          license='Apache-2.0',
          classifiers=['License :: OSI Approved :: Apache Software License',
                       'Development Status :: 5 - Production/Stable',
                       'Operating System :: OS Independent',
                       'Programming Language :: Python',
                       'Programming Language :: Python :: 3',
                       'Programming Language :: Python :: 3.6',
                       'Programming Language :: Python :: 3.7',
                       'Programming Language :: Python :: 3.8'],
          python_requires='>=3.6',
          url='https://github.com/dmlc/treelite')

    clean_up()
