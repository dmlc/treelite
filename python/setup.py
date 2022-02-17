# coding: utf-8
"""Setup Treelite package."""
import os
import shutil
import subprocess
import collections
import logging
from platform import system
from setuptools import setup, find_packages, Extension
from setuptools.command import build_ext, sdist, install_lib, install

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
        name = 'libtreelite.so'
    elif system() == 'Darwin':
        name = 'libtreelite.dylib'
    elif system() == 'Windows':
        name = 'treelite.dll'
    else:
        raise RuntimeError('Unsupported operating system')
    return name


def copy_tree(src_dir, target_dir):
    """Copy source tree into build directory."""
    logger = logging.getLogger('Treelite copy_tree')
    def clean_copy_tree(src, dst):
        logger.info('Copy tree %s -> %s', src, dst)
        shutil.copytree(src, dst)
        NEED_CLEAN_TREE.add(os.path.abspath(dst))

    def clean_copy_file(src, dst):
        logger.info('Copy tree %s -> %s', src, dst)
        shutil.copy(src, dst)
        NEED_CLEAN_FILE.add(os.path.abspath(dst))

    cmake = os.path.join(src_dir, 'cmake')
    inc = os.path.join(src_dir, 'include')
    src = os.path.join(src_dir, 'src')

    clean_copy_tree(cmake, os.path.join(target_dir, 'cmake'))
    clean_copy_tree(inc, os.path.join(target_dir, 'include'))
    clean_copy_tree(src, os.path.join(target_dir, 'src'))

    cmake_list = os.path.join(src_dir, 'CMakeLists.txt')
    clean_copy_file(cmake_list, os.path.join(target_dir, 'CMakeLists.txt'))
    lic = os.path.join(src_dir, 'LICENSE')
    clean_copy_file(lic, os.path.join(target_dir, 'LICENSE'))


def clean_up():
    """Removed copied files."""
    for path in NEED_CLEAN_TREE:
        shutil.rmtree(path)
    for path in NEED_CLEAN_FILE:
        os.remove(path)


class CMakeExtension(Extension):  # pylint: disable=too-few-public-methods
    """Wrapper for libraries built with CMake"""
    def __init__(self, name):
        super().__init__(name=name, sources=[])


class SDist(sdist.sdist):       # pylint: disable=too-many-ancestors
    """Pack C++ sources into the Python package"""
    logger = logging.getLogger('Treelite sdist')

    def run(self):
        copy_tree(os.path.join(CURRENT_DIR, os.path.pardir),
                  os.path.join(CURRENT_DIR, 'treelite'))
        super().run()


class BuildExt(build_ext.build_ext):  # pylint: disable=too-many-ancestors
    """Custom build_ext command using CMake."""

    logger = logging.getLogger('Treelite build_ext')

    # pylint: disable=too-many-arguments,no-self-use
    def build(self, src_dir, build_dir, generator, build_tool=None):
        """Build the core library with CMake."""
        cmake_cmd = ['cmake', src_dir, generator]

        # Flag for cross-compiling for Apple Silicon
        # We use environment variable because it's the only way to pass down custom flags
        # through the cibuildwheel package, which otherwise calls `python setup.py bdist_wheel`
        # command.
        if 'CIBW_TARGET_OSX_ARM64' in os.environ:
            cmake_cmd.append("-DCMAKE_OSX_ARCHITECTURES=arm64")

        self.logger.info('Run CMake command: %s', str(cmake_cmd))
        subprocess.check_call(cmake_cmd, cwd=build_dir)

        if system() != 'Windows':
            nproc = os.cpu_count()
            build_cmd = [build_tool, 'treelite', '-j' + str(nproc)]
            subprocess.check_call(build_cmd, cwd=build_dir)
        else:
            subprocess.check_call(['cmake', '--build', '.', '--config', 'Release',
                                   '--target', 'treelite'], cwd=build_dir)

    def build_cmake_extension(self):
        """Configure and build using CMake"""
        src_dir = 'treelite'
        try:
            copy_tree(os.path.join(CURRENT_DIR, os.path.pardir),
                      os.path.join(self.build_temp, src_dir))
        except Exception:  # pylint: disable=broad-except
            copy_tree(src_dir, os.path.join(self.build_temp, src_dir))
        build_dir = self.build_temp
        global BUILD_TEMP_DIR  # pylint: disable=global-statement
        BUILD_TEMP_DIR = build_dir
        libtreelite = os.path.abspath(
            os.path.join(CURRENT_DIR, os.path.pardir, USER_OPTIONS['cmake-build-dir'].value,
                         lib_name()))

        if os.path.exists(libtreelite):
            self.logger.info('Found shared library, skipping build.')
            return

        self.logger.info('Building from source. %s', lib_name())
        if not os.path.exists(build_dir):
            os.mkdir(build_dir)
        if shutil.which('ninja'):
            build_tool = 'ninja'
        else:
            build_tool = 'make'

        if system() == 'Windows':
            # Pick up from LGB, just test every possible tool chain.
            for vs in ('-GVisual Studio 16 2019', '-GVisual Studio 15 2017',
                       '-GVisual Studio 14 2015', '-GMinGW Makefiles'):
                try:
                    self.build(src_dir, build_dir, vs)
                    self.logger.info(
                        '%s is used for building Windows distribution.', vs)
                    break
                except subprocess.CalledProcessError:
                    continue
        else:
            gen = '-GNinja' if build_tool == 'ninja' else '-GUnix Makefiles'
            self.build(src_dir, build_dir, gen, build_tool)

    def build_extension(self, ext):
        """Override the method for dispatching."""
        if isinstance(ext, CMakeExtension):
            self.build_cmake_extension()
        else:
            super().build_extension(ext)

    def copy_extensions_to_source(self):
        """Dummy override.  Invoked during editable installation."""
        if not os.path.exists(
                os.path.join(CURRENT_DIR, os.path.pardir, USER_OPTIONS['cmake-build-dir'].value,
                             lib_name())):
            raise ValueError('For using editable installation, please ' +
                             'build the shared object first with CMake.')


class InstallLib(install_lib.install_lib):
    """Install native library into Python package"""
    logger = logging.getLogger('Treelite install_lib')

    def install(self):

        outfiles = super().install()

        # Copy shared library
        libtreelite_name = lib_name()
        dst_dir = os.path.join(self.install_dir, 'treelite', 'lib')
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        dst = os.path.join(dst_dir, libtreelite_name)
        # CMake build dir is specified relative to the project root directory
        src_dir = os.path.join(CURRENT_DIR, os.path.pardir, USER_OPTIONS['cmake-build-dir'].value)
        if os.path.isdir(src_dir):
            # The library was built by CMake
            src = os.path.join(src_dir, libtreelite_name)
            if not os.path.exists(src):
                raise Exception(
                    f'Did not find {libtreelite_name} from directory {os.path.normpath(src_dir)}.' +
                    f' Run CMake first to build shared lib {libtreelite_name}.'
                )
            self.logger.info('Using %s built by CMake', libtreelite_name)
        else:
            # The library was built by setup.py
            assert BUILD_TEMP_DIR is not None
            build_dir = BUILD_TEMP_DIR
            src = os.path.join(build_dir, libtreelite_name)
            assert os.path.exists(src)
            self.logger.info('Using %s built by setup.py', libtreelite_name)
        self.logger.info('Installing shared library: %s -> %s', src, dst)
        dst, _ = self.copy_file(src, dst)
        outfiles.append(dst)

        # Copy VERSION
        dst_dir = os.path.join(self.install_dir, 'treelite')
        assert os.path.isdir(dst_dir)
        dst = os.path.join(dst_dir, 'VERSION')
        src = os.path.join(CURRENT_DIR, 'treelite', 'VERSION')
        assert os.path.exists(src)
        self.logger.info('Installing VERSION: %s -> %s', src, dst)
        dst, _ = self.copy_file(src, dst)
        outfiles.append(dst)

        return outfiles


class Install(install.install):  # pylint: disable=too-many-instance-attributes
    """Parse user options"""
    logger = logging.getLogger('Treelite install')
    user_options = install.install.user_options + list(
        (k + ('' if v.is_boolean else '='), None, v.description) for k, v in USER_OPTIONS.items())

    def initialize_options(self):
        super().initialize_options()
        for k, v in USER_OPTIONS.items():
            arg = k.replace('-', '_')
            setattr(self, arg, v.value)

    def run(self):
        for k, _ in USER_OPTIONS.items():
            arg = k.replace('-', '_')
            if hasattr(self, arg):
                USER_OPTIONS[k] = USER_OPTIONS[k]._replace(value=getattr(self, arg))
        super().run()


if __name__ == '__main__':
    # Supported commands:
    # From PyPI:
    # - pip install treelite
    # From source tree `treelite/python`:
    # - python setup.py install
    # - python setup.py bdist_wheel && pip install <wheel-name>
    with open(os.path.join(CURRENT_DIR, 'treelite/VERSION'), 'r', encoding='UTF-8') as f:
        version = f.read().strip()
    logging.basicConfig(level=logging.INFO)
    setup(name='treelite',
          version=version,
          description='Treelite: model compiler for decision trees',
          install_requires=['numpy', 'scipy'],
          ext_modules=[CMakeExtension('libtreelite')],
          cmdclass={
              'build_ext': BuildExt,
              'sdist': SDist,
              'install_lib': InstallLib,
              'install': Install
          },
          extras_require={
              'scikit-learn': ['scikit-learn']
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
