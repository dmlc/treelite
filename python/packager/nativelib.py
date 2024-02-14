"""
Functions for building libtreelite
"""

import logging
import os
import pathlib
import shutil
import subprocess
import sys
from platform import system
from typing import Optional

from .build_config import BuildConfiguration


def _lib_name() -> str:
    """Return platform dependent shared object name."""
    if system() in ["Linux", "OS400"] or system().upper().endswith("BSD"):
        name = "libtreelite.so"
    elif system() == "Darwin":
        name = "libtreelite.dylib"
    elif system() == "Windows":
        name = "treelite.dll"
    else:
        raise NotImplementedError(f"System {system()} not supported")
    return name


def build_libtreelite(
    cpp_src_dir: pathlib.Path,
    build_dir: pathlib.Path,
    build_config: BuildConfiguration,
) -> pathlib.Path:
    """Build libtreelite in a temporary directory and obtain the path to built libtreelite"""
    logger = logging.getLogger("treelite.packager.build_libtreelite")

    if not cpp_src_dir.is_dir():
        raise RuntimeError(f"Expected {cpp_src_dir} to be a directory")
    logger.info(
        "Building %s from the C++ source files in %s...", _lib_name(), str(cpp_src_dir)
    )

    def _build(*, generator: str) -> None:
        cmake_cmd = [
            "cmake",
            str(cpp_src_dir),
            generator,
        ]
        cmake_cmd.extend(build_config.get_cmake_args())

        # Flag for cross-compiling for Apple Silicon
        # We use environment variable because it's the only way to pass down custom flags
        # through the cibuildwheel package, which calls `pip wheel` command.
        if "CIBW_TARGET_OSX_ARM64" in os.environ:
            cmake_cmd.extend(
                ["-DCMAKE_OSX_ARCHITECTURES=arm64", "-DDETECT_CONDA_ENV=OFF"]
            )

        logger.info("CMake args: %s", str(cmake_cmd))
        subprocess.check_call(cmake_cmd, cwd=build_dir)

        if system() == "Windows":
            subprocess.check_call(
                ["cmake", "--build", ".", "--config", "Release"], cwd=build_dir
            )
        else:
            nproc = os.cpu_count()
            assert build_tool is not None
            subprocess.check_call([build_tool, f"-j{nproc}"], cwd=build_dir)

    if system() == "Windows":
        supported_generators = (
            "-GVisual Studio 17 2022",
            "-GVisual Studio 16 2019",
            "-GVisual Studio 15 2017",
            "-GMinGW Makefiles",
        )
        for generator in supported_generators:
            try:
                _build(generator=generator)
                logger.info(
                    "Successfully built %s using generator %s", _lib_name(), generator
                )
                break
            except subprocess.CalledProcessError as e:
                logger.info(
                    "Tried building with generator %s but failed with exception %s",
                    generator,
                    str(e),
                )
                # Empty build directory
                shutil.rmtree(build_dir)
                build_dir.mkdir()
        else:
            raise RuntimeError(
                "None of the supported generators produced a successful build!"
                f"Supported generators: {supported_generators}"
            )
    else:
        build_tool = "ninja" if shutil.which("ninja") else "make"
        generator = "-GNinja" if build_tool == "ninja" else "-GUnix Makefiles"
        try:
            _build(generator=generator)
        except subprocess.CalledProcessError as e:
            logger.info("Failed to build with OpenMP. Exception: %s", str(e))
            build_config.use_openmp = False
            _build(generator=generator)

    return build_dir / _lib_name()


def locate_local_libtreelite(
    toplevel_dir: pathlib.Path,
    logger: logging.Logger,
) -> Optional[pathlib.Path]:
    """
    Locate libtreelite from the local project directory's lib/ subdirectory.
    """
    libtreelite = toplevel_dir.parent / "build" / _lib_name()
    if libtreelite.exists():
        logger.info("Found %s at %s", libtreelite.name, str(libtreelite.parent))
        return libtreelite
    logger.info("Did not find %s at %s", libtreelite.name, str(libtreelite.parent))
    return None


def locate_or_build_libtreelite(
    toplevel_dir: pathlib.Path,
    build_dir: pathlib.Path,
    build_config: BuildConfiguration,
) -> pathlib.Path:
    """Locate libtreelite; if not exist, build it"""
    logger = logging.getLogger("treelite.packager.locate_or_build_libtreelite")

    if build_config.use_system_libtreelite:
        # If the user explicitly specifies the path for libtreelite, use it
        if build_config.system_libtreelite_dir:
            p = pathlib.Path(build_config.system_libtreelite_dir)
            logger.info(
                "system_libtreelite_dir was specified. Locating %s from path %s...",
                _lib_name(),
                str(p),
            )
            sys_prefix_candidates = [p.expanduser().resolve()]
        else:
            # Find libtreelite from system prefix
            sys_prefix = pathlib.Path(sys.base_prefix)
            sys_prefix_candidates = [
                sys_prefix / "lib",
                # Paths possibly used on Windows
                sys_prefix / "bin",
                sys_prefix / "Library",
                sys_prefix / "Library" / "bin",
                sys_prefix / "Library" / "lib",
            ]
            sys_prefix_candidates = [
                p.expanduser().resolve() for p in sys_prefix_candidates
            ]
        for candidate_dir in sys_prefix_candidates:
            libtreelite_sys = candidate_dir / _lib_name()
            if libtreelite_sys.exists():
                logger.info("Using system treelite: %s", str(libtreelite_sys))
                return libtreelite_sys
        rec_msg = (
            "Make sure that system_libtreelite_dir is a valid path"
            if build_config.system_libtreelite_dir
            else "Consider setting system_libtreelite_dir in the build configuration"
        )
        raise RuntimeError(
            f"use_system_libtreelite was specified but {_lib_name()} is not found. {rec_msg}. "
            "Paths searched (in order): \n"
            + "\n".join([f"* {str(p)}" for p in sys_prefix_candidates])
        )

    libtreelite = locate_local_libtreelite(toplevel_dir, logger=logger)
    if libtreelite is not None:
        return libtreelite

    if toplevel_dir.joinpath("cpp_src").exists():
        # Source distribution; all C++ source files to be found in cpp_src/
        cpp_src_dir = toplevel_dir.joinpath("cpp_src")
    else:
        # Probably running "pip install ." from python-package/
        cpp_src_dir = toplevel_dir.parent
        if not cpp_src_dir.joinpath("CMakeLists.txt").exists():
            raise RuntimeError(f"Did not find CMakeLists.txt from {cpp_src_dir}")
    return build_libtreelite(
        cpp_src_dir, build_dir=build_dir, build_config=build_config
    )
