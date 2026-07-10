"""Find the path to Treelite dynamic library files."""

import os
import pathlib
import sys
from typing import List


class TreeliteLibraryNotFound(Exception):
    """Error thrown by when Treelite is not found"""


def find_lib_path() -> List[pathlib.Path]:
    """Find the path to Treelite dynamic library files.

    Returns
    -------
    lib_path
       List of all found library path to Treelite
    """
    curr_path = pathlib.Path(__file__).expanduser().absolute().parent
    dll_path = [
        # When installed, libtreelite will be installed in <site-package-dir>/lib
        curr_path / "lib",
        # Editable installation
        curr_path.parent.parent / "build",
        # Use libtreelite from a system prefix, if available. This should be the last option.
        pathlib.Path(sys.base_prefix).expanduser().resolve() / "lib",
    ]

    if sys.platform == "win32":
        # On Windows, Conda may install libs in different paths
        sys_prefix = pathlib.Path(sys.base_prefix)
        dll_path.extend(
            [
                sys_prefix / "bin",
                sys_prefix / "Library",
                sys_prefix / "Library" / "bin",
                sys_prefix / "Library" / "lib",
            ]
        )
        dll_path = [p.joinpath("treelite.dll") for p in dll_path]
    elif sys.platform.startswith(("linux", "freebsd", "emscripten", "OS400")):
        dll_path = [p.joinpath("libtreelite.so") for p in dll_path]
    elif sys.platform == "darwin":
        dll_path = [p.joinpath("libtreelite.dylib") for p in dll_path]
    elif sys.platform == "cygwin":
        dll_path = [p.joinpath("cygtreelite.dll") for p in dll_path]
    else:
        raise RuntimeError(f"Unrecognized platform: {sys.platform}")

    lib_path = [p for p in dll_path if p.exists() and p.is_file()]

    # TREELITE_BUILD_DOC is defined by sphinx conf.
    if not lib_path and not os.environ.get("TREELITE_BUILD_DOC", False):
        link = "https://treelite.readthedocs.io/en/latest/install.html"
        msg = (
            "Cannot find Treelite Library in the candidate path.  "
            + "List of candidates:\n- "
            + ("\n- ".join(str(x) for x in dll_path))
            + "\nTreelite Python package path: "
            + str(curr_path)
            + "\nsys.base_prefix: "
            + sys.base_prefix
            + "\nSee: "
            + link
            + " for installing Treelite."
        )
        raise TreeliteLibraryNotFound(msg)
    return lib_path
