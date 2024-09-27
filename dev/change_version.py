"""
This script changes the version field in different parts of the code base.
"""

import argparse
import pathlib
import re
from typing import Optional, TypeVar

R = TypeVar("R")
ROOT = pathlib.Path(__file__).parent.parent.expanduser().resolve()
PY_PACKAGE = ROOT / "python"


def update_cmake(major: int, minor: int, patch: int) -> None:
    """Change version in CMakeLists.txt"""
    version = f"{major}.{minor}.{patch}"
    with open(ROOT / "CMakeLists.txt", "r", encoding="utf-8") as fd:
        cmakelist = fd.read()
    pattern = r"project\(treelite LANGUAGES .* VERSION ([0-9]+\.[0-9]+\.[0-9]+)\)"
    matched = re.search(pattern, cmakelist)
    assert matched, "Couldn't find the version string in CMakeLists.txt."
    cmakelist = cmakelist[: matched.start(1)] + version + cmakelist[matched.end(1) :]
    with open(ROOT / "CMakeLists.txt", "w", encoding="utf-8") as fd:
        fd.write(cmakelist)


def update_pypkg(
    major: int,
    minor: int,
    patch: int,
    *,
    is_rc: bool,
    is_dev: bool,
    rc_ver: Optional[int] = None,
) -> None:
    """Change version in the Python package"""
    version = f"{major}.{minor}.{patch}"
    if is_rc:
        assert rc_ver
        version = version + f"rc{rc_ver}"
    if is_dev:
        version = version + "-dev"

    pyver_path = PY_PACKAGE / "treelite" / "VERSION"
    with open(pyver_path, "w", encoding="utf-8") as fd:
        fd.write(version + "\n")

    pyprj_path = PY_PACKAGE / "pyproject.toml"
    with open(pyprj_path, "r", encoding="utf-8") as fd:
        pyprj = fd.read()
    matched = re.search('version = "' + r"([0-9]+\.[0-9]+\.[0-9]+.*)" + '"', pyprj)
    assert matched, "Couldn't find version string in pyproject.toml."
    pyprj = pyprj[: matched.start(1)] + version + pyprj[matched.end(1) :]
    with open(pyprj_path, "w", encoding="utf-8") as fd:
        fd.write(pyprj)


def main(args: argparse.Namespace) -> None:
    """Perform version change in all relevant parts of the code base."""
    if args.is_rc and args.is_dev:
        raise ValueError("A release version cannot be both RC and dev.")
    if args.is_rc:
        assert args.rc is not None, "rc field must be specified if is_rc is specified"
        assert args.rc >= 1, "RC version must start from 1."
    else:
        assert args.rc is None, "is_rc must be specified in order to specify rc field"
    update_cmake(args.major, args.minor, args.patch)
    update_pypkg(
        args.major,
        args.minor,
        args.patch,
        is_rc=args.is_rc,
        is_dev=args.is_dev,
        rc_ver=args.rc,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--major", type=int, required=True)
    parser.add_argument("--minor", type=int, required=True)
    parser.add_argument("--patch", type=int, required=True)
    parser.add_argument("--rc", type=int)
    parser.add_argument("--is-rc", type=int, choices=[0, 1], default=0)
    parser.add_argument("--is-dev", type=int, choices=[0, 1], default=0)
    parsed_args = parser.parse_args()
    main(parsed_args)
