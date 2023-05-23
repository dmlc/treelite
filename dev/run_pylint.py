"""Wrapper for Pylint"""

import os
import pathlib
import subprocess
import sys

ROOT_PATH = pathlib.Path(__file__).parent.parent.expanduser().resolve()
PYPKG_PATH = ROOT_PATH / "python"
PYLINTRC_PATH = PYPKG_PATH / ".pylintrc"


def main():
    """Wrapper for Pylint. Add treelite to PYTHONPATH so that pylint doesn't error out"""
    new_env = os.environ.copy()
    new_env["PYTHONPATH"] = str(PYPKG_PATH)

    # sys.argv[1:]: List of source files to check
    subprocess.run(
        ["pylint", "-rn", "-sn", "--rcfile", str(PYLINTRC_PATH)] + sys.argv[1:],
        check=True,
        env=new_env,
    )


if __name__ == "__main__":
    main()
