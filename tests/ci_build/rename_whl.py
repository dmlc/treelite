"""Rename a Python wheel"""

import argparse
import glob
import os
from contextlib import contextmanager


@contextmanager
def cd(path):  # pylint: disable=C0103
    """Temporarily change working directory"""
    path = os.path.normpath(path)
    cwd = os.getcwd()
    os.chdir(path)
    print(f"cd {path}")
    try:
        yield path
    finally:
        os.chdir(cwd)


def main(args):
    """Main function"""
    if not os.path.isdir(args.wheel_dir):
        raise ValueError("wheel_dir argument must be a directory")

    with cd(args.wheel_dir):
        whl_list = list(glob.glob("*.whl"))
        for whl_path in whl_list:
            basename = os.path.basename(whl_path)
            tokens = basename.split("-")
            assert len(tokens) == 5
            keywords = {
                "pkg_name": tokens[0],
                "version": tokens[1],
                "commit_id": args.commit_id,
                "platform_tag": args.platform_tag,
            }
            new_name = (
                "{pkg_name}-{version}+{commit_id}-py3-none-{platform_tag}.whl".format(
                    **keywords
                )
            )
            print(f"Renaming {basename} to {new_name}...")
            os.rename(basename, new_name)


if __name__ == "__main__":
    DESCRIPTION = (
        "Script to rename wheel(s) using a commit ID and platform tag."
        "Note: This script will not recurse into subdirectories."
    )
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("wheel_dir", type=str, help="Directory containing wheels")
    parser.add_argument("commit_id", type=str, help="Hash of current git commit")
    parser.add_argument(
        "platform_tag", type=str, help="Platform tag, PEP 425 compliant"
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
