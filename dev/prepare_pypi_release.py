"""Simple script for preparing a PyPI release.
It fetches Python wheels from the CI pipelines.
tqdm, sh are required to run this script.
"""

# pylint: disable=W0603,C0103,too-many-arguments

import argparse
import os
import subprocess
from typing import List
from urllib.request import urlretrieve

import tqdm
from packaging import version

PREFIX = "https://treelite-wheels.s3.amazonaws.com/"
DIST = os.path.join(os.path.curdir, "python", "dist")


pbar = None


def show_progress(block_num, block_size, total_size):
    """Show file download progress."""
    global pbar
    if pbar is None:
        pbar = tqdm.tqdm(total=total_size / 1024, unit="kB")

    downloaded = block_num * block_size
    if downloaded < total_size:
        upper = (total_size - downloaded) / 1024
        pbar.update(min(block_size / 1024, upper))
    else:
        pbar.close()
        pbar = None


def retrieve(url, filename=None):
    """Download URL"""
    print(f"{url} -> {filename}")
    return urlretrieve(url, filename, reporthook=show_progress)


def latest_hash() -> str:
    """Get latest commit hash."""
    ret = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True)
    assert ret.returncode == 0, "Failed to get latest commit hash."
    commit_hash = ret.stdout.decode("utf-8").strip()
    return commit_hash


def download_wheels(
    platforms: List[str],
    url_prefix: str,
    dest_dir: str,
    src_filename_prefix: str,
    target_filename_prefix: str,
    ext: str = "whl",
) -> List[str]:
    """Download all binary wheels. url_prefix is the URL for remote directory storing
    the release wheels
    """

    filenames = []
    for platform in platforms:
        src_wheel = src_filename_prefix + platform + "." + ext
        url = url_prefix + src_wheel

        target_wheel = target_filename_prefix + platform + "." + ext
        print(f"{src_wheel} -> {target_wheel}")
        filename = os.path.join(dest_dir, target_wheel)
        filenames.append(filename)
        retrieve(url=url, filename=filename)
    return filenames


def download_py_packages(version_str: str, commit_hash: str) -> None:
    """Download Python package files"""
    platforms = [
        "win_amd64",
        "manylinux2014_x86_64",
        "manylinux2014_aarch64",
        "macosx_10_15_x86_64.macosx_11_0_x86_64.macosx_12_0_x86_64",
        "macosx_12_0_arm64",
    ]

    if not os.path.exists(DIST):
        os.mkdir(DIST)

    # Binary wheels (*.whl)
    for pkg, dest_dir in [("treelite", DIST)]:
        src_filename_prefix = f"{pkg}-{version_str}%2B{commit_hash}-py3-none-"
        target_filename_prefix = f"{pkg}-{version_str}-py3-none-"
        filenames = download_wheels(
            platforms, PREFIX, dest_dir, src_filename_prefix, target_filename_prefix
        )
        print(f"List of downloaded wheels: {filenames}\n")

    # Source distribution (*.tar.gz)
    for pkg, dest_dir in [("treelite", DIST)]:
        src_filename_prefix = f"{pkg}-{version_str}%2B{commit_hash}"
        target_filename_prefix = f"{pkg}-{version_str}"
        filenames = download_wheels(
            [""],
            PREFIX,
            dest_dir,
            src_filename_prefix,
            target_filename_prefix,
            "tar.gz",
        )
        print(f"List of downloaded sdist: {filenames}\n")
    print(
        """
Following steps should be done manually:
- Upload pypi package by `python -m twine upload python/dist/* for all wheels.
- Check the uploaded files on `https://pypi.org/project/treelite/<VERSION>/#files` and `pip
  install treelite==<VERSION>` """
    )


def check_path():
    """Ensure that this script is run from the root directory"""
    root = os.path.abspath(os.path.curdir)
    assert os.path.basename(root) == "treelite", "Must be run on project root."


def main(args: argparse.Namespace) -> None:
    """Main function"""
    check_path()

    rel = version.parse(args.release)
    assert isinstance(rel, version.Version)

    print(f"Release: {rel}")
    if rel.is_prerelease:
        # RC release
        assert rel.pre is not None
        rc, _ = rel.pre
        assert rc == "rc"
    commit_hash = latest_hash()

    download_py_packages(args.release, commit_hash)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        type=str,
        required=True,
        help="Version tag, e.g. '1.3.2', or '1.5.0rc1'",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
