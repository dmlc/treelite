"""Simple script for preparing a PyPI release.
It fetches Python wheels from the CI pipelines.
tqdm, sh are required to run this script.
"""

import argparse
import os
import subprocess
from typing import List, Optional
from urllib.request import urlretrieve

import tqdm
from packaging import version
from sh.contrib import git

PREFIX = "https://treelite-wheels.s3.amazonaws.com/"
DIST = os.path.join(os.path.curdir, "python", "dist")
RT_DIST = os.path.join(os.path.curdir, "runtime", "python", "dist")


pbar = None


def show_progress(block_num, block_size, total_size):
    "Show file download progress."
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
    print(f"{url} -> {filename}")
    return urlretrieve(url, filename, reporthook=show_progress)


def latest_hash() -> str:
    "Get latest commit hash."
    ret = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True)
    assert ret.returncode == 0, "Failed to get latest commit hash."
    commit_hash = ret.stdout.decode("utf-8").strip()
    return commit_hash


def download_wheels(
    platforms: List[str],
    url_prefix: str,
    dest_dir: str,
    src_filename_prefix: str,
    target_filename_prefix: str,
    ext: Optional[str] = "whl",
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


def download_py_packages(branch: str, version_str: str, commit_hash: str) -> None:
    platforms = [
        "win_amd64",
        "manylinux2014_x86_64",
        "macosx_10_15_x86_64.macosx_11_0_x86_64.macosx_12_0_x86_64",
        "macosx_12_0_arm64",
    ]

    if not os.path.exists(DIST):
        os.mkdir(DIST)

    if not os.path.exists(RT_DIST):
        os.mkdir(RT_DIST)

    # Binary wheels (*.whl)
    for pkg, dest_dir in [("treelite", DIST), ("treelite_runtime", RT_DIST)]:
        src_filename_prefix = f"{pkg}-{version_str}%2B{commit_hash}-py3-none-"
        target_filename_prefix = f"{pkg}-{version_str}-py3-none-"
        filenames = download_wheels(
            platforms, PREFIX, dest_dir, src_filename_prefix, target_filename_prefix
        )
        print(f"List of downloaded wheels: {filenames}\n")

    # Source distribution (*.tar.gz)
    for pkg, dest_dir in [("treelite", DIST), ("treelite_runtime", RT_DIST)]:
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
- Upload pypi package by `python -m twine upload dist/<Package Name>` for all wheels.
- Create sdist package: `make pippack`.
- Upload sdist package to PyPI
- Check the uploaded files on `https://pypi.org/project/treelite/<VERSION>/#files` and `pip
  install treelite==<VERSION>` """
    )


def check_path():
    root = os.path.abspath(os.path.curdir)
    assert os.path.basename(root) == "treelite", "Must be run on project root."


def main(args: argparse.Namespace) -> None:
    check_path()

    rel = version.parse(args.release)
    assert isinstance(rel, version.Version)

    major = rel.major
    minor = rel.minor
    patch = rel.micro

    print(f"Release: {rel}")
    if not rel.is_prerelease:
        # Major release
        rc: Optional[str] = None
        rc_ver: Optional[int] = None
    else:
        # RC release
        major = rel.major
        minor = rel.minor
        patch = rel.micro
        assert rel.pre is not None
        rc, rc_ver = rel.pre
        assert rc == "rc"

    release = str(major) + "." + str(minor) + "." + str(patch)
    if args.branch is not None:
        branch = args.branch
    else:
        branch = "release_" + str(major) + "." + str(minor)

    # git.clean("-xdf")
    git.checkout(branch)
    git.pull("origin", branch)
    commit_hash = latest_hash()

    download_py_packages(branch, args.release, commit_hash)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        type=str,
        required=True,
        help="Version tag, e.g. '1.3.2', or '1.5.0rc1'",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help=(
            "Optional branch. Usually patch releases reuse the same branch of the"
            " major release, but there can be exception."
        ),
    )
    args = parser.parse_args()
    main(args)
