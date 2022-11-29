#!/bin/bash

set -euo pipefail

echo "##[section]Update Conda"
# Workaround for mamba-org/mamba#488
rm /usr/local/miniconda/pkgs/cache/*.json
conda update -n base -c conda-forge -q -y conda