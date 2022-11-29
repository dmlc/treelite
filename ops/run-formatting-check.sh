#!/bin/bash

set -euo pipefail

echo "##[section]Installing pylint and cpplint..."
source activate dev

echo "##[section]Running pylint and cpplint..."
make lint
