#!/bin/bash

set -euo pipefail

echo "##[section]Running pylint and cpplint..."
make lint
