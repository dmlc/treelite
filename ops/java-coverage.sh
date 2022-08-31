#!/bin/bash

set -euo pipefail

echo "##[section]Running integration tests for Java runtime (treelite4j)..."
cd runtime/java/treelite4j
mvn test -DJNI.args=cpp-coverage

echo "##[section]Submitting Java code (treelite4j) coverage data to CodeCov..."
bash <(curl -s https://codecov.io/bash) || echo "Codecov did not collect coverage reports"
