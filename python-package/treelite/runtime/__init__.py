# coding: utf-8
"""
Runtime API of tree-lite python package.

Runtime API provides the *minimum necessary* tools to deploy tree prediction
modules in the wild.
"""

from __future__ import absolute_import as _abs
import sys
from ..compat import assert_python_min_ver
assert_python_min_ver('2.7', '3.1', 'importlib')
from importlib import import_module

# package will re-export public members of the following scripts/subpackages:
core_packages = ['.predictor']

__all__ = []
for package in core_packages:
  module = import_module(package, __name__)
  for public_member in module.__all__:
    globals()[public_member] = vars(module).get(public_member)
  __all__ += module.__all__
