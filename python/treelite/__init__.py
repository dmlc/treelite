# coding: utf-8

from __future__ import absolute_import as _abs
from .compat import assert_python_min_ver
assert_python_min_ver('2.7', '3.1', 'importlib')
from importlib import import_module

# package will re-export public members of the following scripts/subpackages:
core_packages = ['.core', '.frontend', '.annotator', '.contrib']

__all__ = []
for package in core_packages:
  module = import_module(package, __name__)
  for public_member in module.__all__:
    globals()[public_member] = vars(module).get(public_member)
  __all__ += module.__all__

# runtime is not exposed here -- user should import it separately
