# coding: utf-8
# pylint: disable=W0611
"""Compatibility layer"""

# optional support for Pandas: if unavailable, define a dummy class
try:
    from pandas import DataFrame

    PANDAS_INSTALLED = True
except ImportError:
    class DataFrame():  # pylint: disable=R0903
        """dummy for pandas.DataFrame"""


    PANDAS_INSTALLED = False

__all__ = []
