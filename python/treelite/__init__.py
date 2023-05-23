# coding: utf-8
"""
Treelite: a model compiler for decision tree ensembles
"""
import os

from . import gtil, sklearn
from .frontend import Model, ModelBuilder
from .util import TreeliteError

VERSION_FILE = os.path.join(os.path.dirname(__file__), "VERSION")
with open(VERSION_FILE, "r", encoding="UTF-8") as _f:
    __version__ = _f.read().strip()

__all__ = ["Model", "ModelBuilder", "sklearn", "gtil", "TreeliteError"]
