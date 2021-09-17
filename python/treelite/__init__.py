# coding: utf-8
"""
Treelite: a model compiler for decision tree ensembles
"""
import os

from .frontend import Model, ModelBuilder
from .annotator import Annotator
from .contrib import create_shared, generate_makefile, generate_cmakelists
from .util import TreeliteError
from . import sklearn
from . import gtil

VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')
with open(VERSION_FILE, 'r', encoding='UTF-8') as _f:
    __version__ = _f.read().strip()

__all__ = ['Model', 'ModelBuilder', 'Annotator', 'create_shared', 'generate_makefile',
           'generate_cmakelists', 'sklearn', 'gtil', 'TreeliteError']
