# coding: utf-8
import os

from .predictor import Predictor, DMatrix
from .util import TreeliteRuntimeError

VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')
with open(VERSION_FILE) as f:
    __version__ = f.read().strip()

__all__ = ['Predictor', 'DMatrix', 'TreeliteRuntimeError']
