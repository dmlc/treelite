"""Treelite module"""
import pathlib

from . import frontend, gtil, model_builder, sklearn
from .core import TreeliteError
from .model import Model

VERSION_FILE = pathlib.Path(__file__).parent / "VERSION"
with open(VERSION_FILE, "r", encoding="UTF-8") as _f:
    __version__ = _f.read().strip()

__all__ = [
    "Model",
    "frontend",
    "gtil",
    "sklearn",
    "model_builder",
    "TreeliteError",
    "__version__",
]
