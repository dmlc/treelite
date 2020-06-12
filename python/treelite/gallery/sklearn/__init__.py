# coding: utf-8
"""Converter to ingest scikit-learn models into Treelite"""
import warnings

from ...sklearn import import_model as _import_model

warnings.warn(('treelite.gallery.sklearn has been moved to treelite.sklearn. ' +
               'treelite.gallery.sklearn will be removed in version 1.1.'),
              FutureWarning)


def import_model(sklearn_model):  # pylint: disable=C0111
    return _import_model(sklearn_model)
