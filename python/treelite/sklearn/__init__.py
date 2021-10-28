# coding: utf-8
"""Converter to ingest scikit-learn models into Treelite"""

import ctypes

from ..util import TreeliteError
from ..frontend import Model
from .importer import import_model
from .common import SKLConverterBase
from .gbm_regressor import SKLGBMRegressorMixin
from .gbm_classifier import SKLGBMClassifierMixin
from .gbm_multi_classifier import SKLGBMMultiClassifierMixin
from .rf_regressor import SKLRFRegressorMixin
from .rf_classifier import SKLRFClassifierMixin
from .rf_multi_classifier import SKLRFMultiClassifierMixin


def import_model_with_model_builder(sklearn_model):
    """
    Load a tree ensemble model from a scikit-learn model object using the model builder API.

    .. note:: Use ``import_model`` for production use

        This function exists to demonstrate the use of the model builder API and is slow with
        large models. For production, please use :py:func:`~treelite.sklearn.import_model`
        which is significantly faster.

    Parameters
    ----------
    sklearn_model : object of type \
                    :py:class:`~sklearn.ensemble.RandomForestRegressor` / \
                    :py:class:`~sklearn.ensemble.RandomForestClassifier` / \
                    :py:class:`~sklearn.ensemble.ExtraTreesRegressor` / \
                    :py:class:`~sklearn.ensemble.ExtraTreesClassifier` / \
                    :py:class:`~sklearn.ensemble.GradientBoostingRegressor` / \
                    :py:class:`~sklearn.ensemble.GradientBoostingClassifier`
        Python handle to scikit-learn model

    Returns
    -------
    model : :py:class:`~treelite.Model` object
        loaded model

    Example
    -------

    .. code-block:: python
      :emphasize-lines: 8

      import sklearn.datasets
      import sklearn.ensemble
      X, y = sklearn.datasets.load_boston(return_X_y=True)
      clf = sklearn.ensemble.RandomForestRegressor(n_estimators=10)
      clf.fit(X, y)

      import treelite.sklearn
      model = treelite.sklearn.import_model_with_model_builder(clf)
    """
    try:
        from sklearn.ensemble import RandomForestRegressor as RandomForestR
        from sklearn.ensemble import RandomForestClassifier as RandomForestC
        from sklearn.ensemble import ExtraTreesRegressor as ExtraTreesR
        from sklearn.ensemble import ExtraTreesClassifier as ExtraTreesC
        from sklearn.ensemble import GradientBoostingRegressor as GradientBoostingR
        from sklearn.ensemble import GradientBoostingClassifier as GradientBoostingC
    except ImportError as e:
        raise TreeliteError('This function requires scikit-learn package') from e

    if isinstance(sklearn_model, (RandomForestR, ExtraTreesR)):
        return SKLRFRegressorConverter.process_model(sklearn_model)
    if isinstance(sklearn_model, (RandomForestC, ExtraTreesC)):
        if sklearn_model.n_classes_ == 2:
            return SKLRFClassifierConverter.process_model(sklearn_model)
        if sklearn_model.n_classes_ > 2:
            return SKLRFMultiClassifierConverter.process_model(sklearn_model)
        raise TreeliteError('n_classes_ must be at least 2')
    if isinstance(sklearn_model, GradientBoostingR):
        return SKLGBMRegressorConverter.process_model(sklearn_model)
    if isinstance(sklearn_model, GradientBoostingC):
        if sklearn_model.n_classes_ == 2:
            return SKLGBMClassifierConverter.process_model(sklearn_model)
        if sklearn_model.n_classes_ > 2:
            return SKLGBMMultiClassifierConverter.process_model(sklearn_model)
        raise TreeliteError('n_classes_ must be at least 2')
    raise TreeliteError(f'Unsupported model type {sklearn_model.__class__.__name__}: currently ' +
                        'random forests, extremely randomized trees, and gradient boosted trees ' +
                        'are supported')


class SKLGBMRegressorConverter(SKLGBMRegressorMixin, SKLConverterBase):  # pylint: disable=C0111
    pass


class SKLGBMClassifierConverter(SKLGBMClassifierMixin, SKLConverterBase):  # pylint: disable=C0111
    pass


class SKLGBMMultiClassifierConverter(SKLGBMMultiClassifierMixin, SKLConverterBase):
    # pylint: disable=C0111
    pass


class SKLRFRegressorConverter(SKLRFRegressorMixin, SKLConverterBase):  # pylint: disable=C0111
    pass


class SKLRFClassifierConverter(SKLRFClassifierMixin, SKLConverterBase):  # pylint: disable=C0111
    pass


class SKLRFMultiClassifierConverter(SKLRFMultiClassifierMixin, SKLConverterBase):
    # pylint: disable=C0111
    pass


__all__ = ['import_model_with_model_builder', 'import_model']
