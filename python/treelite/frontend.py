"""Functions to load and build model objects"""
from __future__ import annotations

import pathlib
from typing import Any, Union

from . import compat
from .model import Model


def load_xgboost_model_legacy_binary(filename: Union[str, pathlib.Path]) -> Model:
    """
    Load a tree ensemble model from XGBoost model, stored using
    the legacy binary format. Note: new XGBoost models should
    be stored in the JSON format, to take advantage of the
    latest functionalities of XGBoost.

    Parameters
    ----------
    filename :
        Path to model file

    Returns
    -------
    model : :py:class:`Model`
        Loaded model

    Example
    -------

    .. code-block:: python

       xgb_model = treelite.frontend.load_xgboost_model_legacy_binary(
           "xgboost_model.model")
    """
    return Model(handle=compat.load_xgboost_model_legacy_binary(str(filename)))


def load_xgboost_model(
    filename: Union[str, pathlib.Path], *, allow_unknown_field: bool = False
) -> Model:
    """
    Load a tree ensemble model from XGBoost model, stored using the JSON format.

    Parameters
    ----------
    filename :
        Path to model file
    allow_unknown_field:
        Whether to allow extra fields with unrecognized keys

    Returns
    -------
    model : :py:class:`Model`
        Loaded model

    Example
    -------

    .. code-block:: python

       xgb_model = treelite.frontend.load_xgboost_model("xgboost_model.json")
    """
    return Model(
        handle=compat.load_xgboost_model(
            str(filename), allow_unknown_field=allow_unknown_field
        )
    )


def load_lightgbm_model(filename: Union[str, pathlib.Path]) -> Model:
    """
    Load a tree ensemble model from a LightGBM model file.

    Parameters
    ----------
    filename :
        Path to model file

    Returns
    -------
    model : :py:class:`Model`
        Loaded model

    Example
    -------

    .. code-block:: python

       lgb_model = treelite.frontend.load_lightgbm_model("lightgbm_model.txt")
    """
    return Model(handle=compat.load_lightgbm_model(str(filename)))


def from_xgboost(booster: Any) -> Model:
    """
    Load a tree ensemble model from an XGBoost Booster object

    Parameters
    ----------
    booster : Object of type :py:class:`xgboost.Booster`
        Python handle to XGBoost model

    Returns
    -------
    model : :py:class:`Model`
        Loaded model
    """
    return Model(handle=compat.from_xgboost(booster))


def from_xgboost_json(
    model_json_str: Union[bytes, bytearray, str],
    *,
    allow_unknown_field: bool = False,
) -> Model:
    """
    Load a tree ensemble model from a string containing XGBoost JSON

    Parameters
    ----------
    model_json_str :
        A string specifying an XGBoost model in the XGBoost JSON format
    allow_unknown_field:
        Whether to allow extra fields with unrecognized keys

    Returns
    -------
    model: :py:class:`Model`
        Loaded model
    """
    return Model(
        handle=compat.from_xgboost_json(
            model_json_str, allow_unknown_field=allow_unknown_field
        )
    )


def from_lightgbm(booster: Any) -> Model:
    """
    Load a tree ensemble model from a LightGBM Booster object

    Parameters
    ----------
    booster : object of type :py:class:`lightgbm.Booster`
        Python handle to LightGBM model

    Returns
    -------
    model : :py:class:`Model`
        Loaded model
    """
    return Model(handle=compat.from_lightgbm(booster))


__all__ = [
    "load_xgboost_model_legacy_binary",
    "load_xgboost_model",
    "load_lightgbm_model",
    "from_xgboost",
    "from_xgboost_json",
    "from_lightgbm",
]
