"""Functions to load and build model objects"""

from __future__ import annotations

import ctypes
import json
import pathlib
from typing import Any, Union

from . import compat
from .core import _LIB, _check_call
from .model import Model
from .util import c_str, py_str


def _normalize_path(filename: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Fully expand a path and convert it to an absolute path
    """
    path = pathlib.Path(filename)
    return path.expanduser().resolve()


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
    path = _normalize_path(filename)
    return Model(handle=compat.load_xgboost_model_legacy_binary(str(path)))


def load_xgboost_model(
    filename: Union[str, pathlib.Path],
    *,
    format_choice: str = "use_suffix",
    allow_unknown_field: bool = False,
) -> Model:
    """
    Load a tree ensemble model from XGBoost model, stored using JSON or UBJSON format.

    Parameters
    ----------
    filename :
        Path to model file
    format_choice :
        Method to select the model format

        * ``use_suffix`` (default): Use the suffix of the file name (also known as file
          extension) to detect the format. Files whose names end with ``.json`` will be
          parsed as JSON; all other files will be parsed as UBJSON.
        * ``inspect``: Inspect the first few bytes of the file to heuristically determine
          whether the file is JSON or UBJSON.
        * ``ubjson``: Parse the file as UBJSON.
        * ``json``: Parse the file as JSON.
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

    parser_config = {"allow_unknown_field": allow_unknown_field}
    parser_config_str = json.dumps(parser_config)

    path = _normalize_path(filename)

    def parse_as_json() -> Model:
        return Model(
            handle=compat.load_xgboost_model(
                str(path), allow_unknown_field=allow_unknown_field
            )
        )

    def parse_as_ubjson() -> Model:
        handle = ctypes.c_void_p()
        _check_call(
            _LIB.TreeliteLoadXGBoostModelUBJSON(
                c_str(str(path)), c_str(parser_config_str), ctypes.byref(handle)
            )
        )
        return Model(handle=handle)

    if format_choice == "use_suffix":
        if path.name.endswith(".json"):
            return parse_as_json()
        # File name not ending with .json will be parsed as UBJSON.
        return parse_as_ubjson()

    if format_choice == "inspect":
        detected_format = _detect_xgboost_format(path)
        if detected_format == "json":
            return parse_as_json()
        if detected_format == "ubjson":
            return parse_as_ubjson()
        raise ValueError(
            "Could not detect whether the given XGBoost model is JSON or UBJSON. "
            "Please explicitly set format_choice='json' or format_choice='ubjson'"
        )
    if format_choice == "ubjson":
        return parse_as_ubjson()

    if format_choice == "json":
        return parse_as_json()

    raise ValueError(f"Unknown format_choice argument: {format_choice}")


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
    path = _normalize_path(filename)
    return Model(handle=compat.load_lightgbm_model(str(path)))


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


def from_xgboost_ubjson(
    model_ubjson_str: Union[bytes, bytearray],
    *,
    allow_unknown_field: bool = False,
) -> Model:
    """
    Load a XGBoost model from a byte sequence containing UBJSON

    Parameters
    ----------
    model_ubjson_str :
        A byte sequence specifying an XGBoost model in the UBJSON format
    allow_unknown_field:
        Whether to allow extra fields with unrecognized keys

    Returns
    -------
    model: :py:class:`Model`
        Loaded model
    """
    return Model(
        handle=compat.from_xgboost_ubjson(
            model_ubjson_str, allow_unknown_field=allow_unknown_field
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


def _detect_xgboost_format(path: pathlib.Path) -> str:
    """
    Inspect the first few bytes of an XGBoost model and heuristically determine whether
    it's using the JSON or UBJSON format.

    Parameters
    ----------
    path :
        Path to model file (assumed to be fully resolved with ``path.expanduser().resolve()``)

    Returns
    -------
    "json", "ubjson", or "unknown"
    """
    detected_format = ctypes.c_char_p()
    _check_call(
        _LIB.TreeliteDetectXGBoostFormat(
            c_str(str(path)),
            ctypes.byref(detected_format),
        )
    )
    return py_str(detected_format.value)


__all__ = [
    "load_xgboost_model_legacy_binary",
    "load_xgboost_model",
    "load_lightgbm_model",
    "from_xgboost",
    "from_xgboost_json",
    "from_xgboost_ubjson",
    "from_lightgbm",
]
