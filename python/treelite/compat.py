"""Compatibility layer to enable API migration"""
import ctypes
import json
from typing import Any, Optional, Union

from packaging.version import parse as parse_version

from .core import _LIB, TreeliteError, _check_call
from .util import c_str


def load_xgboost_model_legacy_binary(filename: str) -> Any:
    """
    Load a tree ensemble model from XGBoost model, stored using
    the legacy binary format.

    TODO(hcho3): Move the implementation to treelite.frontend once
                 Model.load() is removed.
    """
    handle = ctypes.c_void_p()
    _check_call(
        _LIB.TreeliteLoadXGBoostModelLegacyBinary(
            c_str(filename), c_str("{}"), ctypes.byref(handle)
        )
    )
    return handle


def load_xgboost_model(filename: str, *, allow_unknown_field: bool) -> Any:
    """
    Load a tree ensemble model from XGBoost model, stored using the JSON format.

    TODO(hcho3): Move the implementation to treelite.frontend once
                 Model.load() is removed.
    """
    parser_config = {"allow_unknown_field": allow_unknown_field}
    parser_config_str = json.dumps(parser_config)
    handle = ctypes.c_void_p()
    _check_call(
        _LIB.TreeliteLoadXGBoostModel(
            c_str(filename), c_str(parser_config_str), ctypes.byref(handle)
        )
    )
    return handle


def load_lightgbm_model(filename: str) -> Any:
    """
    Load a tree ensemble model from a LightGBM model file

    TODO(hcho3): Move the implementation to treelite.frontend once
                 Model.load() is removed.
    """
    handle = ctypes.c_void_p()
    _check_call(
        _LIB.TreeliteLoadLightGBMModel(
            c_str(filename), c_str("{}"), ctypes.byref(handle)
        )
    )
    return handle


def from_xgboost_json(
    model_json_str: Union[str, bytes, bytearray],
    *,
    allow_unknown_field: Optional[bool] = False,
) -> Any:
    """
    Load a tree ensemble model from a string containing XGBoost JSON.

    TODO(hcho3): Move the implementation to treelite.frontend once
                 Model.from_xgboost_json() is removed.
    """
    parser_config = {"allow_unknown_field": allow_unknown_field}
    parser_config_str = json.dumps(parser_config)

    handle = ctypes.c_void_p()
    length = len(model_json_str)
    if isinstance(model_json_str, (bytes, bytearray)):
        json_buffer = ctypes.create_string_buffer(bytes(model_json_str), length)
        _check_call(
            _LIB.TreeliteLoadXGBoostModelFromString(
                json_buffer,
                ctypes.c_size_t(length),
                c_str(parser_config_str),
                ctypes.byref(handle),
            )
        )
    else:
        _check_call(
            _LIB.TreeliteLoadXGBoostModelFromString(
                c_str(model_json_str),
                ctypes.c_size_t(length),
                c_str(parser_config_str),
                ctypes.byref(handle),
            )
        )
    return handle


def from_xgboost(booster: Any) -> Any:
    """
    Load a tree ensemble model from an XGBoost Booster object.

    TODO(hcho3): Move the implementation to treelite.frontend once
                 Model.from_xgboost() is removed.
    """
    try:
        import xgboost
    except ImportError as e:
        raise TreeliteError(
            "xgboost module must be installed to read from "
            + "`xgboost.Booster` object"
        ) from e
    if not isinstance(booster, xgboost.Booster):
        raise ValueError("booster must be of type `xgboost.Booster`")
    xgb_version = parse_version(xgboost.__version__)
    if xgb_version > parse_version("1.5.2"):
        # For XGBoost version 1.6.0 and later, use save_raw() to export models as JSON string
        model_json_str = booster.save_raw(raw_format="json")
        return from_xgboost_json(model_json_str)
    if xgb_version >= parse_version("1.0.0"):
        # Prior to version 1.6.0, XGBoost offer a method to export models as JSON string
        # in-memory. So use __getstate__ instead.
        model_json_str = booster.__getstate__()["handle"]
        return from_xgboost_json(model_json_str)
    # If pre-1.0.0 version of XGBoost is used, use legacy serialization
    handle = ctypes.c_void_p()
    buffer = booster.save_raw()
    ptr = (ctypes.c_char * len(buffer)).from_buffer(buffer)
    length = ctypes.c_size_t(len(buffer))
    _check_call(
        _LIB.TreeliteLoadXGBoostModelLegacyBinaryFromMemoryBuffer(
            ptr, length, ctypes.byref(handle)
        )
    )
    return handle


def from_lightgbm(booster: Any) -> Any:
    """
    Load a tree ensemble model from a LightGBM Booster object

    TODO(hcho3): Move the implementation to treelite.frontend once
                 Model.from_lightgbm() is removed.
    """
    handle = ctypes.c_void_p()
    # Attempt to import lightgbm
    try:
        import lightgbm
    except ImportError as e:
        raise TreeliteError(
            "lightgbm module must be installed to read from `lightgbm.Booster` object"
        ) from e
    if not isinstance(booster, lightgbm.Booster):
        raise ValueError("booster must be of type `lightgbm.Booster`")
    model_str = booster.model_to_string()
    _check_call(
        _LIB.TreeliteLoadLightGBMModelFromString(
            c_str(model_str), c_str("{}"), ctypes.byref(handle)
        )
    )
    return handle
