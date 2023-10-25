"""Treelite Model class"""

from __future__ import annotations

import ctypes
import pathlib
import warnings
from typing import Any, List, Optional, Union

from . import compat
from .core import _LIB, _check_call
from .util import c_array, c_str, py_str


class Model:
    """
    Decision tree ensemble model

    Parameters
    ----------
    handle :
        Handle to C++ object
    """

    def __init__(self, *, handle: Optional[Any] = None):
        self._handle = handle

    def __del__(self):
        if self.handle is not None:
            _check_call(_LIB.TreeliteFreeModel(self._handle))
            self._handle = None

    @property
    def handle(self):
        """Access the handle to the associated C++ object"""
        return self._handle

    @property
    def num_tree(self) -> int:
        """Number of decision trees in the model"""
        if self.handle is None:
            raise AttributeError("Model not loaded yet")
        out = ctypes.c_size_t()
        _check_call(_LIB.TreeliteQueryNumTree(self.handle, ctypes.byref(out)))
        return out.value

    @property
    def num_feature(self) -> int:
        """Number of features used in the model"""
        if self.handle is None:
            raise AttributeError("Model not loaded yet")
        out = ctypes.c_size_t()
        _check_call(_LIB.TreeliteQueryNumFeature(self.handle, ctypes.byref(out)))
        return out.value

    @property
    def input_type(self) -> str:
        """Input type"""
        if self.handle is None:
            raise AttributeError("Model not loaded yet")
        out = ctypes.c_char_p()
        _check_call(_LIB.TreeliteGetInputType(self.handle, ctypes.byref(out)))
        return py_str(out.value)

    @property
    def output_type(self) -> str:
        """Output type"""
        if self.handle is None:
            raise AttributeError("Model not loaded yet")
        out = ctypes.c_char_p()
        _check_call(_LIB.TreeliteGetOutputType(self.handle, ctypes.byref(out)))
        return py_str(out.value)

    @classmethod
    def concatenate(cls, model_objs: List[Model]) -> Model:
        """
        Concatenate multiple model objects into a single model object by copying
        all member trees into the destination model object.

        Parameters
        ----------
        model_objs :
            List of :py:class:`Model` objects
        Returns
        -------
        model : :py:class:`Model` object
            Concatenated model
        Example
        -------
        .. code-block:: python
           concatenated_model = Model.concatenate([model1, model2, model3])
        """
        concatenated_model_handle = ctypes.c_void_p()
        model_obj_handles = []
        for i, obj in enumerate(model_objs):
            if obj.handle is None:
                raise RuntimeError(f"Model at index {i} not loaded yet")
            model_obj_handles.append(obj.handle)
        _check_call(
            _LIB.TreeliteConcatenateModelObjects(
                c_array(ctypes.c_void_p, model_obj_handles),
                len(model_obj_handles),
                ctypes.byref(concatenated_model_handle),
            )
        )
        return Model(handle=concatenated_model_handle)

    @classmethod
    def load(
        cls, filename: str, model_format: str, allow_unknown_field: bool = False
    ) -> Model:
        """
        Deprecated. Please use :py:meth:`~treelite.frontend.load_xgboost_model` instead.
        Load a tree ensemble model from a file.

        Parameters
        ----------
        filename :
            Path to model file
        model_format :
            Model file format. Must be "xgboost", "xgboost_json", or "lightgbm"
        allow_unknown_field:
            Whether to allow extra fields with unrecognized keys. This flag is only
            applicable if model_format="xgboost_json"

        Returns
        -------
        model :
            Loaded model
        """
        model_format = model_format.lower()

        def deprecation_warning(alt: str) -> None:
            warnings.warn(
                (
                    "treelite.Model.load() is deprecated. "
                    f"Use treelite.frontend.{alt}() instead."
                ),
                FutureWarning,
            )

        if model_format == "xgboost":
            deprecation_warning("load_xgboost_model_legacy_binary")
            return Model(handle=compat.load_xgboost_model_legacy_binary(filename))
        if model_format == "xgboost_json":
            deprecation_warning("load_xgboost_model")
            return Model(
                handle=compat.load_xgboost_model(
                    filename, allow_unknown_field=allow_unknown_field
                )
            )
        if model_format == "lightgbm":
            deprecation_warning("load_lightgbm_model")
            return Model(handle=compat.load_lightgbm_model(filename))
        raise ValueError(f"Unknown model format {model_format}")

    @classmethod
    def from_xgboost(cls, booster: Any) -> Model:
        """
        Deprecated. Please use :py:meth:`~treelite.frontend.from_xgboost` instead.
        Load a tree ensemble model from an XGBoost Booster object.

        Parameters
        ----------
        booster : Object of type :py:class:`xgboost.Booster`
            Python handle to XGBoost model

        Returns
        -------
        model :
            Loaded model
        """
        warnings.warn(
            (
                "treelite.Model.from_xgboost() is deprecated. "
                "Use treelite.frontend.from_xgboost() instead."
            ),
            FutureWarning,
        )
        return Model(handle=compat.from_xgboost(booster))

    @classmethod
    def from_xgboost_json(
        cls,
        model_json_str: Union[bytes, bytearray, str],
        *,
        allow_unknown_field: bool = False,
    ) -> Model:
        """
        Deprecated. Please use :py:meth:`~treelite.frontend.from_xgboost_json` instead.
        Load a tree ensemble model from a string containing XGBoost JSON.

        Parameters
        ----------
        model_json_str :
            A string specifying an XGBoost model in the XGBoost JSON format
        allow_unknown_field:
            Whether to allow extra fields with unrecognized keys

        Returns
        -------
        model
            Loaded model
        """
        warnings.warn(
            (
                "treelite.Model.from_xgboost_json() is deprecated. "
                "Use treelite.frontend.from_xgboost_json() instead."
            ),
            FutureWarning,
        )
        return Model(
            handle=compat.from_xgboost_json(
                model_json_str, allow_unknown_field=allow_unknown_field
            )
        )

    @classmethod
    def from_lightgbm(cls, booster):
        """
        Deprecated. Please use :py:meth:`~treelite.frontend.from_lightgbm` instead.
        Load a tree ensemble model from a LightGBM Booster object.

        Parameters
        ----------
        booster : object of type :py:class:`lightgbm.Booster`
            Python handle to LightGBM model

        Returns
        -------
        model : :py:class:`Model` object
            loaded model
        """
        warnings.warn(
            (
                "treelite.Model.from_lightgbm() is deprecated. "
                "Use treelite.frontend.from_lightgbm() instead."
            ),
            FutureWarning,
        )
        return Model(handle=compat.from_lightgbm(booster))

    def dump_as_json(self, *, pretty_print: bool = True) -> str:
        """
        Dump the model as a JSON string. This is useful for inspecting details of the tree
        ensemble model.

        Parameters
        ----------
        pretty_print :
            Whether to pretty-print the JSON string, set this to False to make the string compact

        Returns
        -------
        json_str :
            JSON string representing the model
        """
        json_str = ctypes.c_char_p()
        _check_call(
            _LIB.TreeliteDumpAsJSON(
                self.handle,
                ctypes.c_int(1 if pretty_print else 0),
                ctypes.byref(json_str),
            )
        )
        return py_str(json_str.value)

    def serialize(self, filename: Union[str, pathlib.Path]) -> None:
        """
        Serialize (persist) the model to a checkpoint file in the disk, using a fast binary
        representation. To recover the model from the checkpoint, use :py:func:`deserialize`
        method.

        .. note:: Notes on forward and backward compatibility

            Please see :doc:`/serialization/index`.

        Parameters
        ----------
        filename :
            Path to checkpoint
        """
        filepath = pathlib.Path(filename).expanduser().resolve()
        _check_call(
            _LIB.TreeliteSerializeModelToFile(self.handle, c_str(str(filepath)))
        )

    def serialize_bytes(self) -> bytes:
        """
        Serialize (persist) the model to a byte sequence, using a fast binary representation.
        To recover the model from the byte sequence, use :py:func:`deserialize_bytes` method.

        .. note:: Notes on forward and backward compatibility

            Please see :doc:`/serialization/index`.
        """
        char_ptr_t = ctypes.POINTER(ctypes.c_char)
        out_bytes = char_ptr_t()
        out_bytes_len = ctypes.c_size_t()
        _check_call(
            _LIB.TreeliteSerializeModelToBytes(
                self.handle, ctypes.byref(out_bytes), ctypes.byref(out_bytes_len)
            )
        )
        return ctypes.string_at(out_bytes, out_bytes_len.value)

    @classmethod
    def deserialize(cls, filename: Union[str, pathlib.Path]) -> Model:
        """
        Deserialize (recover) the model from a checkpoint file in the disk. It is expected that
        the file was generated by a call to the :py:func:`serialize` method.

        .. note:: Notes on forward and backward compatibility

            Please see :doc:`/serialization/index`.

        Parameters
        ----------
        filename :
            Path to checkpoint

        Returns
        -------
        model :
            Recovered model
        """
        handle = ctypes.c_void_p()
        filepath = pathlib.Path(filename).expanduser().resolve()
        _check_call(
            _LIB.TreeliteDeserializeModelFromFile(
                c_str(str(filepath)), ctypes.byref(handle)
            )
        )
        return Model(handle=handle)

    @classmethod
    def deserialize_bytes(cls, model_bytes: bytes) -> Model:
        """
        Deserialize (recover) the model from a byte sequence. It is expected that
        the byte sequence was generated by a call to the :py:func:`serialize_bytes` method.

        .. note:: Notes on forward and backward compatibility

            Please see :doc:`/serialization/index`.

        Parameters
        ----------
        model_bytes : :py:class:`bytes <python:bytes>`
            Byte sequence representing the serialized model

        Returns
        -------
        model : :py:class:`Model` object
            Recovered model
        """
        handle = ctypes.c_void_p()
        model_bytes_len = len(model_bytes)
        buffer = ctypes.create_string_buffer(model_bytes, model_bytes_len)
        _check_call(
            _LIB.TreeliteDeserializeModelFromBytes(
                ctypes.pointer(buffer),
                ctypes.c_size_t(model_bytes_len),
                ctypes.byref(handle),
            )
        )
        return Model(handle=handle)
