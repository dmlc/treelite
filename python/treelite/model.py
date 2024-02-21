"""Treelite Model class"""

from __future__ import annotations

import ctypes
import pathlib
import platform
import warnings
from typing import Any, List, Optional, Union

import numpy as np

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
        model : :py:class:`Model`
            Concatenated model

        Example
        -------
        .. code-block:: python

           concatenated_model = treelite.Model.concatenate([model1, model2, model3])
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
        Deprecated; please use :py:meth:`~treelite.frontend.load_xgboost_model` instead.
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
        model : :py:class:`Model`
            Loaded model
        """
        model_format = model_format.lower()

        def deprecation_warning(alt: str):
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
        Deprecated; please use :py:meth:`~treelite.frontend.from_xgboost` instead.
        Load a tree ensemble model from an XGBoost Booster object.

        Parameters
        ----------
        booster : Object of type :py:class:`xgboost.Booster`
            Python handle to XGBoost model

        Returns
        -------
        model : :py:class:`Model`
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
        Deprecated; please use :py:meth:`~treelite.frontend.from_xgboost_json` instead.
        Load a tree ensemble model from a string containing XGBoost JSON.

        Parameters
        ----------
        model_json_str :
            A string specifying an XGBoost model in the XGBoost JSON format
        allow_unknown_field:
            Whether to allow extra fields with unrecognized keys

        Returns
        -------
        model : :py:class:`Model`
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
        Deprecated; please use :py:meth:`~treelite.frontend.from_lightgbm` instead.
        Load a tree ensemble model from a LightGBM Booster object.

        Parameters
        ----------
        booster : object of type :py:class:`lightgbm.Booster`
            Python handle to LightGBM model

        Returns
        -------
        model : :py:class:`Model`
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
        json_str : str
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

    def get_header_accessor(self) -> HeaderAccessor:
        """
        Obtain accessor for fields in the header.
        See :ref:`field_accessors` for more details.
        """
        return HeaderAccessor(self)

    def get_tree_accessor(self, tree_id: int) -> TreeAccessor:
        """
        Obtain accessor for fields in a tree.
        See :ref:`field_accessors` for more details.

        Parameters
        ----------
        tree_id:
            ID of the tree
        """
        return TreeAccessor(self, tree_id=tree_id)

    def serialize(self, filename: Union[str, pathlib.Path]):
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
        model : :py:class:`Model`
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
        model_bytes :
            Byte sequence representing the serialized model

        Returns
        -------
        model : :py:class:`Model`
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


class _TreelitePyBufferFrame(ctypes.Structure):  # pylint: disable=R0903
    """Abridged buffer structure used by Treelite"""

    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("format", ctypes.c_char_p),
        ("itemsize", ctypes.c_size_t),
        ("nitem", ctypes.c_size_t),
    ]


class _PyBuffer(ctypes.Structure):  # pylint: disable=R0902,R0903
    """The full Python buffer structure as defined by PEP 3118"""

    _fields_ = (
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.py_object),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
        ("internal", ctypes.c_void_p),
    )


if platform.python_implementation() == "CPython":
    ctypes.pythonapi.PyMemoryView_FromBuffer.argtypes = [ctypes.POINTER(_PyBuffer)]
    ctypes.pythonapi.PyMemoryView_FromBuffer.restype = ctypes.py_object
    ctypes.pythonapi.PyObject_GetBuffer.argtypes = [
        ctypes.py_object,
        ctypes.POINTER(_PyBuffer),
        ctypes.c_int,
    ]
    ctypes.pythonapi.PyObject_GetBuffer.restype = ctypes.c_int


def _pybuffer2numpy(frame: _TreelitePyBufferFrame) -> np.ndarray:
    if platform.python_implementation() != "CPython":
        raise NotImplementedError("_pybuffer2numpy() not supported on PyPy")
    if not frame.buf:
        if frame.format == b"=l":
            dtype = "int32"
        elif frame.format == b"=Q":
            dtype = "uint64"
        elif frame.format == b"=L":
            dtype = "uint32"
        elif frame.format == b"=B":
            dtype = "uint8"
        elif frame.format == b"=f":
            dtype = "float32"
        elif frame.format == b"=d":
            dtype = "float64"
        else:
            raise RuntimeError(
                f"Unrecognized format string: {frame.format.decode('utf-8')}"
            )
        return np.array([], dtype=dtype)
    py_buf = _PyBuffer()
    py_buf.buf = frame.buf
    py_buf.obj = ctypes.py_object(frame)
    py_buf.len = frame.nitem * frame.itemsize
    py_buf.itemsize = frame.itemsize
    py_buf.readonly = 0
    py_buf.ndim = 1
    py_buf.format = frame.format
    py_buf.shape = (ctypes.c_ssize_t * 1)(frame.nitem)
    py_buf.strides = (ctypes.c_ssize_t * 1)(frame.itemsize)
    py_buf.suboffsets = None
    py_buf.internal = None

    view: memoryview = ctypes.pythonapi.PyMemoryView_FromBuffer(ctypes.byref(py_buf))
    return np.asarray(view)


def _numpy2pybuffer(array: np.ndarray) -> _TreelitePyBufferFrame:
    if platform.python_implementation() != "CPython":
        raise NotImplementedError("_numpy2pybuffer() not supported on PyPy")
    if len(array.shape) != 1:
        raise ValueError("Cannot handle NumPy array that has more than 1 dimension")
    view: memoryview = array.data
    buffer = _PyBuffer()
    if (
        ctypes.pythonapi.PyObject_GetBuffer(
            ctypes.py_object(view),
            ctypes.byref(buffer),
            ctypes.c_int(0),  # PyBUF_SIMPLE
        )
        != 0
    ):
        raise RuntimeError("Call to PyObject_GetBuffer() failed")
    frame = _TreelitePyBufferFrame()
    frame.buf = buffer.buf
    frame.format = buffer.format
    frame.itemsize = buffer.itemsize
    frame.nitem = array.shape[0]
    return frame


class HeaderAccessor:
    """
    Accessor for fields in the header

    Parameters
    ----------
    model:
        The model object
    """

    def __init__(self, model: Model):
        self._model = model

    def get_field(self, name: str) -> Union[np.ndarray, str]:
        """
        Get a field

        Parameters
        ----------
        name:
            Name of the field. Consult :doc:`the model spec </serialization/v4>`
            for the list of fields.

        Returns
        -------
        field: :py:class:`numpy.ndarray` or :py:class:`str`
            Value in the field
            (``str`` for a string field, ``np.ndarray`` for other fields)
        """
        if platform.python_implementation() != "CPython":
            raise NotImplementedError("get_field() not supported on PyPy")
        obj = _TreelitePyBufferFrame()
        _check_call(
            _LIB.TreeliteGetHeaderField(
                self._model.handle,
                c_str(name),
                ctypes.byref(obj),
            )
        )
        array = _pybuffer2numpy(obj)
        if array.dtype == "S1":
            return array.tobytes().decode("utf-8")
        return array

    def set_field(self, name: str, value: Union[np.ndarray, str]):
        """
        Set a field

        Parameters
        ----------
        name:
            Name of the field. Consult :doc:`the model spec </serialization/v4>`
            for the list of fields.
        value:
            New value for the field
            (``str`` for a string field, ``np.ndarray`` for other fields)
        """
        if platform.python_implementation() != "CPython":
            raise NotImplementedError("set_field() not supported on PyPy")
        if isinstance(value, str):
            value = np.frombuffer(value.encode("utf-8"), dtype="S1")
        _check_call(
            _LIB.TreeliteSetHeaderField(
                self._model.handle,
                c_str(name),
                _numpy2pybuffer(value),
            )
        )


class TreeAccessor:
    """
    Accessor for fields in a tree

    Parameters
    ----------
    model:
        The model object
    tree_id:
        ID of the tree
    """

    def __init__(self, model: Model, *, tree_id: int):
        self._model = model
        self._tree_id = tree_id

    def get_field(self, name: str) -> np.ndarray:
        """
        Get a field

        Parameters
        ----------
        name:
            Name of the field. Consult :doc:`the model spec </serialization/v4>`
            for the list of fields.

        Returns
        -------
        field: :py:class:`numpy.ndarray`
            Value in the field
        """
        obj = _TreelitePyBufferFrame()
        _check_call(
            _LIB.TreeliteGetTreeField(
                self._model.handle,
                ctypes.c_uint64(self._tree_id),
                c_str(name),
                ctypes.byref(obj),
            )
        )
        return _pybuffer2numpy(obj)

    def set_field(self, name: str, value: np.ndarray):
        """
        Set a field

        Parameters
        ----------
        name:
            Name of the field. Consult :doc:`the model spec </serialization/v4>`
            for the list of fields.
        value:
            New value for the field
        """
        _check_call(
            _LIB.TreeliteSetTreeField(
                self._model.handle,
                ctypes.c_uint64(self._tree_id),
                c_str(name),
                _numpy2pybuffer(value),
            )
        )


__all__ = ["Model", "HeaderAccessor", "TreeAccessor"]
