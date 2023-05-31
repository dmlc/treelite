# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3
"""Utilities to (de)serialize Treelite objects via Python protocol"""

from cpython.object cimport PyObject
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

import ctypes
from typing import Dict, List, Tuple, Union

import numpy as np

import treelite


cdef extern from "treelite/c_api.h":
    ctypedef void* ModelHandle


cdef extern from "treelite/pybuffer_frame.h" namespace "treelite":
    cdef struct PyBufferFrame:
        void* buf
        char* format
        size_t itemsize
        size_t nitem

cdef extern from "treelite/tree.h" namespace "treelite":
    cdef cppclass Model:
        vector[PyBufferFrame] GetPyBuffer() except +
        @staticmethod
        unique_ptr[Model] CreateFromPyBuffer(vector[PyBufferFrame]) except +


cdef extern from "Python.h":
    Py_buffer* PyMemoryView_GET_BUFFER(PyObject* mview)


cdef class PyBufferFrameWrapper:
    cdef PyBufferFrame _handle
    cdef Py_ssize_t shape[1]
    cdef Py_ssize_t strides[1]

    def __cinit__(self):
        pass

    def __dealloc__(self):
        pass

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = self._handle.itemsize

        self.shape[0] = self._handle.nitem
        self.strides[0] = itemsize

        buffer.buf = self._handle.buf
        buffer.format = self._handle.format
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self._handle.nitem * itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass


cdef PyBufferFrameWrapper MakePyBufferFrameWrapper(PyBufferFrame handle):
    cdef PyBufferFrameWrapper wrapper = PyBufferFrameWrapper()
    wrapper._handle = handle
    return wrapper


cdef list _get_frames(ModelHandle model):
    return [memoryview(MakePyBufferFrameWrapper(v))
            for v in (<Model*>model).GetPyBuffer()]


cdef ModelHandle _init_from_frames(vector[PyBufferFrame] frames) except *:
    return <ModelHandle>Model.CreateFromPyBuffer(frames).release()


def get_frames(model: uintptr_t) -> List[memoryview]:
    return _get_frames(<ModelHandle> model)


def init_from_frames(frames: List[np.ndarray],
                     format_str: List[str], itemsize: List[int]) -> uintptr_t:
    cdef vector[PyBufferFrame] cpp_frames
    cdef Py_buffer* buf
    cdef PyBufferFrame cpp_frame
    format_bytes = [s.encode('utf-8') for s in format_str]
    for i, frame in enumerate(frames):
        x = memoryview(frame)
        buf = PyMemoryView_GET_BUFFER(<PyObject*>x)
        cpp_frame.buf = buf.buf
        cpp_frame.format = format_bytes[i]
        cpp_frame.itemsize = itemsize[i]
        cpp_frame.nitem = buf.len // itemsize[i]
        cpp_frames.push_back(cpp_frame)
    return <uintptr_t> _init_from_frames(cpp_frames)


def _treelite_serialize(
    model: uintptr_t
) -> Dict[str, Union[List[str], List[np.ndarray]]]:
    frames = get_frames(model)
    header = {'format_str': [x.format for x in frames],
              'itemsize': [x.itemsize for x in frames]}
    return {'header': header, 'frames': [np.asarray(x) for x in frames]}


def treelite_serialize(
    model: treelite.Model
) -> Dict[str, Union[List[str], List[np.ndarray]]]:
    return _treelite_serialize(model.handle.value)


def _treelite_deserialize(
    payload: Dict[str, Union[List[str], List[bytes]]]
) -> uintptr_t:
    header, frames = payload['header'], payload['frames']
    return init_from_frames(frames, header['format_str'], header['itemsize'])


def treelite_deserialize(
    payload: Dict[str, Union[List[str], List[bytes]]]
) -> treelite.Model:
    return treelite.Model(handle=ctypes.c_void_p(_treelite_deserialize(payload)))
