# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=protected-access
# pylint: disable=import-error, no-name-in-module, undefined-variable

"""DLPack API of MXNet."""

import ctypes
from .base import _LIB, c_str, check_call, NDArrayHandle

DLPackHandle = ctypes.c_void_p

PyCapsuleDestructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_c_str_dltensor = c_str('dltensor')
_c_str_used_dltensor = c_str('used_dltensor')

def _dlpack_deleter(pycapsule):
    pycapsule = ctypes.c_void_p(pycapsule)
    if ctypes.pythonapi.PyCapsule_IsValid(pycapsule, _c_str_dltensor):
        ptr = ctypes.c_void_p(
            ctypes.pythonapi.PyCapsule_GetPointer(pycapsule, _c_str_dltensor))
        check_call(_LIB.MXNDArrayCallDLPackDeleter(ptr))

_c_dlpack_deleter = PyCapsuleDestructor(_dlpack_deleter)

class DLContext(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int),
                ("device_id", ctypes.c_int)]

class DLDataType(ctypes.Structure):
    _fields_ = [("type_code", ctypes.c_uint8),
                ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16)]
    TYPE_MAP = {
        "int16": (0, 16, 1),
        "int32": (0, 32, 1),
        "int64": (0, 64, 1),
        "bool": (1, 1, 1),
        "uint8": (1, 8, 1),
        "uint16": (1, 16, 1),
        "uint32": (1, 32, 1),
        "uint64": (1, 64, 1),
        'float16': (2, 16, 1),
        "float32": (2, 32, 1),
        "float64": (2, 64, 1),
    }


class DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p),
                ("ctx", DLContext),
                ("ndim", ctypes.c_int),
                ("dtype", DLDataType),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64)]

class DLManagedTensor(ctypes.Structure):
    pass


DeleterFunc = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))


DLManagedTensor._fields_ = [("dl_tensor", DLTensor),           # pylint: disable=protected-access
                            ("manager_ctx", ctypes.c_void_p),
                            ("deleter", DeleterFunc)]

@DeleterFunc
def dl_managed_tensor_deleter(dl_managed_tensor_handle):
    void_p = dl_managed_tensor_handle.contents.manager_ctx
    pyobj = ctypes.cast(void_p, ctypes.py_object)
    ctypes.pythonapi.Py_DecRef(pyobj)

def ndarray_from_dlpack(array_cls):
    """Returns a function that returns specified array_cls from dlpack.

    Returns
    -------
    fn : dlpack -> array_cls
    """
    def from_dlpack(dlpack):
        handle = NDArrayHandle()
        dlpack = ctypes.py_object(dlpack)
        assert ctypes.pythonapi.PyCapsule_IsValid(dlpack, _c_str_dltensor), ValueError(
            'Invalid DLPack Tensor. DLTensor capsules can be consumed only once.')
        dlpack_handle = ctypes.c_void_p(ctypes.pythonapi.PyCapsule_GetPointer(dlpack, _c_str_dltensor))
        check_call(_LIB.MXNDArrayFromDLPack(dlpack_handle, False, ctypes.byref(handle)))
        # Rename PyCapsule (DLPack)
        ctypes.pythonapi.PyCapsule_SetName(dlpack, _c_str_used_dltensor)
        # delete the deleter of the old dlpack
        ctypes.pythonapi.PyCapsule_SetDestructor(dlpack, None)
        return array_cls(handle=handle)
    return from_dlpack


def ndarray_to_dlpack_for_read():
    """Returns a function that returns dlpack for reading from mxnet array.

    Returns
    -------
    fn : tensor -> dlpack
    """
    def to_dlpack_for_read(data):
        data.wait_to_read()
        dlpack = DLPackHandle()
        check_call(_LIB.MXNDArrayToDLPack(data.handle, ctypes.byref(dlpack)))
        return ctypes.pythonapi.PyCapsule_New(dlpack, _c_str_dltensor, _c_dlpack_deleter)
    return to_dlpack_for_read

def ndarray_to_dlpack_for_write():
    """Returns a function that returns dlpack for writing from mxnet array.

    Returns
    -------
    fn : tensor -> dlpack
    """
    def to_dlpack_for_write(data):

        check_call(_LIB.MXNDArrayWaitToWrite(data.handle))
        dlpack = DLPackHandle()
        check_call(_LIB.MXNDArrayToDLPack(data.handle, ctypes.byref(dlpack)))
        return ctypes.pythonapi.PyCapsule_New(dlpack, _c_str_dltensor, _c_dlpack_deleter)
    return to_dlpack_for_write

def ndarray_from_numpy(array_cls, array_create_fn):
    """Returns a function that creates array_cls from numpy array.

    Returns
    -------
    fn : tensor -> dlpack
    """
    def from_numpy(ndarray, zero_copy=True):
        def _make_manager_ctx(obj):
            pyobj = ctypes.py_object(obj)
            void_p = ctypes.c_void_p.from_buffer(pyobj)
            ctypes.pythonapi.Py_IncRef(pyobj)
            return void_p

        def _make_dl_tensor(array):
            if str(array.dtype) not in DLDataType.TYPE_MAP:
                raise ValueError(str(array.dtype) + " is not supported.")
            dl_tensor = DLTensor()
            dl_tensor.data = array.ctypes.data_as(ctypes.c_void_p)
            dl_tensor.ctx = DLContext(1, 0)
            dl_tensor.ndim = array.ndim
            dl_tensor.dtype = DLDataType.TYPE_MAP[str(array.dtype)]
            dl_tensor.shape = array.ctypes.shape_as(ctypes.c_int64)
            dl_tensor.strides = None
            dl_tensor.byte_offset = 0
            return dl_tensor

        def _make_dl_managed_tensor(array):
            c_obj = DLManagedTensor()
            c_obj.dl_tensor = _make_dl_tensor(array)
            c_obj.manager_ctx = _make_manager_ctx(array)
            c_obj.deleter = dl_managed_tensor_deleter
            return c_obj

        if not zero_copy:
            return array_create_fn(ndarray, dtype=ndarray.dtype)

        if not ndarray.flags['C_CONTIGUOUS']:
            raise ValueError("Only c-contiguous arrays are supported for zero-copy")

        ndarray.flags['WRITEABLE'] = False
        c_obj = _make_dl_managed_tensor(ndarray)
        handle = NDArrayHandle()
        check_call(_LIB.MXNDArrayFromDLPack(ctypes.byref(c_obj), True, ctypes.byref(handle)))
        return array_cls(handle=handle)
    return from_numpy
