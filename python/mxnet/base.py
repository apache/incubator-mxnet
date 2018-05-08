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
# pylint: disable=invalid-name, no-member, trailing-comma-tuple
"""ctypes library of mxnet and helper functions."""
from __future__ import absolute_import

import os
import sys
import ctypes
import atexit
import warnings
import inspect
import numpy as np
from . import libinfo
warnings.filterwarnings('default', category=DeprecationWarning)

__all__ = ['MXNetError']
#----------------------------
# library loading
#----------------------------
if sys.version_info[0] == 3:
    string_types = str,
    numeric_types = (float, int, np.generic)
    integer_types = (int, np.int32, np.int64)
    # this function is needed for python3
    # to convert ctypes.char_p .value back to python str
    py_str = lambda x: x.decode('utf-8')
else:
    string_types = basestring,
    numeric_types = (float, int, long, np.generic)
    integer_types = (int, long, np.int32, np.int64)
    py_str = lambda x: x

class _NullType(object):
    """Placeholder for arguments"""
    def __repr__(self):
        return '_Null'

_Null = _NullType()

class MXNetError(Exception):
    """Error that will be throwed by all mxnet functions."""
    pass

class NotImplementedForSymbol(MXNetError):
    """Error: Not implemented for symbol"""
    def __init__(self, function, alias, *args):
        super(NotImplementedForSymbol, self).__init__()
        self.function = function.__name__
        self.alias = alias
        self.args = [str(type(a)) for a in args]
    def __str__(self):
        msg = 'Function {}'.format(self.function)
        if self.alias:
            msg += ' (namely operator "{}")'.format(self.alias)
        if self.args:
            msg += ' with arguments ({})'.format(', '.join(self.args))
        msg += ' is not implemented for Symbol and only available in NDArray.'
        return msg

class NotSupportedForSparseNDArray(MXNetError):
    """Error: Not supported for SparseNDArray"""
    def __init__(self, function, alias, *args):
        super(NotSupportedForSparseNDArray, self).__init__()
        self.function = function.__name__
        self.alias = alias
        self.args = [str(type(a)) for a in args]
    def __str__(self):
        msg = 'Function {}'.format(self.function)
        if self.alias:
            msg += ' (namely operator "{}")'.format(self.alias)
        if self.args:
            msg += ' with arguments ({})'.format(', '.join(self.args))
        msg += ' is not supported for SparseNDArray and only available in NDArray.'
        return msg

class MXCallbackList(ctypes.Structure):
    """Structure that holds Callback information. Passed to CustomOpProp."""
    _fields_ = [
        ('num_callbacks', ctypes.c_int),
        ('callbacks', ctypes.POINTER(ctypes.CFUNCTYPE(ctypes.c_int))),
        ('contexts', ctypes.POINTER(ctypes.c_void_p))
        ]


def _load_lib():
    """Load library by searching possible path."""
    lib_path = libinfo.find_lib_path()
    lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_LOCAL)
    # DMatrix functions
    lib.MXGetLastError.restype = ctypes.c_char_p
    return lib

# version number
__version__ = libinfo.__version__
# library instance of mxnet
_LIB = _load_lib()

# type definitions
mx_uint = ctypes.c_uint
mx_float = ctypes.c_float
mx_float_p = ctypes.POINTER(mx_float)
mx_real_t = np.float32
NDArrayHandle = ctypes.c_void_p
FunctionHandle = ctypes.c_void_p
OpHandle = ctypes.c_void_p
CachedOpHandle = ctypes.c_void_p
SymbolHandle = ctypes.c_void_p
ExecutorHandle = ctypes.c_void_p
DataIterCreatorHandle = ctypes.c_void_p
DataIterHandle = ctypes.c_void_p
KVStoreHandle = ctypes.c_void_p
RecordIOHandle = ctypes.c_void_p
RtcHandle = ctypes.c_void_p
CudaModuleHandle = ctypes.c_void_p
CudaKernelHandle = ctypes.c_void_p
ProfileHandle = ctypes.c_void_p
#----------------------------
# helper function definition
#----------------------------
def check_call(ret):
    """Check the return value of C API call.

    This function will raise an exception when an error occurs.
    Wrap every API call with this function.

    Parameters
    ----------
    ret : int
        return value from API calls.
    """
    if ret != 0:
        raise MXNetError(py_str(_LIB.MXGetLastError()))


if sys.version_info[0] < 3:
    def c_str(string):
        """Create ctypes char * from a Python string.

        Parameters
        ----------
        string : string type
            Python string.

        Returns
        -------
        str : c_char_p
            A char pointer that can be passed to C API.

        Examples
        --------
        >>> x = mx.base.c_str("Hello, World")
        >>> print x.value
        Hello, World
        """
        return ctypes.c_char_p(string)

    def c_str_array(strings):
        """Create ctypes const char ** from a list of Python strings.

        Parameters
        ----------
        strings : list of string
            Python strings.

        Returns
        -------
        (ctypes.c_char_p * len(strings))
            A const char ** pointer that can be passed to C API.
        """
        arr = (ctypes.c_char_p * len(strings))()
        arr[:] = strings
        return arr

else:
    def c_str(string):
        """Create ctypes char * from a Python string.

        Parameters
        ----------
        string : string type
            Python string.

        Returns
        -------
        str : c_char_p
            A char pointer that can be passed to C API.

        Examples
        --------
        >>> x = mx.base.c_str("Hello, World")
        >>> print(x.value)
        b"Hello, World"
        """
        return ctypes.c_char_p(string.encode('utf-8'))

    def c_str_array(strings):
        """Create ctypes const char ** from a list of Python strings.

        Parameters
        ----------
        strings : list of string
            Python strings.

        Returns
        -------
        (ctypes.c_char_p * len(strings))
            A const char ** pointer that can be passed to C API.
        """
        arr = (ctypes.c_char_p * len(strings))()
        arr[:] = [s.encode('utf-8') for s in strings]
        return arr

class _MXClassPropertyDescriptor(object):

    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, clas=None):
        if clas is None:
            clas = type(obj)
        return self.fget.__get__(obj, clas)()

    def __set__(self, obj, value):
        if not self.fset:
            raise MXNetError("cannot use the setter: %s to set attribute".format(obj.__name__))
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self

def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return _MXClassPropertyDescriptor(func)



def c_array(ctype, values):
    """Create ctypes array from a Python array.

    Parameters
    ----------
    ctype : ctypes data type
        Data type of the array we want to convert to, such as mx_float.

    values : tuple or list
        Data content.

    Returns
    -------
    out : ctypes array
        Created ctypes array.

    Examples
    --------
    >>> x = mx.base.c_array(mx.base.mx_float, [1, 2, 3])
    >>> print len(x)
    3
    >>> x[1]
    2.0
    """
    out = (ctype * len(values))()
    out[:] = values
    return out


def c_array_buf(ctype, buf):
    """Create ctypes array from a Python buffer.
    For primitive types, using the buffer created with array.array is faster
    than a c_array call.

    Parameters
    ----------
    ctype : ctypes data type
        Data type of the array we want to convert to, such as mx_float.

    buf : buffer type
        Data content.

    Returns
    -------
    out : ctypes array
        Created ctypes array.

    Examples
    --------
    >>> x = mx.base.c_array_buf(mx.base.mx_float, array.array('i', [1, 2, 3]))
    >>> print len(x)
    3
    >>> x[1]
    2.0
    """
    return (ctype * len(buf)).from_buffer(buf)

def c_handle_array(objs):
    """Create ctypes const void ** from a list of MXNet objects with handles.

    Parameters
    ----------
    objs : list of NDArray/Symbol.
        MXNet objects.

    Returns
    -------
    (ctypes.c_void_p * len(objs))
        A void ** pointer that can be passed to C API.
    """
    arr = (ctypes.c_void_p * len(objs))()
    arr[:] = [o.handle for o in objs]
    return arr

def ctypes2buffer(cptr, length):
    """Convert ctypes pointer to buffer type.

    Parameters
    ----------
    cptr : ctypes.POINTER(ctypes.c_char)
        Pointer to the raw memory region.
    length : int
        The length of the buffer.

    Returns
    -------
    buffer : bytearray
        The raw byte memory buffer.
    """
    if not isinstance(cptr, ctypes.POINTER(ctypes.c_char)):
        raise TypeError('expected char pointer')
    res = bytearray(length)
    rptr = (ctypes.c_char * length).from_buffer(res)
    if not ctypes.memmove(rptr, cptr, length):
        raise RuntimeError('memmove failed')
    return res

def ctypes2numpy_shared(cptr, shape):
    """Convert a ctypes pointer to a numpy array.

    The resulting NumPy array shares the memory with the pointer.

    Parameters
    ----------
    cptr : ctypes.POINTER(mx_float)
        pointer to the memory region

    shape : tuple
        Shape of target `NDArray`.

    Returns
    -------
    out : numpy_array
        A numpy array : numpy array.
    """
    if not isinstance(cptr, ctypes.POINTER(mx_float)):
        raise RuntimeError('expected float pointer')
    size = 1
    for s in shape:
        size *= s
    dbuffer = (mx_float * size).from_address(ctypes.addressof(cptr.contents))
    return np.frombuffer(dbuffer, dtype=np.float32).reshape(shape)


def build_param_doc(arg_names, arg_types, arg_descs, remove_dup=True):
    """Build argument docs in python style.

    arg_names : list of str
        Argument names.

    arg_types : list of str
        Argument type information.

    arg_descs : list of str
        Argument description information.

    remove_dup : boolean, optional
        Whether remove duplication or not.

    Returns
    -------
    docstr : str
        Python docstring of parameter sections.
    """
    param_keys = set()
    param_str = []
    for key, type_info, desc in zip(arg_names, arg_types, arg_descs):
        if key in param_keys and remove_dup:
            continue
        if key == 'num_args':
            continue
        param_keys.add(key)
        ret = '%s : %s' % (key, type_info)
        if len(desc) != 0:
            ret += '\n    ' + desc
        param_str.append(ret)
    doc_str = ('Parameters\n' +
               '----------\n' +
               '%s\n')
    doc_str = doc_str % ('\n'.join(param_str))
    return doc_str


def _notify_shutdown():
    """Notify MXNet about a shutdown."""
    check_call(_LIB.MXNotifyShutdown())

atexit.register(_notify_shutdown)


def add_fileline_to_docstring(module, incursive=True):
    """Append the definition position to each function contained in module.

    Examples
    --------
    # Put the following codes at the end of a file
    add_fileline_to_docstring(__name__)
    """

    def _add_fileline(obj):
        """Add fileinto to a object.
        """
        if obj.__doc__ is None or 'From:' in obj.__doc__:
            return
        fname = inspect.getsourcefile(obj)
        if fname is None:
            return
        try:
            line = inspect.getsourcelines(obj)[-1]
        except IOError:
            return
        obj.__doc__ += '\n\nFrom:%s:%d' % (fname, line)

    if isinstance(module, str):
        module = sys.modules[module]
    for _, obj in inspect.getmembers(module):
        if inspect.isbuiltin(obj):
            continue
        if inspect.isfunction(obj):
            _add_fileline(obj)
        if inspect.ismethod(obj):
            _add_fileline(obj.__func__)
        if inspect.isclass(obj) and incursive:
            add_fileline_to_docstring(obj, False)


def _as_list(obj):
    """A utility function that converts the argument to a list if it is not already.

    Parameters
    ----------
    obj : object

    Returns
    -------
    If `obj` is a list or tuple, return it. Otherwise, return `[obj]` as a
    single-element list.

    """
    if isinstance(obj, (list, tuple)):
        return obj
    else:
        return [obj]


_OP_NAME_PREFIX_LIST = ['_contrib_', '_linalg_', '_sparse_', '_image_']


def _get_op_name_prefix(op_name):
    """
    Check whether the given op_name starts with any words in `_OP_NAME_PREFIX_LIST`.
    If found, return the prefix; else, return an empty string.
    """
    for prefix in _OP_NAME_PREFIX_LIST:
        if op_name.startswith(prefix):
            return prefix
    return ""


# pylint: enable=too-many-locals, invalid-name
def _init_op_module(root_namespace, module_name, make_op_func):
    """
    Registers op functions created by `make_op_func` under
    `root_namespace.module_name.[submodule_name]`,
    where `submodule_name` is one of `_OP_SUBMODULE_NAME_LIST`.

    Parameters
    ----------
    root_namespace : str
        Top level module name, `mxnet` in the current cases.
    module_name : str
        Second level module name, `ndarray` and `symbol` in the current cases.
    make_op_func : function
        Function for creating op functions for `ndarray` and `symbol` modules.
    """
    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()

    check_call(_LIB.MXListAllOpNames(ctypes.byref(size),
                                     ctypes.byref(plist)))
    op_names = []
    for i in range(size.value):
        op_names.append(py_str(plist[i]))

    module_op = sys.modules["%s.%s.op" % (root_namespace, module_name)]
    module_internal = sys.modules["%s.%s._internal" % (root_namespace, module_name)]
    # contrib module in the old format (deprecated)
    # kept here for backward compatibility
    # use mx.nd.contrib or mx.sym.contrib from now on
    contrib_module_name_old = "%s.contrib.%s" % (root_namespace, module_name)
    contrib_module_old = sys.modules[contrib_module_name_old]
    submodule_dict = {}
    for op_name_prefix in _OP_NAME_PREFIX_LIST:
        submodule_dict[op_name_prefix] =\
            sys.modules["%s.%s.%s" % (root_namespace, module_name, op_name_prefix[1:-1])]
    for name in op_names:
        hdl = OpHandle()
        check_call(_LIB.NNGetOpHandle(c_str(name), ctypes.byref(hdl)))
        op_name_prefix = _get_op_name_prefix(name)
        module_name_local = module_name
        if len(op_name_prefix) > 0:
            func_name = name[len(op_name_prefix):]
            cur_module = submodule_dict[op_name_prefix]
            module_name_local = "%s.%s.%s" % (root_namespace, module_name, op_name_prefix[1:-1])
        elif name.startswith('_'):
            func_name = name
            cur_module = module_internal
        else:
            func_name = name
            cur_module = module_op

        function = make_op_func(hdl, name, func_name)
        function.__module__ = module_name_local
        setattr(cur_module, function.__name__, function)
        cur_module.__all__.append(function.__name__)


        if op_name_prefix == '_contrib_':
            hdl = OpHandle()
            check_call(_LIB.NNGetOpHandle(c_str(name), ctypes.byref(hdl)))
            func_name = name[len(op_name_prefix):]

            function = make_op_func(hdl, name, func_name)
            function.__module__ = contrib_module_name_old
            setattr(contrib_module_old, function.__name__, function)
            contrib_module_old.__all__.append(function.__name__)


def _generate_op_module_signature(root_namespace, module_name, op_code_gen_func):
    """
    Generate op functions created by `op_code_gen_func` and write to the source file
    of `root_namespace.module_name.[submodule_name]`,
    where `submodule_name` is one of `_OP_SUBMODULE_NAME_LIST`.

    Parameters
    ----------
    root_namespace : str
        Top level module name, `mxnet` in the current cases.
    module_name : str
        Second level module name, `ndarray` and `symbol` in the current cases.
    op_code_gen_func : function
        Function for creating op functions for `ndarray` and `symbol` modules.
    """
    def get_module_file(module_name):
        """Return the generated module file based on module name."""
        path = os.path.dirname(__file__)
        module_path = module_name.split('.')
        module_path[-1] = 'gen_'+module_path[-1]
        file_name = os.path.join(path, '..', *module_path) + '.py'
        module_file = open(file_name, 'w')
        dependencies = {'symbol': ['from ._internal import SymbolBase',
                                   'from ..base import _Null'],
                        'ndarray': ['from ._internal import NDArrayBase',
                                    'from ..base import _Null']}
        module_file.write('# File content is auto-generated. Do not modify.'+os.linesep)
        module_file.write('# pylint: skip-file'+os.linesep)
        module_file.write(os.linesep.join(dependencies[module_name.split('.')[1]]))
        return module_file
    def write_all_str(module_file, module_all_list):
        """Write the proper __all__ based on available operators."""
        module_file.write(os.linesep)
        module_file.write(os.linesep)
        all_str = '__all__ = [' + ', '.join(["'%s'"%s for s in module_all_list]) + ']'
        module_file.write(all_str)

    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()

    check_call(_LIB.MXListAllOpNames(ctypes.byref(size),
                                     ctypes.byref(plist)))
    op_names = []
    for i in range(size.value):
        op_names.append(py_str(plist[i]))

    module_op_file = get_module_file("%s.%s.op" % (root_namespace, module_name))
    module_op_all = []
    module_internal_file = get_module_file("%s.%s._internal"%(root_namespace, module_name))
    module_internal_all = []
    submodule_dict = {}
    for op_name_prefix in _OP_NAME_PREFIX_LIST:
        submodule_dict[op_name_prefix] =\
            (get_module_file("%s.%s.%s" % (root_namespace, module_name,
                                           op_name_prefix[1:-1])), [])
    for name in op_names:
        hdl = OpHandle()
        check_call(_LIB.NNGetOpHandle(c_str(name), ctypes.byref(hdl)))
        op_name_prefix = _get_op_name_prefix(name)
        if len(op_name_prefix) > 0:
            func_name = name[len(op_name_prefix):]
            cur_module_file, cur_module_all = submodule_dict[op_name_prefix]
        elif name.startswith('_'):
            func_name = name
            cur_module_file = module_internal_file
            cur_module_all = module_internal_all
        else:
            func_name = name
            cur_module_file = module_op_file
            cur_module_all = module_op_all

        code, _ = op_code_gen_func(hdl, name, func_name, True)
        cur_module_file.write(os.linesep)
        cur_module_file.write(code)
        cur_module_all.append(func_name)

    for (submodule_f, submodule_all) in submodule_dict.values():
        write_all_str(submodule_f, submodule_all)
        submodule_f.close()
    write_all_str(module_op_file, module_op_all)
    module_op_file.close()
    write_all_str(module_internal_file, module_internal_all)
    module_internal_file.close()
