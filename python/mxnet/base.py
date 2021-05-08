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
# pylint: disable=invalid-name, no-member, trailing-comma-tuple, bad-mcs-classmethod-argument, unnecessary-pass, too-many-lines, wrong-import-position
"""ctypes library of mxnet and helper functions."""

import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np

from . import libinfo

__all__ = ['MXNetError']
#----------------------------
# library loading
#----------------------------

# pylint: disable=pointless-statement
try:
    basestring
    long
except NameError:
    basestring = str
    long = int
# pylint: enable=pointless-statement

integer_types = (int, long, _np.int32, _np.int64)
numeric_types = (float, int, long, _np.generic)
string_types = basestring,
error_types = {}

# this function is needed for python3
# to convert ctypes.char_p .value back to python str
if sys.platform == "win32":
    encoding = 'cp' + str(ctypes.cdll.kernel32.GetACP())
    py_str = lambda x: x.decode(encoding)
else:
    py_str = lambda x: x.decode('utf-8')


def data_dir_default():
    """

    :return: default data directory depending on the platform and environment variables
    """
    system = platform.system()
    if system == 'Windows':
        return os.path.join(os.environ.get('APPDATA'), 'mxnet')
    else:
        return os.path.join(os.path.expanduser("~"), '.mxnet')


def data_dir():
    """

    :return: data directory in the filesystem for storage, for example when downloading models
    """
    return os.getenv('MXNET_HOME', data_dir_default())


class _NullType(object):
    """Placeholder for arguments"""
    def __repr__(self):
        return '_Null'


_Null = _NullType()


class MXNetError(RuntimeError):
    """Default error thrown by MXNet functions.

    MXNetError will be raised if you do not give any error type specification,
    """

def register_error(func_name=None, cls=None):
    """Register an error class so it can be recognized by the ffi error handler.

    Parameters
    ----------
    func_name : str or function or class
        The name of the error function.

    cls : function
        The function to create the class

    Returns
    -------
    fregister : function
        Register function if f is not specified.

    Examples
    --------
    .. code-block:: python

      @mxnet.error.register_error
      class MyError(RuntimeError):
          pass

      err_inst = mxnet.error.create_ffi_error("MyError: xyz")
      assert isinstance(err_inst, MyError)
    """
    if callable(func_name):
        cls = func_name
        func_name = cls.__name__

    def register(mycls):
        """internal register function"""
        err_name = func_name if isinstance(func_name, str) else mycls.__name__
        error_types[err_name] = mycls
        return mycls
    if cls is None:
        return register
    return register(cls)


def _valid_error_name(name):
    """Check whether name is a valid error name."""
    return all(x.isalnum() or x in "_." for x in name)


def _find_error_type(line):
    """Find the error name given the first line of the error message.

    Parameters
    ----------
    line : str
        The first line of error message.

    Returns
    -------
    name : str The error name
    """
    end_pos = line.find(":")
    if end_pos == -1:
        return None
    err_name = line[:end_pos]
    if _valid_error_name(err_name):
        return err_name
    return None


def c2pyerror(err_msg):
    """Translate C API error message to python style.

    Parameters
    ----------
    err_msg : str
        The error message.

    Returns
    -------
    new_msg : str
        Translated message.

    err_type : str
        Detected error type.
    """
    arr = err_msg.split("\n")
    if arr[-1] == "":
        arr.pop()
    err_type = _find_error_type(arr[0])
    trace_mode = False
    stack_trace = []
    message = []
    for line in arr:
        if trace_mode:
            if line.startswith("  "):
                stack_trace.append(line)
            else:
                trace_mode = False
        if not trace_mode:
            if line.startswith("Stack trace"):
                trace_mode = True
            else:
                message.append(line)
    out_msg = ""
    if stack_trace:
        out_msg += "Traceback (most recent call last):\n"
        out_msg += "\n".join(reversed(stack_trace)) + "\n"
    out_msg += "\n".join(message)
    return out_msg, err_type

@register_error
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


def get_last_ffi_error():
    """Create error object given result of MXGetLastError.

    Returns
    -------
    err : object
        The error object based on the err_msg
    """
    c_err_msg = py_str(_LIB.MXGetLastError())
    py_err_msg, err_type = c2pyerror(c_err_msg)
    if err_type is not None and err_type.startswith("mxnet.error."):
        err_type = err_type[10:]
    return error_types.get(err_type, MXNetError)(py_err_msg)


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
        raise get_last_ffi_error()


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


# pylint: disable=line-too-long
def _load_lib():
    """Load library by searching possible path."""
    lib_path = libinfo.find_lib_path()
    try:
        if sys.version_info >= (3, 8) and os.name == "nt":
            # use LOAD_WITH_ALTERED_SEARCH_PATH, For simplicity, let's just fill the numbers.
            # pylint: disable=E1123
            lib = ctypes.CDLL(lib_path[0], winmode=0x00000008)
        else:
            lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_LOCAL)
        # DMatrix functions
        lib.MXGetLastError.restype = ctypes.c_char_p
    except OSError as e:
        if "libcudnn" in e.args[0]:
            e.args = (e.args[0]+'\nNotes: Starting from version 1.8.0, cuDNN and NCCL should be installed by users in advance. \
                      \nPlease follow the instructions in https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html to install cuDNN.',)
            raise OSError(e) from None
        if "libnccl" in e.args[0]:
            e.args = (e.args[0]+'\nNotes: Starting from version 1.8.0, cuDNN and NCCL should be installed by users in advance. \
                      \nPlease follow the instructions in https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html to install NCCL.',)
            raise OSError(e) from None
        if "libquadmath" in e.args[0]:
            e.args = (e.args[0]+'\nNotes: As libquadmath.so.0 is a GPL library and MXNet part of the Apache Software Foundation, \
                      \nMXNet must not redistribute libquadmath.so.0 as part of the Pypi package and users must manually install it. \
                      \nOn Debian based systems, including Ubuntu, run sudo apt install libquadmath0 to install the shared library. \
                      \nOn RHEL based systems, including CentOS, run sudo yum install libquadmath to install the shared library. ')
            raise OSError(e) from None
        raise
    else:
        return lib


# version number
__version__ = libinfo.__version__
# library instance of mxnet
_LIB = _load_lib()

# type definitions
mx_int = ctypes.c_int
mx_uint = ctypes.c_uint
mx_int64 = ctypes.c_int64
mx_float = ctypes.c_float
mx_float_p = ctypes.POINTER(mx_float)
mx_real_t = _np.float32
NDArrayHandle = ctypes.c_void_p
FunctionHandle = ctypes.c_void_p
OpHandle = ctypes.c_void_p
CachedOpHandle = ctypes.c_void_p
SymbolHandle = ctypes.c_void_p
DataIterCreatorHandle = ctypes.c_void_p
DataIterHandle = ctypes.c_void_p
DatasetHandle = ctypes.c_void_p
BatchifyFunctionhandle = ctypes.c_void_p
KVStoreHandle = ctypes.c_void_p
RecordIOHandle = ctypes.c_void_p
RtcHandle = ctypes.c_void_p
CudaModuleHandle = ctypes.c_void_p
CudaKernelHandle = ctypes.c_void_p
ProfileHandle = ctypes.c_void_p


#----------------------------
# helper function definition
#----------------------------
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
    return _np.frombuffer(dbuffer, dtype=_np.float32).reshape(shape)


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


_OP_NAME_PREFIX_LIST = ['_contrib_', '_linalg_', '_sparse_', '_image_', '_random_']


def _get_op_name_prefix(op_name):
    """
    Check whether the given op_name starts with any words in `_OP_NAME_PREFIX_LIST`.
    If found, return the prefix; else, return an empty string.
    """
    for prefix in _OP_NAME_PREFIX_LIST:
        if op_name.startswith(prefix):
            return prefix
    return ""


# pylint: enable=invalid-name
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
        op_name = py_str(plist[i])
        if not _is_np_op(op_name):
            op_names.append(op_name)

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
            if op_name_prefix != '_random_' or name.endswith('_like'):
                func_name = name[len(op_name_prefix):]
                cur_module = submodule_dict[op_name_prefix]
                module_name_local = "%s.%s.%s" % (root_namespace, module_name, op_name_prefix[1:-1])
            else:
                func_name = name
                cur_module = module_internal
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
    license_lines = [
        '# Licensed to the Apache Software Foundation (ASF) under one',
        '# or more contributor license agreements.  See the NOTICE file',
        '# distributed with this work for additional information',
        '# regarding copyright ownership.  The ASF licenses this file',
        '# to you under the Apache License, Version 2.0 (the',
        '# "License"); you may not use this file except in compliance',
        '# with the License.  You may obtain a copy of the License at',
        '#',
        '#   http://www.apache.org/licenses/LICENSE-2.0',
        '#',
        '# Unless required by applicable law or agreed to in writing,',
        '# software distributed under the License is distributed on an',
        '# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY',
        '# KIND, either express or implied.  See the License for the',
        '# specific language governing permissions and limitations',
        '# under the License.',
        '',
    ]
    license_str = os.linesep.join(license_lines)
    def get_module_file(module_name):
        """Return the generated module file based on module name."""
        path = os.path.dirname(__file__)
        module_path = module_name.split('.')
        module_path[-1] = 'gen_' + module_path[-1]
        file_name = os.path.join(path, '..', *module_path) + '.py'
        module_file = open(file_name, 'w', encoding="utf-8")
        dependencies = {'symbol': ['from ._internal import SymbolBase',
                                   'from ..base import _Null'],
                        'ndarray': ['from ._internal import NDArrayBase',
                                    'from ..base import _Null']}
        module_file.write('# coding: utf-8')
        module_file.write(license_str)
        module_file.write('# File content is auto-generated. Do not modify.' + os.linesep)
        module_file.write('# pylint: skip-file' + os.linesep)
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
        op_name = py_str(plist[i])
        if not _is_np_op(op_name):
            op_names.append(op_name)

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

ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p


_NP_OP_PREFIX = '_np_'
_NP_OP_SUBMODULE_LIST = ['_random_', '_linalg_']
_NP_OP_IMPLEMENTED_SET = {'_np_reshape'}

_NP_EXT_OP_PREFIX = '_npx_'
_NP_EXT_OP_SUBMODULE_LIST = ['_image_', '_random_']
_NP_EXT_OP_IMPLEMENTED_SET = {'_npx_softmax', '_npx_log_softmax', '_npx_masked_softmax',
                              '_npx_masked_log_softmax', '_npx_activation',
                              '_npx_batch_norm', '_npx_fully_connected', '_npx_pick',
                              '_npx_convolution', '_npx_deconvolution', '_npx_pooling',
                              '_npx_dropout', '_npx_one_hot', '_npx_rnn', '_npx_embedding',
                              '_npx_topk', '_npx_layer_norm', '_npx_leaky_relu', '_npx_batch_dot',
                              '_npx_broadcast_like', '_npx_arange_like'}

_NP_INTERNAL_OP_PREFIX = '_npi_'

_NP_OUTPUT_IS_LIST_OPERATORS = {'_npi_split', '_npi_hsplit'}


def _is_np_op(op_name):
    return op_name.startswith(_NP_OP_PREFIX) or op_name.startswith(_NP_EXT_OP_PREFIX)\
           or op_name.startswith(_NP_INTERNAL_OP_PREFIX)


def _output_is_list(op_name):
    """ Whether the output of the operator is a list.

    Parameters
    ----------
    op_name : Name of the operator

    Returns
    -------

    """
    if _is_np_op(op_name):
        return op_name in _NP_OUTPUT_IS_LIST_OPERATORS
    return False


def _get_op_submodule_name(op_name, op_name_prefix, submodule_name_list):
    """Get the submodule name of a specific op"""
    assert op_name.startswith(op_name_prefix)
    for submodule_name in submodule_name_list:
        if op_name[len(op_name_prefix):].startswith(submodule_name):
            return submodule_name
    return ""


def _init_np_op_module(root_module_name, np_module_name, mx_module_name, make_op_func):
    """
    Register numpy operators in namespaces `mxnet.numpy`, `mxnet.ndarray.numpy`
    and `mxnet.symbol.numpy`. They are used in imperative mode, Gluon APIs w/o hybridization,
    and Gluon APIs w/ hybridization, respectively. Essentially, operators with the same name
    registered in three namespaces, respectively share the same functionality in C++ backend.
    Different namespaces are needed for dispatching operator calls in Gluon's `HybridBlock` by `F`.

    Parameters
    ----------
    root_module_name : str
        Top level module name, `mxnet` in the current cases.
    np_module_name : str
        Second level module name, `numpy` or `numpy_extension` in the current case.
    make_op_func : function
        Function for creating op functions.
    """
    from . import _numpy_op_doc as _np_op_doc
    if np_module_name == 'numpy':
        op_name_prefix = _NP_OP_PREFIX
        submodule_name_list = _NP_OP_SUBMODULE_LIST
        op_implemented_set = _NP_OP_IMPLEMENTED_SET
    elif np_module_name == 'numpy_extension':
        op_name_prefix = _NP_EXT_OP_PREFIX
        submodule_name_list = _NP_EXT_OP_SUBMODULE_LIST
        op_implemented_set = _NP_EXT_OP_IMPLEMENTED_SET
    elif np_module_name == 'numpy._internal':
        op_name_prefix = _NP_INTERNAL_OP_PREFIX
        submodule_name_list = []
        op_implemented_set = set()
    else:
        raise ValueError('unsupported np module name {}'.format(np_module_name))

    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()
    check_call(_LIB.MXListAllOpNames(ctypes.byref(size), ctypes.byref(plist)))
    op_names = []
    for i in range(size.value):
        name = py_str(plist[i])
        if mx_module_name != 'symbol':
            if name.startswith(op_name_prefix) and name not in op_implemented_set:
                op_names.append(name)
        else:
            if name.startswith(op_name_prefix):
                op_names.append(name)

    if mx_module_name is None:
        # register np/npx ops for imperative programming
        op_module_name = "%s.%s._op" % (root_module_name, np_module_name)  # e.g. mxnet.numpy._op
        op_submodule_name = "%s.%s" % (root_module_name, np_module_name)  # e.g. mxnet.numpy.random
    elif mx_module_name in ('ndarray', 'symbol'):
        # register numpy internal ops and np/npx ops for use in Gluon
        # np internal ops are registered in mxnet.ndarray/symbol.numpy._internal
        # np ops are registered in mxnet.ndarray/symbol.numpy._op
        # npx ops are registered in mxnet.ndarray/symbol.numpy_extension._op
        op_module_name = "%s.%s.%s" % (root_module_name, mx_module_name, np_module_name)
        if op_name_prefix != _NP_INTERNAL_OP_PREFIX:
            op_module_name += '._op'
        # e.g. mxnet.symbol.numpy.random
        op_submodule_name = "%s.%s.%s" % (root_module_name, mx_module_name, np_module_name)
    else:
        raise ValueError('unsupported mxnet module {}'.format(mx_module_name))
    op_submodule_name += '.%s'

    op_module = sys.modules[op_module_name]
    submodule_dict = {}
    for submodule_name in submodule_name_list:
        submodule_dict[submodule_name] = sys.modules[op_submodule_name % submodule_name[1:-1]]
    for name in op_names:
        hdl = OpHandle()
        check_call(_LIB.NNGetOpHandle(c_str(name), ctypes.byref(hdl)))
        submodule_name = _get_op_submodule_name(name, op_name_prefix, submodule_name_list)
        if len(submodule_name) > 0:
            func_name = name[(len(op_name_prefix) + len(submodule_name)):]
            cur_module = submodule_dict[submodule_name]
            module_name_local = op_submodule_name % submodule_name[1:-1]
        else:
            func_name = name[len(op_name_prefix):]
            cur_module = op_module
            module_name_local =\
                op_module_name[:-len('._op')] if op_module_name.endswith('._op') else op_module_name

        function = make_op_func(hdl, name, func_name)
        function.__module__ = module_name_local
        setattr(cur_module, function.__name__, function)
        cur_module.__all__.append(function.__name__)

        if hasattr(_np_op_doc, name):
            function.__doc__ = getattr(_np_op_doc, name).__doc__
        else:
            function.__doc__ = re.sub('NDArray', 'ndarray', function.__doc__)
