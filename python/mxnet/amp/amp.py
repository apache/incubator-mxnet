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
"""Functions for enabling AMP (automatic mixed precision)."""
__all__ = ['init', 'init_trainer', 'scale_loss', 'unscale', 'convert_model',
           'convert_hybrid_block', 'list_lp16_ops', 'list_fp32_ops',
           'list_lp16_fp32_ops', 'list_conditional_fp32_ops',
           'list_widest_type_cast', 'list_loss_output_functions', 'list_lp16_use_fp32_params',
           'convert_symbol']

from array import array
import ctypes
import inspect
import logging
import contextlib
import sys
import numpy as np

from mxnet import numpy
from .. import symbol
from ..context import gpu
from ..symbol import Symbol
from ..symbol import contrib as symbol_contrib
from .. import ndarray
from ..ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from . import lists
from ..gluon import Block, trainer
from .. import base
from ..base import (_NP_OP_PREFIX, _NP_OP_SUBMODULE_LIST, _NP_EXT_OP_PREFIX,
                    _NP_EXT_OP_SUBMODULE_LIST, _NP_INTERNAL_OP_PREFIX,
                    c_str_array, SymbolHandle, check_call, _LIB, mx_uint, c_array_buf)
from .. import optimizer as opt
from .loss_scaler import LossScaler
from ..operator import get_all_registered_operators_grouped

bfloat16 = np.dtype([('bfloat16', np.uint16)])

float_types_gpu = (np.float16, np.float32)
float_types_cpu = (bfloat16, np.float32)

def _cast_symbol_NDArray(s, dtype, is_numpy_module=False):
    if isinstance(s, Symbol):
        amp_cast = symbol.numpy._internal.amp_cast if is_numpy_module else symbol.amp_cast
        return amp_cast(s, dtype=dtype)
    if isinstance(s, NDArray):
        if is_numpy_module:
            def amp_cast(s, dtype=None):  # pylint: disable=function-redefined
                if not isinstance(dtype, str):
                    dtype = np.dtype(dtype).name
                return ndarray.numpy._api_internal.amp_cast(s, dtype)
        else:
            amp_cast = ndarray.amp_cast
        if s.dtype != dtype and (s.dtype in float_types_gpu and s.context.device_type != 'cpu' or
                                 s.dtype in float_types_cpu and s.context.device_type == 'cpu'):
            return amp_cast(s, dtype=dtype)
    return s

def _get_nd_fun_to_wrap(name, module, submodule_dict):
    module_internal = getattr(module, "_internal")
    prefix = base._get_op_name_prefix(name)
    if prefix:
        if prefix != '_random_' or name.endswith('_like'):
            func_name = name[len(prefix):]
            cur_module = submodule_dict[prefix]
        else:
            func_name = name
            cur_module = module_internal
    elif name.startswith('_'):
        func_name = name
        cur_module = module_internal
    else:
        func_name = name
        cur_module = module
    return func_name, [cur_module]

def _get_np_fun_to_wrap(name, ns_prefix):
    for pre, mod, subs in ((_NP_OP_PREFIX, 'numpy', _NP_OP_SUBMODULE_LIST),
                           (_NP_EXT_OP_PREFIX, 'numpy_extension', _NP_EXT_OP_SUBMODULE_LIST),
                           (_NP_INTERNAL_OP_PREFIX, 'numpy._internal', [])):
        if name.startswith(pre):
            nm = name[len(pre):]
            for sub in subs:
                if nm.startswith(sub):
                    func, modules = nm[len(sub):], [sys.modules[f'{ns_prefix}.{mod}.{sub[1:-1]}']]
                    break
            else:
                func, modules = nm, [sys.modules[f'{ns_prefix}.{mod}']]
                break
    else:
        assert False, f'Unable to find target module for {name} in {ns_prefix}'
    if name.startswith(_NP_INTERNAL_OP_PREFIX) and ns_prefix == 'mxnet.ndarray':
        if hasattr(ndarray.numpy._api_internal, func):
            modules.append(ndarray.numpy._api_internal)
    return func, modules

def _wrap_module_functions(module, is_numpy_module, target_dtype, get_aliases, get_cond_aliases,
                           get_fun_to_wrap, target_precision_ops=None, conditional_fp32_ops=None,
                           fp32_ops=None):

    if is_numpy_module:
        def amp_cast(s, dtype=None):  # pylint: disable=function-redefined
            if not isinstance(dtype, str):
                dtype = np.dtype(dtype).name
            return ndarray.numpy._api_internal.amp_cast(s, dtype)
    else:
        amp_cast = ndarray.amp_cast
    sy_mod = symbol.numpy._internal if is_numpy_module else symbol

    def _ndarray_wrapper(f, target_dtype, fp32_param=None, cond_arg=None):
        def _new_fun(*args, **kwargs):
            if cond_arg is not None:
                if (cond_arg[0] not in kwargs or
                        kwargs[cond_arg[0]] not in cond_arg[1]):
                    return f(*args, **kwargs)
            if fp32_param:
                new_args = []
                for i, x in enumerate(args):
                    if fp32_param[i]:
                        new_args.append(x)
                    else:
                        new_args.append(_cast_symbol_NDArray(x, target_dtype, is_numpy_module))
            else:
                new_args = list(map(
                    lambda x: _cast_symbol_NDArray(x, target_dtype, is_numpy_module), args))
            args = tuple(new_args)
            if fp32_param:
                new_kwargs = {}
                for k, v in kwargs.items():
                    if k in fp32_param:
                        new_kwargs[k] = v
                    else:
                        new_kwargs[k] = _cast_symbol_NDArray(v, target_dtype, is_numpy_module)
                    kwargs = new_kwargs
            else:
                kwargs = {k: _cast_symbol_NDArray(v, target_dtype, is_numpy_module)
                          for k, v in kwargs.items()}
            return f(*args, **kwargs)
        _new_fun.__name__ = f.__name__
        _new_fun.__module__ = f.__module__
        _new_fun.__doc__ = f.__doc__
        return _new_fun

    def _symbol_wrapper(f, target_dtype, fp32_param=None, cond_arg=None):
        def _new_fun(*args, **kwargs):
            if cond_arg is not None:
                if (cond_arg[0] not in kwargs or
                        kwargs[cond_arg[0]] not in cond_arg[1]):
                    return f(*args, **kwargs)
            sym = f(*args, **kwargs)
            inputs = sym.get_children()
            aux = sym.list_auxiliary_states()
            if fp32_param:
                new_inputs = []
                for i, x in enumerate(inputs):
                    if (x.name in aux) or fp32_param[i]:
                        new_inputs.append(x)
                    else:
                        new_inputs.append(_cast_symbol_NDArray(x, target_dtype, is_numpy_module))
                inputs = new_inputs
            else:
                inputs = list(map(lambda x: _cast_symbol_NDArray(x, target_dtype, is_numpy_module)
                                  if x.name not in aux else x, inputs))
            atomic_sym = sym._gen_atomic_symbol()
            wrapped_sym = atomic_sym(*inputs)
            wrapped_sym._set_attr(name=sym.name)
            return wrapped_sym
        _new_fun.__name__ = f.__name__
        _new_fun.__module__ = f.__module__
        _new_fun.__doc__ = f.__doc__
        return _new_fun

    def _symbol_widest_wrapper(f):
        def _new_fun(*args, **kwargs):
            symbols = []
            is_symbol = False
            args = list(args)
            for i, arg in enumerate(args):
                if isinstance(arg, (Symbol, NDArray)):
                    symbols.append((args, i, arg))
                    is_symbol = is_symbol or isinstance(arg, Symbol)
            for k, arg in kwargs.items():
                if isinstance(arg, (Symbol, NDArray)):
                    symbols.append((kwargs, k, arg))
                    is_symbol = is_symbol or isinstance(arg, Symbol)
            if not is_symbol:
                # NDArray case
                widest_type = target_dtype
                for _, _, arg in symbols:
                    if isinstance(arg, NDArray):
                        if arg.dtype == np.float32:
                            widest_type = np.float32
                for arr, index, arg in symbols:
                    if arg.dtype != widest_type and arg.dtype == target_dtype:
                        arr[index] = amp_cast(arg, dtype=widest_type)
            else:
                # Symbol case
                sym_to_check = list(map(lambda x: x[2], symbols))
                casted_syms = sy_mod.amp_multicast(*sym_to_check, num_outputs=len(sym_to_check))
                symbols = list(map(lambda x_y: (x_y[0][0], x_y[0][1], x_y[1]),
                                   zip(symbols, casted_syms)))
                for arr, index, arg in symbols:
                    arr[index] = arg

            return f(*args, **kwargs)
        _new_fun.__name__ = f.__name__
        _new_fun.__module__ = f.__module__
        _new_fun.__doc__ = f.__doc__
        return _new_fun

    _wrapper = _symbol_wrapper if module in (symbol, Symbol, symbol_contrib) else _ndarray_wrapper

    fp32_param_list = list_lp16_use_fp32_params(target_dtype)
    wrap_list = target_precision_ops if target_precision_ops is not None \
                    else list_lp16_ops(target_dtype)
    for fun_name in get_aliases(wrap_list):
        fun_name, modules = get_fun_to_wrap(fun_name, module)
        for cur_module in modules:
            f_to_wrap = getattr(cur_module, fun_name)
            fp32_param = fp32_param_list[fun_name] if (fp32_param_list and fun_name in fp32_param_list) else None
            setattr(cur_module, fun_name, _wrapper(f_to_wrap, target_dtype, fp32_param=fp32_param))
            if not is_numpy_module and cur_module == module:
                setattr(module.op, fun_name, _wrapper(f_to_wrap, target_dtype, fp32_param=fp32_param))

    wrap_list = fp32_ops if fp32_ops is not None else list_fp32_ops(target_dtype)
    for fun_name in get_aliases(wrap_list):
        fun_name, modules = get_fun_to_wrap(fun_name, module)
        for cur_module in modules:
            f_to_wrap = getattr(cur_module, fun_name)
            setattr(cur_module, fun_name, _wrapper(f_to_wrap, np.float32))
            if not is_numpy_module and cur_module == module:
                setattr(module.op, fun_name, _wrapper(f_to_wrap, np.float32))

    wrap_list = conditional_fp32_ops if conditional_fp32_ops is not None \
                    else list_conditional_fp32_ops(target_dtype)
    for fun_name, arg, arg_values in get_cond_aliases(wrap_list):
        fun_name, modules = get_fun_to_wrap(fun_name, module)
        for cur_module in modules:
            f_to_wrap = getattr(cur_module, fun_name)
            setattr(cur_module, fun_name, _wrapper(f_to_wrap, np.float32, cond_arg=(arg, arg_values)))
            if not is_numpy_module and cur_module == module:
                setattr(module.op, fun_name, _wrapper(f_to_wrap, np.float32, cond_arg=(arg, arg_values)))

    for fun_name in get_aliases(list_widest_type_cast(target_dtype)):
        fun_name, modules = get_fun_to_wrap(fun_name, module)
        for cur_module in modules:
            f_to_wrap = getattr(cur_module, fun_name)
            setattr(cur_module, fun_name, _symbol_widest_wrapper(f_to_wrap))
            if not is_numpy_module and cur_module == module:
                setattr(module.op, fun_name, _symbol_widest_wrapper(f_to_wrap))

def _wrap_loss_output_functions(module, ls, target_dtype):
    if module == ndarray:
        def _wrapper(f):
            def _scaling_wrapper(*args, **kwargs):
                if 'grad_scale' in kwargs:
                    kwargs['grad_scale'] = kwargs['grad_scale'] * ls.loss_scale
                else:
                    kwargs['grad_scale'] = ls.loss_scale
                return f(*args, **kwargs)
            _scaling_wrapper.__name__ = f.__name__
            _scaling_wrapper.__module__ = f.__module__
            _scaling_wrapper.__doc__ = f.__doc__
            return _scaling_wrapper
    else:
        def _wrapper(f):
            def _warning_wrapper(*args, **kwargs):
                logging.warning("%s does not support dynamic loss scaling "
                                "in symbolic and hybridized execution.", f.__name__)
                return f(*args, **kwargs)
            _warning_wrapper.__name__ = f.__name__
            _warning_wrapper.__module__ = f.__module__
            _warning_wrapper.__doc__ = f.__doc__
            return _warning_wrapper

    for fun_name in list_loss_output_functions(target_dtype):
        try:
            f_to_wrap = getattr(module, fun_name)
            setattr(module, fun_name, _wrapper(f_to_wrap))
        except AttributeError:
            pass

_amp_initialized = False
_amp_loss_scale_initialized = False
_loss_scaler = None

@contextlib.contextmanager
def scale_loss(loss, optimizer_or_trainer):
    assert optimizer_or_trainer._amp_loss_scaler is not None, \
        'Loss scaler is not initialized, did you forget to call amp.init_trainer()?'
    optimizer_or_trainer._scale = (optimizer_or_trainer._amp_original_scale /
                                   optimizer_or_trainer._amp_loss_scaler.loss_scale)
    if isinstance(loss, (list, tuple)):
        yield [l * optimizer_or_trainer._amp_loss_scaler.loss_scale for l in loss]
    else:
        yield optimizer_or_trainer._amp_loss_scaler.loss_scale * loss

def warn_if_model_exists():
    for f in inspect.stack():
        for k, v in f.frame.f_locals.items():
            if isinstance(v, Block):
                logging.warning('Block %s created in [%s:%d] before AMP init.',
                                k, f.filename, f.lineno)
                return

def init(target_dtype='float16', target_precision_ops=None,
         conditional_fp32_ops=None, fp32_ops=None):
    """Initialize AMP (automatic mixed precision).

    This needs to be done before model creation.

    Parameters
    ----------
    target_dtype : {'float16', 'bfloat16'}
        Target low precision type for AMP. Currently only float16 and bfloat16 are supported.
    target_precision_ops : list of string
        Override the list of functions casted to target_dtype. Entries in this list
        are names of the functions casted to target_dtype.
    conditional_fp32_ops : list of (string, string, list of string)
        Override the list of functions conditionally casted to FP32. The format
        of the list is (name of the function, name of the parameter, list of
        values of the parameter that make the function be casted to FP32).
    fp32_ops : list of string
        Override the list of functions casted to FP32. Entries in this list
        are names of the functions casted to FP32.
    """
    global _amp_initialized
    global _loss_scaler
    if not _amp_initialized:
        assert target_dtype in ['float16', np.float16, 'bfloat16', bfloat16], \
               "AMP currently supports only float16 or bfloat16 as a target_dtype"
        _amp_initialized = True
        logging.info("Using AMP")
        if target_dtype == "bfloat16":
            target_dtype = bfloat16
        else:
            target_dtype = np.dtype(target_dtype)

        warn_if_model_exists()

        ops = get_all_registered_operators_grouped()
        get_aliases_nd = lambda l: [a for op in l for a in ops[op] if not base._is_np_op(a)]
        get_aliases_np = lambda l: [a for op in l for a in ops[op] if base._is_np_op(a)]
        get_aliases_np_pub = lambda l: [a for op in l for a in ops[op]
                                        if a.startswith(('_np_', '_npx_'))]
        get_cond_aliases_nd = lambda l: [(a, *rest) for op, *rest in l for a in ops[op]
                                         if not base._is_np_op(a)]
        get_cond_aliases_np = lambda l: [(a, *rest) for op, *rest in l for a in ops[op]
                                         if base._is_np_op(a)]
        get_cond_aliases_np_pub = lambda l: [(a, *rest) for op, *rest in l for a in ops[op]
                                             if a.startswith(('_np_', '_npx_'))]
        sy_submodules = {p:getattr(symbol, p[1:-1]) for p in base._OP_NAME_PREFIX_LIST}
        get_sy_fun = lambda fun, mod: _get_nd_fun_to_wrap(fun, mod, sy_submodules)
        nd_submodules = {p:getattr(ndarray, p[1:-1]) for p in base._OP_NAME_PREFIX_LIST}
        get_nd_fun = lambda fun, mod: _get_nd_fun_to_wrap(fun, mod, nd_submodules)
        get_np_sy_fun = lambda fun, mod: _get_np_fun_to_wrap(fun, "mxnet.symbol")
        get_np_nd_fun = lambda fun, mod: _get_np_fun_to_wrap(fun, "mxnet.ndarray")
        get_np_fun = lambda fun, mode: _get_np_fun_to_wrap(fun, "mxnet")
        todo = [
            (symbol, False, get_aliases_nd, get_cond_aliases_nd, get_sy_fun),
            (ndarray, False, get_aliases_nd, get_cond_aliases_nd, get_nd_fun),
            (symbol.numpy, True, get_aliases_np, get_cond_aliases_np, get_np_sy_fun),
            (ndarray.numpy, True, get_aliases_np, get_cond_aliases_np, get_np_nd_fun),
            (numpy, True, get_aliases_np_pub, get_cond_aliases_np_pub, get_np_fun),
        ]
        _loss_scaler = LossScaler()
        for module, is_numpy, get_aliases, get_cond_aliases, get_fun in todo:
            _wrap_module_functions(module, is_numpy, target_dtype, get_aliases, get_cond_aliases,
                                   get_fun, target_precision_ops, conditional_fp32_ops, fp32_ops)
            _wrap_loss_output_functions(module, _loss_scaler, target_dtype)

def init_trainer(optimizer_or_trainer):
    """Initialize trainer or optimizer to work with AMP dynamic loss scaling.

    Parameters
    ----------
    optimizer_or_trainer : Optimizer or Trainer
        MXNet Optimizer or Gluon trainer to initialize with AMP
    """
    global _amp_loss_scale_initialized
    global _amp_initialized
    global _loss_scaler
    assert _amp_initialized, "AMP not initialized, did you forget to call amp.init()?"
    if not _amp_loss_scale_initialized:
        _amp_loss_scale_initialized = True
        loss_scaler = _loss_scaler
    else:
        loss_scaler = LossScaler()
    #_wrap_output
    if isinstance(optimizer_or_trainer, trainer.Trainer):
        optimizer_or_trainer._amp_loss_scaler = loss_scaler
        optimizer_or_trainer._amp_original_scale = optimizer_or_trainer._scale
        trainer.Trainer.amp_loss_scale = property(lambda self: self._amp_loss_scaler.loss_scale)
    elif isinstance(optimizer_or_trainer, opt.Optimizer):
        raise TypeError("AMP is currently only compatible with Gluon Trainer")
    else:
        raise TypeError("optimizer_or_trainer should be a Gluon Trainer or "
                        "an optimizer, instead is %s" % type(optimizer_or_trainer))

def unscale(optimizer_or_trainer):
    """Check and unscale the gradients manually. This function should only be used
    if accessing gradients is necessary, e.g. for gradient clipping.

    Parameters
    ----------
    optimizer_or_trainer : Optimizer or Trainer
        MXNet optimizer or Gluon Trainer used when scaling the gradients
    """
    if isinstance(optimizer_or_trainer, trainer.Trainer):
        valid_grads = [p._grad for p in optimizer_or_trainer._params if p._grad is not None]
        for grads in valid_grads:
            # TODO(ptredak): make a bulked unscale
            for g in grads:
                g[:] *= optimizer_or_trainer._scale
        optimizer_or_trainer._scale = 1.
    elif isinstance(optimizer_or_trainer, opt.Optimizer):
        # TODO(ptredak): make it work with the optimizer
        raise TypeError("AMP is currently only compatible with Gluon Trainer")
    else:
        raise TypeError("optimizer_or_trainer should be a Gluon Trainer or "
                        "an optimizer, instead is %s" % type(optimizer_or_trainer))

def convert_symbol(sym, target_dtype="float16", target_dtype_ops=None,
                   fp32_ops=None, conditional_fp32_ops=None,
                   excluded_sym_names=None, data_names=None,
                   cast_optional_params=False):
    """Given a symbol object representing a neural network of data type FP32 and target_dtype,
    add cast layers according to the op lists (target_dtype_ops, fp32_ops,
    conditional_fp32_ops) if provided, otherwise use the default
    lists provided by the framework.

    Parameters
    ----------
    sym : Symbol
        FP32 neural network symbol
    target_dtype : str or numpy, optional defaults to float16
        currently only supports float16 and bfloat16. The target dtype indicates to add cast layers
        when possible so that lower precision computation can be leveraged.
    target_dtype_ops : list of strs, optional
        Override the list of operator names casted to the target_dtype.
        If None, uses the framework's default list to be casted to target_dtype.
    fp32_ops : list of strs, optional
        Override the list of operator names casted to FP32.
        If None, uses the framework's default list to be casted to FP32.
    conditional_fp32_ops : list of (string, string, list of string), optional
        Override the list of functions to be casted to FP32.
        The format of the list is
        (name of the function, name of the parameter,
         list of values of the parameter that make the operator to be casted to FP32)
    excluded_sym_names : list of strs, optional
        A list of strings that represent the names of symbols that users want to exclude
        from being casted to LP16 or FP32.
    data_names : list of strs, optional
        A list of strings that represent input data tensor names to the model
    cast_optional_params : bool, default False
        Whether to cast the arg_params and aux_params that don't require to be in LP16
        because of a cast layer following it, but will reduce the computation and memory
        overhead of the model if casted.
    """
    assert isinstance(sym, Symbol), "First argument to convert_symbol should be Symbol"

    assert target_dtype in ['float16', 'bfloat16'], \
               "Only target_dtype float16 and bfloat16 are supported currently"

    if target_dtype == 'bfloat16':
        target_dtype = bfloat16

    if target_dtype_ops is not None:
        assert isinstance(target_dtype_ops, list), "target_dtype_ops should be a list of strs"
    else:
        target_dtype_ops = list_lp16_ops(target_dtype)

    if fp32_ops is not None:
        assert isinstance(fp32_ops, list), "fp32_ops should be a list of strs"
    else:
        fp32_ops = list_fp32_ops(target_dtype)

    if conditional_fp32_ops is not None:
        assert isinstance(conditional_fp32_ops, list), "conditional_fp32_ops should be a list"
    else:
        conditional_fp32_ops = list_conditional_fp32_ops(target_dtype)

    original_conditional_op_names = []
    conditional_op_names = []
    param_names = []
    param_vals = []
    indptr = [0]
    for conditional_fp32_op in conditional_fp32_ops:
        assert isinstance(conditional_fp32_op[0], str) and isinstance(conditional_fp32_op[1], str) \
            and isinstance(conditional_fp32_op[2], list), "conditional_fp32_ops should be a list of " \
                                                          "(str, str, list of str)"
        param_vals += conditional_fp32_op[2]
        indptr.append(len(param_vals))
        param_names.append(conditional_fp32_op[1])
        conditional_op_names.append(conditional_fp32_op[0])

    if excluded_sym_names is not None:
        assert isinstance(excluded_sym_names, list), "excluded_sym_names should be a list of strs"
    else:
        excluded_sym_names = []

    for original_conditional_fp32_op in list_conditional_fp32_ops(target_dtype):
        original_conditional_op_names.append(original_conditional_fp32_op[0])

    # Op lists should not have intersection
    common_ops = set(target_dtype_ops) & set(fp32_ops)
    assert len(common_ops) == 0, "Ops cannot be in two or more lists. " \
                                 "Common ops in target_dtype_ops and fp32_ops {}".format(common_ops)
    common_ops = set(target_dtype_ops) & set(conditional_op_names)
    assert len(common_ops) == 0, "Ops cannot be in two or more lists. " \
                                 "Common ops in target_dtype_ops and conditional_fp32_ops {}".format(common_ops)
    common_ops = set(conditional_op_names) & set(fp32_ops)
    assert len(common_ops) == 0, "Ops cannot be in two or more lists. " \
                                 "Common ops in fp32_ops and conditional_fp32_ops {}".format(common_ops)

    combined_ops = set(target_dtype_ops + fp32_ops + conditional_op_names)
    all_lp16_fp32_ops = set(list_lp16_ops(target_dtype) + list_fp32_ops(target_dtype)
                            + list_lp16_fp32_ops(target_dtype) + original_conditional_op_names)

    illegal_ops = combined_ops - all_lp16_fp32_ops
    assert not illegal_ops, '''Can only choose ops from one of the three lists
                            for lp16_ops and fp32_ops
                            1. amp.list_lp16_ops(target_dtype)
                            2. amp.list_fp32_ops(target_dtype)
                            3. amp.list_lp16_fp32_ops(target_dtype)
                            4. amp.list_conditional_fp32_ops(target_dtype)
                            Op %s not in any of them''' % (illegal_ops)

    widest_dtype_ops = list_widest_type_cast(target_dtype)
    if target_dtype == bfloat16:
        target_dtype = _DTYPE_NP_TO_MX[bfloat16]
    else:
        target_dtype = _DTYPE_NP_TO_MX[np.dtype(target_dtype).type]

    # Prepare a data_names list based on list_inputs if its not provided
    # Add all names in list for the nodes in the symbol which don't have
    # __dtype__ set
    attr_dict = sym.attr_dict()
    if data_names is None:
        data_names = []
        for sym_name in sym.list_inputs():
            if not sym_name in attr_dict:
                data_names.append(sym_name)
                continue
            if not "__dtype__" in attr_dict[sym_name]:
                data_names.append(sym_name)
    model_param_names = list(set(sym.list_inputs()) - set(data_names))

    # Since assumption is that it is a FP32 model, set dtypes for all
    # data_names to float32
    str_keys = []
    sdata = []
    for k in data_names:
        str_keys.append(k)
        sdata.append(0)
    keys = c_str_array(str_keys)
    out = SymbolHandle()
    check_call(_LIB.MXReducePrecisionSymbol(sym.handle,
                                            ctypes.byref(out),
                                            mx_uint(len(sdata)),
                                            c_array_buf(ctypes.c_int, array('i', sdata)),
                                            mx_uint(len(indptr)),
                                            c_array_buf(ctypes.c_int, array('i', indptr)),
                                            ctypes.byref(ctypes.c_int(target_dtype)),
                                            ctypes.c_int(cast_optional_params),
                                            mx_uint(len(target_dtype_ops)),
                                            mx_uint(len(fp32_ops)),
                                            mx_uint(len(widest_dtype_ops)),
                                            mx_uint(len(conditional_op_names)),
                                            mx_uint(len(excluded_sym_names)),
                                            mx_uint(len(model_param_names)),
                                            c_str_array(target_dtype_ops),
                                            c_str_array(fp32_ops),
                                            c_str_array(widest_dtype_ops),
                                            c_str_array(conditional_op_names),
                                            c_str_array(excluded_sym_names),
                                            c_str_array(param_names),
                                            c_str_array(param_vals),
                                            c_str_array(model_param_names),
                                            keys))
    return Symbol(out)

def convert_model(sym, arg_params, aux_params, target_dtype="float16", target_dtype_ops=None,
                  fp32_ops=None, conditional_fp32_ops=None, excluded_sym_names=None,
                  cast_optional_params=False):
    """API for converting a model from FP32 model to a mixed precision model.
    MXNet tries to convert the FP32 model to mixed precision model by adding
    cast layers using amp_cast and amp_multicast operators which can be used for inference use cases.
    The decision on which cast layer to add is based on hardcoded lists for Automatic Mixed Precision
    in MXNet. These lists can be overridden by the user by providing their own lists
    using : targe_precision_ops, fp32_ops, widest_precision_ops, conditional_fp32_ops

    arg_params : dict
        Dictionary of name to `NDArray`.
    aux_params : dict
        Dictionary of name to `NDArray`.
    target_dtype : str
        Currently only supports float16 and bfloat 16. The target dtype indicates to add cast layers
        when possible so that lower precision computation can be leveraged.
    target_dtype_ops : list of strs
        Override the list of operator names casted to target_dtype.
        If None, uses the framework's default list to be casted to target dtype.
    fp32_ops : list of strs
        Override the lists of operator names casted to FP32.
        If None, uses the framework's default list to be casted to FP32.
    widest_dtype_ops : list of strs
        A list of op names provided by user which should run in widest precision among its inputs.
        If None, uses the framework's default list of widest_precision_ops.
    conditional_fp32_ops : list of (string, string, list of string)
        Override the list of operators to be casted to FP32.
        The format of the list is
        (name of the function, name of the parameter,
         list of values of the parameter that make the operator to be casted to
        fp32)
    excluded_sym_names : list of strs
        A list of strings that represent the names of symbols that users want to exclude
        from being executed in lower precision.
    cast_optional_params : bool, default False
        Whether to cast the arg_params and aux_params that don't require to be in LP16
        because of a cast layer following it, but will reduce the computation and memory
        overhead of the model if casted.
    """
    if excluded_sym_names is None:
        excluded_sym_names = []
        if not isinstance(excluded_sym_names, list):
            raise ValueError('excluded_sym_names must be a list of strings representing'
                             ' the names of the symbols that should not be casted,'
                             ' while received type %s' % str(type(excluded_sym_names)))
    assert target_dtype in ['float16', 'bfloat16'], \
               "Only target_dtype float16 and bfloat16 are supported currently"

    assert isinstance(sym, Symbol), "First argument to convert_model should be Symbol"
    assert isinstance(arg_params, dict), "Second argument to convert_model should be a dict of name to ndarray"
    assert isinstance(aux_params, dict), "Third argument to convert_model should be a dict of name to ndarray"

    param_names = list(arg_params.keys()) + list(aux_params.keys())

    # Only pass non params as data_names, param types can be inferred
    data_names = list(set(sym.list_inputs()) - set(param_names))
    sym = convert_symbol(sym, target_dtype, target_dtype_ops,
                         fp32_ops, conditional_fp32_ops,
                         excluded_sym_names, data_names,
                         cast_optional_params)

    # If dtype is set for params, cast the param to that dtype
    attr_dict = sym.attr_dict()
    for sym_name in sym.list_arguments():
        if sym_name in attr_dict and "__dtype__" in attr_dict[sym_name]:
            if attr_dict[sym_name]["__dtype__"] != "-1":
                typ = _DTYPE_MX_TO_NP[int(attr_dict[sym_name]["__dtype__"])]
                if typ == bfloat16:
                    arg_params[sym_name] = _cast_symbol_NDArray(arg_params[sym_name], bfloat16)
                else:
                    arg_params[sym_name] = arg_params[sym_name].astype(typ)

    for sym_name in sym.list_auxiliary_states():
        if sym_name in attr_dict and "__dtype__" in attr_dict[sym_name]:
            if attr_dict[sym_name]["__dtype__"] != "-1":
                typ = _DTYPE_MX_TO_NP[int(attr_dict[sym_name]["__dtype__"])]
                if typ == bfloat16:
                    aux_params[sym_name] = _cast_symbol_NDArray(aux_params[sym_name], bfloat16)
                else:
                    aux_params[sym_name] = aux_params[sym_name].astype(typ)

    # Return the converted symbol and casted params
    return sym, arg_params, aux_params

def convert_hybrid_block(block, target_dtype="float16", target_dtype_ops=None,
                         fp32_ops=None, conditional_fp32_ops=None,
                         excluded_sym_names=None, ctx=gpu(0),
                         cast_optional_params=False):
    """Given a hybrid block/symbol block representing a FP32 model and a target_dtype,
    return a block with mixed precision support which can be used for inference use cases.

    Parameters
    ----------
    block : HybridBlock or SymbolBlock object
        FP32 HybridBlock or SymbolBlock object
    target_dtype : str or numpy
        currently only supports float16 and bfloat16. The target dtype indicates to add cast layers
        when possible so that lower precision computation can be leveraged.
    target_precision_ops : list of strs
        Override the list of operator names casted to target_dtype.
        If None, uses the framework's default list to be casted to FP32.
    conditional_fp32_ops : list of (str, str, list of str)
        Override the list of functions to be casted to FP32.
        The format of the list is
        (name of the function, name of the parameter,
         list of values of the parameter that make the operator to be casted to FP32
    excluded_sym_names : list of strs
        A list of strings that represent the names of symbols that users want to exclude
        from being quantized
    ctx : Context
        Context on which model parameters should live
    cast_optional_params : bool, default False
        Whether to cast the arg_params and aux_params that don't require to be in LP16
        because of a cast layer following it, but will reduce the computation and memory
        overhead of the model if casted.
    """
    from ..gluon import HybridBlock, SymbolBlock
    assert isinstance(block, HybridBlock), "block input should be a HybridBlock"
    if not block._cached_graph:
        raise RuntimeError(
            "Please first call block.hybridize() and then run forward with "
            "this block at least once before calling export.")

    # Prepare inputs to pass to the convert_symbol API
    inputs, sym = block._cached_graph
    input_names = []
    for inp in inputs:
        input_names.append(inp.name)
    converted_sym = convert_symbol(sym, target_dtype, target_dtype_ops,
                                   fp32_ops, conditional_fp32_ops,
                                   excluded_sym_names, data_names=input_names,
                                   cast_optional_params=cast_optional_params)

    arg_names = set(converted_sym.list_arguments())
    aux_names = set(converted_sym.list_auxiliary_states())
    arg_dict = {}

    # If dtype for the param was set in the json, cast the
    # param to this dtype
    attr_dict = converted_sym.attr_dict()
    for param in block.collect_params().values():
        name = param.name
        if name in arg_names:
            arg_dict['arg:%s'%name] = param._reduce()
            if name in attr_dict and "__dtype__" in attr_dict[name]:
                if attr_dict[name]["__dtype__"] != "-1":
                    typ = _DTYPE_MX_TO_NP[int(attr_dict[name]["__dtype__"])]
                    if typ == bfloat16:
                        arg_dict['arg:%s' % name] = _cast_symbol_NDArray(arg_dict['arg:%s' % name], bfloat16)
                    else:
                        arg_dict['arg:%s'%name] = arg_dict['arg:%s'%name].astype(typ)
        else:
            assert name in aux_names
            arg_dict['aux:%s'%name] = param._reduce()
            if name in attr_dict and "__dtype__" in attr_dict[name]:
                if attr_dict[name]["__dtype__"] != "-1":
                    typ = _DTYPE_MX_TO_NP[int(attr_dict[name]["__dtype__"])]
                    if typ == bfloat16:
                        arg_dict['aux:%s' % name] = _cast_symbol_NDArray(arg_dict['aux:%s' % name], 'bfloat16')
                    else:
                        arg_dict['aux:%s'%name] = arg_dict['aux:%s'%name].astype(typ)

    # Create a symbolblock and cast the params to the dtypes based
    # on the dtype information from the converted_symbol
    ret = SymbolBlock(converted_sym, inputs)
    for key, param in ret.collect_params().items():
        arg_param_name = "arg:%s" % key
        if arg_param_name in arg_dict and param.dtype != arg_dict[arg_param_name].dtype:
            param.cast(arg_dict[arg_param_name].dtype)

        aux_param_name = "aux:%s" % key
        if aux_param_name in arg_dict and param.dtype != arg_dict[aux_param_name].dtype:
            param.cast(arg_dict[aux_param_name].dtype)

    ret.load_dict(arg_dict, ctx=ctx)
    return ret

def list_lp16_ops(target_dtype):
    """Get the default list of LP16 ops for AMP
    """
    if target_dtype in ['float16', np.float16]:
        return lists.symbol_fp16.FP16_FUNCS
    else:
        assert (target_dtype == bfloat16), "not supported type"
        return lists.symbol_bf16.BF16_FUNCS

def list_fp32_ops(target_dtype):
    """Get the default list of FP32 ops for AMP
    """
    if target_dtype in ['float16', np.float16]:
        return lists.symbol_fp16.FP32_FUNCS
    else:
        assert (target_dtype == bfloat16), "not supported type"
        return lists.symbol_bf16.FP32_FUNCS

def list_lp16_fp32_ops(target_dtype):
    """Get the default list of ops which run in both LP16 and FP32
    """
    if target_dtype in ['float16', np.float16]:
        return lists.symbol_fp16.FP16_FP32_FUNCS
    else:
        assert (target_dtype == bfloat16), "not supported type"
        return lists.symbol_bf16.BF16_FP32_FUNCS

def list_conditional_fp32_ops(target_dtype):
    """Get the conditional fp32 ops list
    """
    if target_dtype in ['float16', np.float16]:
        return lists.symbol_fp16.CONDITIONAL_FP32_FUNCS
    else:
        assert (target_dtype == bfloat16), "not supported type"
        return lists.symbol_bf16.CONDITIONAL_FP32_FUNCS

def list_widest_type_cast(target_dtype):
    """Get the widest type cast ops list
    """
    if target_dtype in ['float16', np.float16]:
        return lists.symbol_fp16.WIDEST_TYPE_CASTS
    else:
        assert (target_dtype == bfloat16), "not supported type"
        return lists.symbol_bf16.WIDEST_TYPE_CASTS

def list_loss_output_functions(target_dtype):
    """Get loss function list
    """
    if target_dtype in ['float16', np.float16]:
        return lists.symbol_fp16.LOSS_OUTPUT_FUNCTIONS
    else:
        assert (target_dtype == bfloat16), "not supported type"
        return lists.symbol_bf16.LOSS_OUTPUT_FUNCTIONS

def list_lp16_use_fp32_params(target_dtype):
    """ Get the params restrict for LP16

    """
    if target_dtype in ['float16', np.float16]:
        return None
    else:
        assert (target_dtype == bfloat16), "not supported type"
        return lists.symbol_bf16.BF16_USE_FP32_PARAMS
