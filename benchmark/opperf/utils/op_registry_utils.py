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

"""Utilities to interact with MXNet operator registry."""
import ctypes
import sys
from mxnet.base import _LIB, check_call, py_str, OpHandle, c_str, mx_uint

# We will use all operators inside NDArray Module
mx_nd_module = sys.modules["mxnet.ndarray.op"]


def _select_ops(operator_names, filters=("_contrib", "_"), merge_op_forward_backward=True):
    """From a given list of operators, filter out all operator names starting with given filters and prepares
    a dictionary of operator with attributes - 'has_backward' and 'nd_op_handle = mxnet.ndarray.op'

    By default, merge forward and backward operators for a given op into one operator and sets the attribute
    'has_backward' for the operator.

    By default, filter out all Contrib operators that starts with '_contrib' and internal operators that
    starts with '_'.

    Parameters
    ----------
    operator_names: List[str]
        List of operator names.
    filters: Tuple(str)
        Tuple of filters to apply on operator names.
    merge_op_forward_backward: Boolean, Default - True
        Merge forward and backward operators for a given op in to one op.

    Returns
    -------
    {"operator_name": {"has_backward", "nd_op_handle"}}
    """
    mx_operators = {}
    operators_with_backward = []

    if merge_op_forward_backward:
        filters += ("_backward",)

    for cur_op_name in operator_names:
        if not cur_op_name.startswith(filters):
            mx_operators[cur_op_name] = {"has_backward": False,
                                         "nd_op_handle": getattr(mx_nd_module, cur_op_name)}

        if cur_op_name.startswith("_backward_"):
            operators_with_backward.append(cur_op_name)

    if merge_op_forward_backward:
        # Identify all operators that can run backward.
        for op_with_backward in operators_with_backward:
            op_name = op_with_backward.split("_backward_")[1]
            if op_name in mx_operators:
                mx_operators[op_name]["has_backward"] = True

    return mx_operators


def _get_all_registered_ops():
    """Get all registered MXNet operator names.


    Returns
    -------
    ["operator_name"]
    """
    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()

    check_call(_LIB.MXListAllOpNames(ctypes.byref(size),
                                     ctypes.byref(plist)))

    mx_registered_operator_names = [py_str(plist[i]) for i in range(size.value)]
    return mx_registered_operator_names


def _get_op_handles(op_name):
    """Get handle for an operator with given name - op_name.

    Parameters
    ----------
    op_name: str
        Name of operator to get handle for.
    """
    op_handle = OpHandle()
    check_call(_LIB.NNGetOpHandle(c_str(op_name), ctypes.byref(op_handle)))
    return op_handle


def _get_op_arguments(op_handle):
    """Given operator name and handle, fetch operator arguments - number of arguments,
    argument names, argument types.

    Parameters
    ----------
    op_handle: OpHandle
        Handle for the operator

    Returns
    -------
    (narg, arg_names, arg_types)
    """
    real_name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = mx_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()
    key_var_num_args = ctypes.c_char_p()
    ret_type = ctypes.c_char_p()

    check_call(_LIB.MXSymbolGetAtomicSymbolInfo(
        op_handle, ctypes.byref(real_name), ctypes.byref(desc),
        ctypes.byref(num_args),
        ctypes.byref(arg_names),
        ctypes.byref(arg_types),
        ctypes.byref(arg_descs),
        ctypes.byref(key_var_num_args),
        ctypes.byref(ret_type)))

    narg = int(num_args.value)
    arg_names = [py_str(arg_names[i]) for i in range(narg)]
    arg_types = [py_str(arg_types[i]) for i in range(narg)]

    return narg, arg_names, arg_types


def _set_op_arguments(mx_operators):
    """Fetch and set operator arguments - nargs, arg_names, arg_types
    """
    for op_name in mx_operators:
        op_handle = _get_op_handles(op_name)
        narg, arg_names, arg_types = _get_op_arguments(op_handle)
        mx_operators[op_name]["params"] = {"narg": narg,
                                           "arg_names": arg_names,
                                           "arg_types": arg_types}


def _get_all_mxnet_operators():
    # Step 1 - Get all registered op names and filter it
    operator_names = _get_all_registered_ops()
    mx_operators = _select_ops(operator_names)

    # Step 2 - Get all parameters for the operators
    _set_op_arguments(mx_operators)
    return mx_operators


def prepare_op_inputs(arg_params, arg_values):
    # For each default value combination, prepare inputs
    inputs = []
    for arg_value in arg_values:
        inp = {}
        for arg_name in arg_params["params"]["arg_names"]:
            inp[arg_name] = arg_value[arg_name]
        inputs.append(inp)
    return inputs


def get_all_broadcast_binary_operators():
    """Gets all binary broadcast operators registered with MXNet.

    Returns
    -------
    {"operator_name": {"has_backward", "nd_op_handle", "params"}}
    """
    # Get all mxnet operators
    mx_operators = _get_all_mxnet_operators()

    # Filter for binary broadcast operators
    binary_broadcast_mx_operators = {}
    for op_name, op_params in mx_operators.items():
        if op_name.startswith("broadcast_") and op_params["params"]["narg"] == 2 and \
                "lhs" in op_params["params"]["arg_names"] and \
                "rhs" in op_params["params"]["arg_names"]:
            binary_broadcast_mx_operators[op_name] = mx_operators[op_name]
    return binary_broadcast_mx_operators


def get_all_elemen_wise_binary_operators():
    """Gets all binary elemen_wise operators registered with MXNet.

    Returns
    -------
    {"operator_name": {"has_backward", "nd_op_handle", "params"}}
    """
    # Get all mxnet operators
    mx_operators = _get_all_mxnet_operators()

    # Filter for binary elemen_wise operators
    binary_elemen_wise_mx_operators = {}
    for op_name, op_params in mx_operators.items():
        if op_name.startswith("elemwise_") and op_params["params"]["narg"] == 2 and \
                "lhs" in op_params["params"]["arg_names"] and \
                "rhs" in op_params["params"]["arg_names"]:
            binary_elemen_wise_mx_operators[op_name] = mx_operators[op_name]
    return binary_elemen_wise_mx_operators
