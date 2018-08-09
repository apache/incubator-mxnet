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

""" Module to enable the use of TensorRT optimized graphs."""

import ctypes
import logging
import os

from mxnet.symbol import Symbol

from ..base import _LIB, SymbolHandle, MXNetError
from ..base import check_call


def set_use_tensorrt(status):
    """
    Set an environment variable which will enable or disable the use of TensorRT in the backend.
    Note: this is useful for A/B testing purposes.
    :param status: Boolean, true if TensorRT optimization should be applied, False for legacy
    behaviour.
    """
    os.environ["MXNET_USE_TENSORRT"] = str(int(status))


def get_use_tensorrt():
    """
    Get an environment variable which describes if TensorRT is currently enabled in the backend.
    Note: this is useful for A/B testing purposes.
    :return: Boolean, true if TensorRT optimization should be applied, False for legacy
    behaviour.
    """
    return bool(int(os.environ.get("MXNET_USE_TENSORRT", 0)) == 1)


def get_optimized_symbol(executor):
    """Get optimized symbol.

    Parameters
    ----------
    executor :
        An executor for which you want to see an optimized symbol.  Getting an optimized symbol
        is useful to compare and verify the work TensorRT has done against a legacy behaviour.

    Returns
    -------
    symbol : nnvm::Symbol
        The nnvm symbol optimized.
    """
    handle = SymbolHandle()
    try:
        check_call(_LIB.MXExecutorGetOptimizedSymbol(executor.handle, ctypes.byref(handle)))
        result = Symbol(handle=handle)
        return result
    except MXNetError:
        logging.error('Error while trying to fetch TRT optimized symbol for graph.  Please ensure'
                      'build was compiled with MXNET_USE_TENSORRT enabled.')
        raise
