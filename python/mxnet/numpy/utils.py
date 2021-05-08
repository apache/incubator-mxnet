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

"""Util functions for the numpy module."""



import numpy as onp

__all__ = ['float16', 'float32', 'float64', 'uint8', 'uint16', 'int16', 'int32', 'int8', 'int64',
           'bool', 'bool_', 'pi', 'inf', 'nan', 'PZERO', 'NZERO', 'newaxis', 'finfo',
           'e', 'NINF', 'PINF', 'NAN', 'NaN',
           '_STR_2_DTYPE_']

float16 = onp.float16
float32 = onp.float32
float64 = onp.float64
uint8 = onp.uint8
uint16 = onp.uint16
int16 = onp.int16
int32 = onp.int32
int8 = onp.int8
int64 = onp.int64
bool_ = onp.bool_
bool = onp.bool

pi = onp.pi
inf = onp.inf
nan = onp.nan
PZERO = onp.PZERO
NZERO = onp.NZERO
NINF = onp.NINF
PINF = onp.PINF
e = onp.e
NAN = onp.NAN
NaN = onp.NaN

newaxis = None
finfo = onp.finfo

_STR_2_DTYPE_ = {'float16': float16, 'float32': float32, 'float64':float64, 'float': float64,
                 'uint8': uint8, 'int8': int8, 'uint16': uint16, 'int16': int16, 'int32': int32,
                 'int64': int64, 'int': int64, 'bool': bool, 'bool_': bool_, 'None': None}

_ONP_OP_MODULES = [onp, onp.linalg, onp.random, onp.fft]


def _get_np_op(name):
    """Get official NumPy operator with `name`. If not found, raise ValueError."""
    for mod in _ONP_OP_MODULES:
        op = getattr(mod, name, None)
        if op is not None:
            return op
    raise ValueError('Operator `{}` is not supported by `mxnet.numpy`.'.format(name))
