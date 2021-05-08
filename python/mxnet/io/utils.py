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

"""utility functions for io.py"""
from collections import OrderedDict

import warnings
import numpy as np
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import h5py
except ImportError:
    h5py = None

from ..ndarray.sparse import CSRNDArray
from ..ndarray.sparse import array as sparse_array
from ..ndarray import NDArray
from ..ndarray import array

def _init_data(data, allow_empty, default_name):
    """Convert data into canonical form."""
    assert (data is not None) or allow_empty
    if data is None:
        data = []

    if isinstance(data, (np.ndarray, NDArray, h5py.Dataset)
                  if h5py else (np.ndarray, NDArray)):
        data = [data]
    if isinstance(data, list):
        if not allow_empty:
            assert(len(data) > 0)
        if len(data) == 1:
            data = OrderedDict([(default_name, data[0])])  # pylint: disable=redefined-variable-type
        else:
            data = OrderedDict(  # pylint: disable=redefined-variable-type
                [('_%d_%s' % (i, default_name), d) for i, d in enumerate(data)])
    if not isinstance(data, dict):
        raise TypeError("Input must be NDArray, numpy.ndarray, h5py.Dataset " +
                        "a list of them or dict with them as values")
    for k, v in data.items():
        if not isinstance(v, (NDArray, h5py.Dataset) if h5py else NDArray):
            try:
                data[k] = array(v)
            except:
                raise TypeError(("Invalid type '%s' for %s, " % (type(v), k)) +
                                "should be NDArray, numpy.ndarray or h5py.Dataset")

    return list(sorted(data.items()))


def _has_instance(data, dtype):
    """Return True if ``data`` has instance of ``dtype``.
    This function is called after _init_data.
    ``data`` is a list of (str, NDArray)"""
    for item in data:
        _, arr = item
        if isinstance(arr, dtype):
            return True
    return False


def _getdata_by_idx(data, idx):
    """Shuffle the data."""
    shuffle_data = []

    for k, v in data:
        if (isinstance(v, h5py.Dataset) if h5py else False):
            shuffle_data.append((k, v))
        elif isinstance(v, CSRNDArray):
            shuffle_data.append((k, sparse_array(v.asscipy()[idx], v.context)))
        else:
            shuffle_data.append((k, array(v.asnumpy()[idx], v.context)))

    return shuffle_data
