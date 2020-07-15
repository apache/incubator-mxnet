#!/usr/bin/env python

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
"""MXNet: a concise, fast and flexible framework for deep learning."""

from .context import Context, current_context, cpu, gpu, cpu_pinned
from . import engine, error
from .base import MXNetError
from .util import is_np_shape, set_np_shape, np_shape, use_np_shape
from .util import is_np_array, np_array, use_np_array, use_np
from .util import is_np_default_dtype, np_default_dtype, use_np_default_dtype
from . import base

# version info
__version__ = base.__version__

from . import contrib
from . import ndarray
# use mx.nd as short for mx.ndarray
from . import ndarray as nd
from . import numpy
# use mx.np as short for mx.numpy
from . import numpy as np
from . import numpy_extension
# use mx.npx as short for mx.numpy_extension
from . import numpy_extension as npx
from . import name
# use mx.sym as short for mx.symbol
from . import symbol as sym
# use mx.np_symbol as short for mx.symbol.numpy
from .symbol.numpy import _symbol as np_symbol
from . import symbol
from . import symbol_doc
from . import io
from . import recordio
from . import operator
# use mx.rnd as short for mx.random
from . import random as rnd
from . import random
from . import optimizer
from . import model
from . import notebook
from . import initializer
# use mx.init as short for mx.initializer
from . import initializer as init
from . import visualization
# use mx.viz as short for mx.visualization
from . import visualization as viz
from . import callback
# from . import misc
from . import lr_scheduler
# Runtime compile module
from . import rtc
# Attribute scope to add attributes to symbolic graphs
from .attribute import AttrScope

from . import torch
# use mx.th as short for mx.torch
from . import torch as th

from . import profiler
from . import log

from . import image
# use mx.img as short for mx.image
from . import image as img

from . import test_utils

from . import gluon

from . import _deferred_compute

# With the native kvstore module (such as 'dist_sync_device'), the module launches a separate
# process when role is set to "server". This should be done after other modules are initialized.
# Otherwise this may result in errors when unpickling custom LR scheduler/optimizers.
# For example, the LRScheduler in gluoncv depends on a specific version of MXNet, and
# checks the __version__ attr of MXNet, which is not set on kvstore server due to the
# fact that kvstore-server module is imported before the __version__ attr is set.
# use mx.kv as short for mx.kvstore
from . import kvstore as kv

# Dynamic library module should be done after ndarray and symbol are initialized
from . import library
from . import tvmop

from . import numpy_op_signature
from . import numpy_dispatch_protocol
from . import numpy_op_fallback

from . import _global_var

from . import _api_internal
from . import api
from . import container

npx.set_np()
