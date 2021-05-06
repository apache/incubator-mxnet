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
#pylint: disable-msg=too-many-arguments

"""Exports an MXNet model to the ONNX model format"""
import logging
import numpy as np

from mxnet.base import string_types
from mxnet import symbol
from ._export_onnx import MXNetGraph
from ._export_helper import load_module


def get_operator_support(opset_version=None):
    """Return a list of MXNet operators supported by the current/specified opset
    """
    try:
        from onnx.defs import onnx_opset_version
    except ImportError:
        raise ImportError("Onnx and protobuf need to be installed. "
                          + "Instructions to install - https://github.com/onnx/onnx")
    if opset_version is None:
        opset_version = onnx_opset_version()
    all_versions = range(opset_version, 11, -1)
    ops = set()
    for ver in all_versions:
        if ver in MXNetGraph.registry_:
            ops.update(MXNetGraph.registry_[ver].keys())
    ops = list(ops)
    ops.sort()
    return ops


def export_model(sym, params, in_shapes=None, in_types=np.float32,
                 onnx_file_path='model.onnx', verbose=False, dynamic=False,
                 dynamic_input_shapes=None, run_shape_inference=False, input_type=None,
                 input_shape=None, model_specific_logics=None, cheat_sheet=None):
    """Exports the MXNet model file, passed as a parameter, into ONNX model.
    Accepts both symbol,parameter objects as well as json and params filepaths as input.
    Operator support and coverage -
    https://github.com/apache/incubator-mxnet/tree/v1.x/python/mxnet/onnx#operator-support-matrix

    Parameters
    ----------
    sym : str or symbol object
        Path to the json file or Symbol object
    params : str or dict or list of dict
        str - Path to the params file
        dict - params dictionary (Including both arg_params and aux_params)
        list - list of length 2 that contains arg_params and aux_params
    in_shapes : List of tuple
        Input shape of the model e.g [(1,3,224,224)]
    in_types : data type or list of data types
        Input data type e.g. np.float32, or [np.float32, np.int32]
    onnx_file_path : str
        Path where to save the generated onnx file
    verbose : Boolean
        If True will print logs of the model conversion
    dynamic: Boolean
        If True will allow for dynamic input shapes to the model
    dynamic_input_shapes: list of tuple
        Specifies the dynamic input_shapes. If None then all dimensions are set to None
    run_shape_inference : Boolean
        If True will run shape inference on the model
    input_type : data type or list of data types
        This is the old name of in_types. We keep this parameter name for backward compatibility
    in_shapes : List of tuple
        This is the old name of in_shapes. We keep this parameter name for backward compatibility
    model_specific_logics : str
        Specifies if model-specific conversion logic should be used. Refer to ./_op_translations/
    cheat_sheet : dict of str to str
        This is a dict that stors some hyperparameters values or additional info about the model that
        would be used in model-specific conversion functions

    Returns
    -------
    onnx_file_path : str
        Onnx file path

    Notes
    -----
    This method is available when you ``import mxnet.onnx``

    """

    try:
        from onnx import helper, mapping, shape_inference
        from onnx.defs import onnx_opset_version
    except ImportError:
        raise ImportError("Onnx and protobuf need to be installed. "
                          + "Instructions to install - https://github.com/onnx/onnx")

    if input_type is not None:
        in_types = input_type

    if input_shape is not None:
        in_shapes = input_shape

    converter = MXNetGraph()
    opset_version = onnx_opset_version()

    if not isinstance(in_types, list):
        in_types = [in_types for _ in range(len(in_shapes))]
    in_types_t = [mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(i_t)] for i_t in in_types]
    assert len(in_types) == len(in_shapes), "The lengths of in_types and in_shapes must equal"
    # if input parameters are strings(file paths), load files and create symbol parameter objects
    if isinstance(sym, string_types) and isinstance(params, string_types):
        logging.info("Converting json and weight file to sym and params")
        sym_obj, params_obj = load_module(sym, params)
        onnx_graph = converter.create_onnx_graph_proto(sym_obj, params_obj, in_shapes,
                                                       in_types_t,
                                                       verbose=verbose, opset_version=opset_version,
                                                       dynamic=dynamic, dynamic_input_shapes=dynamic_input_shapes,
                                                       model_specific_logics=model_specific_logics,
                                                       cheat_sheet=cheat_sheet)
    elif isinstance(sym, symbol.Symbol) and isinstance(params, dict):
        onnx_graph = converter.create_onnx_graph_proto(sym, params, in_shapes,
                                                       in_types_t,
                                                       verbose=verbose, opset_version=opset_version,
                                                       dynamic=dynamic, dynamic_input_shapes=dynamic_input_shapes,
                                                       model_specific_logics=model_specific_logics,
                                                       cheat_sheet=cheat_sheet)
    elif isinstance(sym, symbol.Symbol) and isinstance(params, list) and len(params) == 2:
        # when params contains arg_params and aux_params
        p = {}
        p.update(params[0])
        p.update(params[1])
        onnx_graph = converter.create_onnx_graph_proto(sym, p, in_shapes,
                                                       in_types_t,
                                                       verbose=verbose, opset_version=opset_version,
                                                       dynamic=dynamic, dynamic_input_shapes=dynamic_input_shapes)
    else:
        raise ValueError("Input sym and params should either be files or objects")

    # Create the model (ModelProto)
    onnx_model = helper.make_model(onnx_graph)

    # Run shape inference on the model. Due to ONNX bug/incompatibility this may or may not crash
    if run_shape_inference:
        try:
            onnx_model = shape_inference.infer_shapes(onnx_model)
        except: # pylint: disable=bare-except
            logging.info("Shape inference failed, original export is kept.")

    # Save model on disk
    with open(onnx_file_path, "wb") as file_handle:
        serialized = onnx_model.SerializeToString()
        file_handle.write(serialized)
        logging.info("Input shape of the model %s ", in_shapes)
        logging.info("Exported ONNX file %s saved to disk", onnx_file_path)

    return onnx_file_path
