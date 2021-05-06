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
#
# Based on
# https://github.com/NVIDIA/mxnet_to_onnx/blob/master/mx2onnx_converter/mx2onnx_converter.py#
#  Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
#  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#  PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
#  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# coding: utf-8
# pylint: disable=invalid-name,too-many-locals,no-self-use,too-many-arguments,
# pylint: disable=maybe-no-member,too-many-nested-blocks
"""MXNet to ONNX graph converter functions"""
import logging
import json

import numpy as np
from mxnet import ndarray as nd


class MXNetGraph(object):
    """Class to convert MXNet to ONNX graph"""
    registry_ = {}
    input_output_maps_ = {}

    def __init__(self):
        # topologically sorted nodes
        self.nodes = []
        self.input_tensors = []
        self.output_tensors = []

    @staticmethod
    def register(op_name, opset_version=12):
        """Register operators"""
        def wrapper(func):
            """Helper function to map functions"""
            try:
                import onnx as _
                op_map = MXNetGraph.registry_.setdefault(opset_version, {})
                op_map[op_name] = func
            except ImportError:
                pass
            return func

        return wrapper

    @staticmethod
    def convert_layer(node, **kwargs):
        """Convert MXNet layer to ONNX"""
        try:
            from onnx.defs import onnx_opset_version
        except ImportError:
            raise ImportError("Onnx and protobuf need to be installed. "
                              + "Instructions to install - https://github.com/onnx/onnx")

        op = str(node["op"])
        opset_version = kwargs.get("opset_version", onnx_opset_version())
        # fallback to older opset versions if op is not registered in current version
        convert_func = None
        for op_version in range(opset_version, 11, -1):
            if op_version not in MXNetGraph.registry_ or op not in MXNetGraph.registry_[op_version]:
                continue
            convert_func = MXNetGraph.registry_[op_version][op]
            break

        model_specific_logics = kwargs.get("model_specific_logics", None)
        if model_specific_logics:
            assert model_specific_logics in MXNetGraph.registry_,\
                "Model specific converion logics for %s is not found" % model_specific_logics
            if op in  MXNetGraph.registry_[model_specific_logics]:
                logging.info("Found model-specific %s conversion logic for model %s",
                             op, model_specific_logics)
                convert_func = MXNetGraph.registry_[model_specific_logics][op]

        if convert_func is None:
            raise AttributeError("No conversion function registered for op type %s yet." % op)

        ret = convert_func(node, **kwargs)
        # in case the conversion function does not specify the returned dtype, we just return None
        # as the second value
        if isinstance(ret, list):
            return ret, None
        else:
            return ret

    @staticmethod
    def split_params(sym, params):
        """Helper function to split params dictionary into args and aux params

        Parameters
        ----------
        sym : :class:`~mxnet.symbol.Symbol`
            MXNet symbol object
        params : dict of ``str`` to :class:`~mxnet.ndarray.NDArray`
            Dict of converted parameters stored in ``mxnet.ndarray.NDArray`` format

        Returns
        -------
        arg_params : dict of ``str`` to :class:`~mxnet.ndarray.NDArray`
            Dict of converted parameters stored in ``mxnet.ndarray.NDArray`` format
        aux_params : dict of ``str`` to :class:`~mxnet.ndarray.NDArray`
            Dict of converted parameters stored in ``mxnet.ndarray.NDArray`` format
        """
        arg_params = {}
        aux_params = {}
        for args in sym.list_arguments():
            if args in params:
                arg_params.update({args: nd.array(params[args])})
        for aux in sym.list_auxiliary_states():
            if aux in params:
                aux_params.update({aux: nd.array(params[aux])})
        return arg_params, aux_params

    @staticmethod
    def get_outputs(sym, params, in_shapes, output_label, in_types, dynamic=False,
                    dynamic_input_shapes=None):
        """Helper function to collect the output names, types, and shapes

        Parameters
        ----------
        sym : :class:`~mxnet.symbol.Symbol`
            MXNet symbol object
        params : dict of ``str`` to :class:`~mxnet.ndarray.NDArray`
            Dict of converted parameters stored in ``mxnet.ndarray.NDArray`` format
        in_shapes : list of tuple
            Input shapes
        out_label : ``str``
            Name of label typically used in loss that may be left in graph. This name is
            removed from list of inputs required by symbol
        in_types : list of Int
            Input ONNX data types
        dynamic : Boolean
            If True will allow for dynamic input shapes to the model
        dynamic_input_shapes: list of tuple
            Specifies the dynamic input_shapes. If None then all dimensions are set to None

        Returns
        in_shapes : list of tuple
            Updated input shapes
        graph_outputs : dict ``str`` to dict
            This maps output name to {'shape':tuple, 'dtype':Int}
        -------
        """
        from onnx import mapping
        import re

        # Collect graph output names
        out_names = list()
        for name in sym.list_outputs():
            if name.endswith('_state_output'): # handel special cases for RNN operator
                out_names.append(name[:-len('_state_output')]+'1')
            elif name.endswith('_statecell_output'): # handel special cases for RNN operator
                out_names.append(name[:-len('_statecell_output')]+'2')
            elif name.endswith('_output'):
                out_names.append(name[:-len('_output')])
            elif re.search('.*_output[0-9]$', name):
                out_names.append(name[:-len('_output0')]+name[-1])
            else:
                logging.info("output '%s' does not end with '_output'", name)
                out_names.append(name)

        # Collect graph output shapes
        # Remove any input listed in params from sym.list_inputs() and bind them to the input shapes provided
        # by user. Also remove output_label, which is the name of the label symbol that may have been used
        # as the label for loss during training.
        inputs = {n: tuple(s) for n, s in
                  zip([n for n in sym.list_inputs() if n not in params and n != output_label],
                      in_shapes)}
        # Add params and their shape to list of inputs
        inputs.update({n: v.shape for n, v in params.items() if n in sym.list_inputs()})
        # Provide input data as well as input params to infer_shape()
        _, out_shapes, _ = sym.infer_shape(**inputs)
        if dynamic:
            # Keep the dimensionality of the output shapes but change the values to None
            out_shapes = [tuple(None for _ in i_s) for i_s in out_shapes]

            if dynamic_input_shapes is None:
                # Set all dimensions to None
                in_shapes = [tuple(None for _ in i_s) for i_s in in_shapes]
            else:
                assert len(in_shapes) == len(dynamic_input_shapes), "The length of " \
                    "dynamic_input_shapes must equal to the length of in_shapes."
                for i_s, d_i_s in zip(in_shapes, dynamic_input_shapes):
                    assert len(i_s) == len(d_i_s), "The dimensionality " \
                        "of each shape must match."
                in_shapes = dynamic_input_shapes
        else:
            assert dynamic_input_shapes is None, "dynamic_input_shapes is specified. Please " \
                "set dynamic_input_shapes=True to enable dynamic input shapes"

        # Collect graph output types
        # Remove any input listed in params from sym.list_inputs() and bind them to the input types provided
        # by user. Also remove output_label
        in_dtypes = {n: mapping.TENSOR_TYPE_TO_NP_TYPE[t] for n, t in
                     zip([n for n in sym.list_inputs() if n not in params and n != output_label],
                         in_types)}
        # Add params and their types to list of inputs
        in_dtypes.update({n: v.dtype for n, v in params.items() if n in sym.list_inputs()})
        _, out_type, _ = sym.infer_type(**in_dtypes)
        out_types = [mapping.NP_TYPE_TO_TENSOR_TYPE[o(0).dtype] for o in out_type]

        # Make sure the types, names, and shapes all align up
        assert len(out_types) == len(out_names) == len(out_shapes)

        # Bind output shapes/types with output names
        graph_outputs = {n: {'shape': s, 'dtype': d} for n, s, d in zip(out_names, out_shapes, out_types)}

        return in_shapes, graph_outputs

    @staticmethod
    def convert_weights_to_numpy(weights_dict):
        """Convert weights to numpy"""
        return dict([(k.replace("arg:", "").replace("aux:", ""), v.asnumpy())
                     for k, v in weights_dict.items()])

    def create_onnx_graph_proto(self, sym, params, in_shapes, in_types, verbose=False,
                                opset_version=None, dynamic=True, dynamic_input_shapes=None,
                                model_specific_logics=None, cheat_sheet=None):
        """Convert MXNet graph to ONNX graph

        Parameters
        ----------
        sym : :class:`~mxnet.symbol.Symbol`
            MXNet symbol object
        params : dict of ``str`` to :class:`~mxnet.ndarray.NDArray`
            Dict of converted parameters stored in ``mxnet.ndarray.NDArray`` format
        in_shapes : List of tuple
            Input shape of the model e.g [(1,3,224,224)]
        in_types : List of Int
            Input ONNX data types
        verbose : Boolean
            If true will print logs of the model conversion
        opset_version : Int
            ONNX opset version to use for export, defaults to latest supported by onnx package
        dynamic: Boolean
            If True will allow for dynamic input shapes to the model
        dynamic_input_shapes: list of tuple
            Specifies the dynamic input_shapes. If None then all dimensions are set to None
        model_specific_logics: str
            Specifies if model-specific conversion logic should be used. Refer to ./_op_translations/
        cheat_sheet : dict of str to str
            This is a dict that stors some hyperparameters values or additional info about the model that
            would be used in model-specific conversion functions

        Returns
        -------
        graph : GraphProto
            ONNX graph
        """
        try:
            from onnx import (checker, helper, NodeProto, ValueInfoProto, TensorProto)
            from onnx.helper import make_tensor_value_info
            from onnx.defs import onnx_opset_version
        except ImportError:
            raise ImportError("Onnx and protobuf need to be installed. "
                              + "Instructions to install - https://github.com/onnx/onnx")

        if opset_version is None:
            opset_version = onnx_opset_version()

        # When MXNet model is saved to json file , MXNet adds a node for label.
        # The name of this node is, name of the last node + "_label" ( i.e if last node
        # name is "Softmax", this node will have a name "Softmax_label". Also, the new node
        # will always be second last node in the json graph.
        # Deriving the output_label name.
        output_label = sym.get_internals()[len(sym.get_internals()) - 1].name + "_label"

        weights = MXNetGraph.convert_weights_to_numpy(params)

        mx_graph = json.loads(sym.tojson())["nodes"]

        class NodeOutput:
            def __init__(self, name, dtype):
                self.name = name
                self.dtype = np.dtype(dtype)

        initializer = []
        all_processed_nodes = []
        onnx_processed_nodes = []
        onnx_processed_inputs = []
        onnx_processed_outputs = []
        outputs_lookup = []

        # Determine graph output names, shapes, and dtypes. Also update in_shapes
        in_shapes, graph_outputs = MXNetGraph.get_outputs(sym, params, in_shapes, output_label,
                                                          in_types, dynamic, dynamic_input_shapes)
        appeared_names = set()
        graph_input_idx = 0
        for idx, node in enumerate(mx_graph):
            op = node["op"]
            # check if the current node has the same name as nodes before
            if node["name"] in appeared_names:
                node["name"] = 'idx_' + str(idx) + '_' + node["name"]
            else:
                appeared_names.add(node["name"])
            name = node["name"]
            if verbose:
                logging.info("Converting idx: %d, op: %s, name: %s", idx, op, name)

            # A node is an input node if its op_name is "null" and is not
            # in params dict
            if op == "null" and name not in params:
                # Handle graph input

                # Skip output_label node, as this node is not part of graph
                # Refer to "output_label" assignment above for more details.
                if name == output_label:
                    continue

                converted, dtypes = MXNetGraph.convert_layer(
                    node,
                    is_input=True,
                    mx_graph=mx_graph,
                    weights=weights,
                    in_shape=in_shapes[graph_input_idx],
                    in_type=in_types[graph_input_idx],
                    proc_nodes=all_processed_nodes,
                    initializer=initializer,
                    outputs_lookup=outputs_lookup,
                )
                graph_input_idx += 1
            else:
                # Handle graph layers
                converted, dtypes = MXNetGraph.convert_layer(
                    node,
                    is_input=False,
                    mx_graph=mx_graph,
                    weights=weights,
                    proc_nodes=all_processed_nodes,
                    initializer=initializer,
                    outputs_lookup=outputs_lookup,
                    idx=idx,
                    opset_version=opset_version,
                    model_specific_logics=model_specific_logics,
                    cheat_sheet=cheat_sheet
                )
            if isinstance(converted, list):
                # Collect all the node's output names
                node_possible_names = [name] + [name + str(i) for i in range(10)]
                node_output_names = []
                # Collect all the graph's output names
                graph_output_names = []
                # Iterate for all converted nodes
                for converted_node in converted:
                    # If converted node is ValueInfoProto, add it in inputs
                    if isinstance(converted_node, ValueInfoProto):
                        onnx_processed_inputs.append(converted_node)
                    # If converted node is NodeProto, add it in processed nodes list
                    elif isinstance(converted_node, NodeProto):
                        onnx_processed_nodes.append(converted_node)
                        # some operators have multiple outputs,
                        # therefore, check all output node names
                        node_names = list(converted_node.output)
                        for nodename in node_names:
                            if nodename in node_possible_names:
                                node_output_names.append(nodename)
                            if nodename in graph_outputs:
                                graph_output_names.append(nodename)
                                if verbose:
                                    logging.info("Output node is: %s", nodename)
                    elif isinstance(converted_node, TensorProto):
                        raise ValueError("Did not expect TensorProto")
                    else:
                        raise ValueError("node is of an unrecognized type: %s" % type(node))

                    all_processed_nodes.append(converted_node)

                # if node_output_names is empty then we use the last returned node as output
                if not node_output_names:
                    node_output_names = [converted[-1].name]
                # process node outputs (sort by alphabetical order)
                node_output_names.sort()
                # match the output names to output dtypes
                if dtypes is not None:
                    assert len(node_output_names) == len(dtypes)
                    node_outputs = [NodeOutput(node_output_names[i], dtypes[i])
                                    for i in range(len(dtypes))]
                else:
                    # in case dtypes is None, we just default to the dtype of the first input
                    assert len(node["inputs"]) > 0
                    first_input = node["inputs"][0]
                    first_input_dtype = outputs_lookup[first_input[0]][first_input[1]].dtype
                    node_outputs = [NodeOutput(n, first_input_dtype)
                                    for n in node_output_names]
                outputs_lookup.append(node_outputs)

                # process graph outputs (sort by alphabetical order)
                graph_output_names.sort()
                for nodename in graph_output_names:
                    onnx_processed_outputs.append(
                        make_tensor_value_info(
                            name=nodename,
                            elem_type=graph_outputs[nodename]['dtype'],
                            shape=graph_outputs[nodename]['shape']
                        )
                    )

            else:
                logging.info("Operator converter function should always return a list")

        # sometimes the graph output can also be in the intializer
        for i in initializer:
            if i.name in graph_outputs:
                onnx_processed_outputs.append(
                    make_tensor_value_info(
                        name=i.name,
                        elem_type=graph_outputs[i.name]['dtype'],
                        shape=graph_outputs[i.name]['shape']
                    )
                )

        graph = helper.make_graph(
            onnx_processed_nodes,
            "mxnet_converted_model",
            onnx_processed_inputs,
            onnx_processed_outputs
        )

        graph.initializer.extend(initializer)

        checker.check_graph(graph)
        return graph
