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

from .... import ndarray as nd


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
    def register(op_name):
        """Register operators"""
        def wrapper(func):
            """Helper function to map functions"""
            try:
                import onnx as _
                MXNetGraph.registry_[op_name] = func
            except ImportError:
                pass
            return func

        return wrapper

    @staticmethod
    def convert_layer(node, **kwargs):
        """Convert MXNet layer to ONNX"""
        op = str(node["op"])
        if op not in MXNetGraph.registry_:
            raise AttributeError("No conversion function registered for op type %s yet." % op)
        convert_func = MXNetGraph.registry_[op]
        return convert_func(node, **kwargs)

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
    def get_outputs(sym, params, in_shape, in_label):
        """ Infer output shapes and return dictionary of output name to shape

        :param :class:`~mxnet.symbol.Symbol` sym: symbol to perform infer shape on
        :param dic of (str, nd.NDArray) params:
        :param list of tuple(int, ...) in_shape: list of all input shapes
        :param  in_label: name of label typically used in loss that may be left in graph. This name is
            removed from list of inputs required by symbol
        :return: dictionary of output name to shape
        :rtype: dict of (str, tuple(int, ...))
        """
        # remove any input listed in params from sym.list_inputs() and bind them to the input shapes provided
        # by user. Also remove in_label, which is the name of the label symbol that may have been used
        # as the label for loss during training.
        inputs = {n: tuple(s) for n, s in zip([n for n in sym.list_inputs() if n not in params and n != in_label],
                                              in_shape)}
        # Add params and their shape to list of inputs
        inputs.update({n: v.shape for n, v in params.items() if n in sym.list_inputs()})
        # Provide input data as well as input params to infer_shape()
        _, out_shapes, _ = sym.infer_shape(**inputs)

        out_names = list()
        for name in sym.list_outputs():
            if name.endswith('_output'):
                out_names.append(name[:-len('_output')])
            else:
                logging.info("output '%s' does not end with '_output'", name)
                out_names.append(name)

        assert len(out_shapes) == len(out_names)
        # bind output shapes with output names
        graph_outputs = {n: s for n, s in zip(out_names, out_shapes)}

        return graph_outputs

    @staticmethod
    def convert_weights_to_numpy(weights_dict):
        """Convert weights to numpy"""
        return dict([(k.replace("arg:", "").replace("aux:", ""), v.asnumpy())
                     for k, v in weights_dict.items()])

    def create_onnx_graph_proto(self, sym, params, in_shape, in_type, verbose=False):
        """Convert MXNet graph to ONNX graph

        Parameters
        ----------
        sym : :class:`~mxnet.symbol.Symbol`
            MXNet symbol object
        params : dict of ``str`` to :class:`~mxnet.ndarray.NDArray`
            Dict of converted parameters stored in ``mxnet.ndarray.NDArray`` format
        in_shape : List of tuple
            Input shape of the model e.g [(1,3,224,224)]
        in_type : data type
            Input data type e.g. np.float32
        verbose : Boolean
            If true will print logs of the model conversion

        Returns
        -------
        graph : GraphProto
            ONNX graph
        """
        try:
            from onnx import (checker, helper, NodeProto, ValueInfoProto, TensorProto)
            from onnx.helper import make_tensor_value_info
        except ImportError:
            raise ImportError("Onnx and protobuf need to be installed. "
                              + "Instructions to install - https://github.com/onnx/onnx")

        # When MXNet model is saved to json file , MXNet adds a node for label.
        # The name of this node is, name of the last node + "_label" ( i.e if last node
        # name is "Softmax", this node will have a name "Softmax_label". Also, the new node
        # will always be second last node in the json graph.
        # Deriving the output_label name.
        output_label = sym.get_internals()[len(sym.get_internals()) - 1].name + "_label"

        weights = MXNetGraph.convert_weights_to_numpy(params)

        mx_graph = json.loads(sym.tojson())["nodes"]

        initializer = []
        all_processed_nodes = []
        onnx_processed_nodes = []
        onnx_processed_inputs = []
        onnx_processed_outputs = []
        index_lookup = []

        # Determine output shape
        graph_outputs = MXNetGraph.get_outputs(sym, params, in_shape, output_label)

        if verbose:
            for node in mx_graph:
                op = str(node['op'])
                if op not in MXNetGraph.registry_:
                    logging.warning("No conversion function registered for op type %s yet.", op)


        graph_input_idx = 0
        for idx, node in enumerate(mx_graph):
            op = node["op"]
            name = node["name"]
            if verbose:
                logging.info("Converting idx: %d, op: %s, name: %s", idx, op, name)

            # A node is an input node if its op_name is "null" and is not
            # in params dict
            if op == "null" and name not in params:
                # Handling graph input

                # Skipping output_label node, as this node is not part of graph
                # Refer "output_label" assignment above for more details.
                if name == output_label:
                    continue
                converted = MXNetGraph.convert_layer(
                    node,
                    is_input=True,
                    mx_graph=mx_graph,
                    weights=weights,
                    in_shape=in_shape[graph_input_idx],
                    in_type=in_type,
                    proc_nodes=all_processed_nodes,
                    initializer=initializer,
                    index_lookup=index_lookup)
                graph_input_idx += 1

            else:
                # Handling graph layers
                converted = MXNetGraph.convert_layer(
                    node,
                    is_input=False,
                    mx_graph=mx_graph,
                    weights=weights,
                    in_shape=in_shape,
                    in_type=in_type,
                    proc_nodes=all_processed_nodes,
                    initializer=initializer,
                    index_lookup=index_lookup,
                    idx=idx
                )

            if isinstance(converted, list):
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
                            if nodename in graph_outputs:
                                onnx_processed_outputs.append(
                                    make_tensor_value_info(
                                        name=nodename,
                                        elem_type=in_type,
                                        shape=graph_outputs[nodename]
                                    )
                                )
                                if verbose:
                                    logging.info("Output node is: %s", nodename)
                    elif isinstance(converted_node, TensorProto):
                        raise ValueError("Did not expect TensorProto")
                    else:
                        raise ValueError("node is of an unrecognized type: %s" % type(node))

                    all_processed_nodes.append(converted_node)

                if idx > 0:
                    # Handling extra node added to the graph if the MXNet model was
                    # saved to json file,
                    # refer "output_label" initialization above for more details.
                    # if extra node was added then prev_index to the last node is adjusted.
                    if idx == (len(mx_graph) - 1) and \
                            mx_graph[len(mx_graph)-2]["name"] == output_label:
                        prev_index = index_lookup[idx - 2]
                    else:
                        prev_index = index_lookup[idx - 1]

                    index_lookup.append(prev_index+len(converted))
                else:
                    index_lookup.append(len(converted) - 1)
            else:
                logging.info("Operator converter function should always return a list")

        graph = helper.make_graph(
            onnx_processed_nodes,
            "mxnet_converted_model",
            onnx_processed_inputs,
            onnx_processed_outputs
        )

        graph.initializer.extend(initializer)

        checker.check_graph(graph)
        return graph
