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
# pylint: disable=invalid-name, too-many-locals, fixme
# pylint: disable=too-many-branches, too-many-statements
# pylint: disable=too-many-arguments
# pylint: disable=dangerous-default-value
"""Visualization module"""

import re
import copy
import json
import warnings
from .symbol import Symbol

def _str2tuple(string):
    """Convert shape string to list, internal use only.

    Parameters
    ----------
    string: str
        Shape string.

    Returns
    -------
    list of str
        Represents shape.
    """
    return re.findall(r"\d+", string)

def print_summary(symbol, shape=None, line_length=120, positions=[.44, .64, .74, 1.]):
    """Convert symbol for detail information.

    Parameters
    ----------
    symbol: Symbol
        Symbol to be visualized.
    shape: dict
        A dict of shapes, str->shape (tuple), given input shapes.
    line_length: int
        Rotal length of printed lines
    positions: list
        Relative or absolute positions of log elements in each line.

    Returns
    ------
    None

    Notes
    -----
    If ``mxnet`` is imported, the visualization module can be used in its short-form.
    For example, if we ``import mxnet`` as follows::

        import mxnet

    this method in visualization module can be used in its short-form as::

        mxnet.viz.print_summary(...)

    """
    if not isinstance(symbol, Symbol):
        raise TypeError("symbol must be Symbol")
    show_shape = False
    if shape is not None:
        show_shape = True
        interals = symbol.get_internals()
        _, out_shapes, _ = interals.infer_shape(**shape)
        if out_shapes is None:
            raise ValueError("Input shape is incomplete")
        shape_dict = dict(zip(interals.list_outputs(), out_shapes))
    conf = json.loads(symbol.tojson())
    nodes = conf["nodes"]
    heads = set(conf["heads"][0])
    if positions[-1] <= 1:
        positions = [int(line_length * p) for p in positions]
    # header names for the different log elements
    to_display = ['Layer (type)', 'Output Shape', 'Param #', 'Previous Layer']
    def print_row(fields, positions):
        """Print format row.

        Parameters
        ----------
        fields: list
            Information field.
        positions: list
            Field length ratio.
        Returns
        ------
        None
        """
        line = ''
        for i, field in enumerate(fields):
            line += str(field)
            line = line[:positions[i]]
            line += ' ' * (positions[i] - len(line))
        print(line)
    print('_' * line_length)
    print_row(to_display, positions)
    print('=' * line_length)
    def print_layer_summary(node, out_shape):
        """print layer information

        Parameters
        ----------
        node: dict
            Node information.
        out_shape: dict
            Node shape information.
        Returns
        ------
            Node total parameters.
        """
        op = node["op"]
        pre_node = []
        pre_filter = 0
        if op != "null":
            inputs = node["inputs"]
            for item in inputs:
                input_node = nodes[item[0]]
                input_name = input_node["name"]
                if input_node["op"] != "null" or item[0] in heads:
                    # add precede
                    pre_node.append(input_name)
                    if show_shape:
                        if input_node["op"] != "null":
                            key = input_name + "_output"
                        else:
                            key = input_name
                        if key in shape_dict:
                            shape = shape_dict[key][1:]
                            pre_filter = pre_filter + int(shape[0])
        cur_param = 0
        if op == 'Convolution':
            if "no_bias" in node["attrs"] and node["attrs"]["no_bias"] == 'True':
                num_group = int(node['attrs'].get('num_group', '1'))
                cur_param = pre_filter * int(node["attrs"]["num_filter"]) \
                   // num_group
                for k in _str2tuple(node["attrs"]["kernel"]):
                    cur_param *= int(k)
            else:
                num_group = int(node['attrs'].get('num_group', '1'))
                cur_param = pre_filter * int(node["attrs"]["num_filter"]) \
                   // num_group
                for k in _str2tuple(node["attrs"]["kernel"]):
                    cur_param *= int(k)
                cur_param += int(node["attrs"]["num_filter"])
        elif op == 'FullyConnected':
            if "no_bias" in node["attrs"] and node["attrs"]["no_bias"] == 'True':
                cur_param = pre_filter * int(node["attrs"]["num_hidden"])
            else:
                cur_param = (pre_filter+1) * int(node["attrs"]["num_hidden"])
        elif op == 'BatchNorm':
            key = node["name"] + "_output"
            if show_shape:
                num_filter = shape_dict[key][1]
                cur_param = int(num_filter) * 2
        elif op == 'Embedding':
            cur_param = int(node["attrs"]['input_dim']) * int(node["attrs"]['output_dim'])
        if not pre_node:
            first_connection = ''
        else:
            first_connection = pre_node[0]
        fields = [node['name'] + '(' + op + ')',
                  "x".join([str(x) for x in out_shape]),
                  cur_param,
                  first_connection]
        print_row(fields, positions)
        if len(pre_node) > 1:
            for i in range(1, len(pre_node)):
                fields = ['', '', '', pre_node[i]]
                print_row(fields, positions)
        return cur_param
    total_params = 0
    for i, node in enumerate(nodes):
        out_shape = []
        op = node["op"]
        if op == "null" and i > 0:
            continue
        if op != "null" or i in heads:
            if show_shape:
                if op != "null":
                    key = node["name"] + "_output"
                else:
                    key = node["name"]
                if key in shape_dict:
                    out_shape = shape_dict[key][1:]
        total_params += print_layer_summary(nodes[i], out_shape)
        if i == len(nodes) - 1:
            print('=' * line_length)
        else:
            print('_' * line_length)
    print("Total params: {params}".format(params=total_params))
    print('_' * line_length)

def plot_network(symbol, title="plot", save_format='pdf', shape=None, dtype=None, node_attrs={},
                 hide_weights=True):
    """Creates a visualization (Graphviz digraph object) of the given computation graph.
    Graphviz must be installed for this function to work.

    Parameters
    ----------
    title: str, optional
        Title of the generated visualization.
    symbol: Symbol
        A symbol from the computation graph. The generated digraph will visualize the part
        of the computation graph required to compute `symbol`.
    shape: dict, optional
        Specifies the shape of the input tensors. If specified, the visualization will include
        the shape of the tensors between the nodes. `shape` is a dictionary mapping
        input symbol names (str) to the corresponding tensor shape (tuple).
    dtype: dict, optional
        Specifies the type of the input tensors. If specified, the visualization will include
        the type of the tensors between the nodes. `dtype` is a dictionary mapping
        input symbol names (str) to the corresponding tensor type (e.g. `numpy.float32`).
    node_attrs: dict, optional
        Specifies the attributes for nodes in the generated visualization. `node_attrs` is
        a dictionary of Graphviz attribute names and values. For example::

            node_attrs={"shape":"oval","fixedsize":"false"}

        will use oval shape for nodes and allow variable sized nodes in the visualization.
    hide_weights: bool, optional
        If True (default), then inputs with names of form *_weight* (corresponding to weight
        tensors) or *_bias* (corresponding to bias vectors) will be hidden for a cleaner
        visualization.

    Returns
    -------
    dot: Digraph
        A Graphviz digraph object visualizing the computation graph to compute `symbol`.

    Example
    -------
    >>> net = mx.sym.Variable('data')
    >>> net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=128)
    >>> net = mx.sym.Activation(data=net, name='relu1', act_type="relu")
    >>> net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=10)
    >>> digraph = mx.viz.plot_network(net, shape={'data':(100,200)},
    ... node_attrs={"fixedsize":"false"})
    >>> digraph.view()

    Notes
    -----
    If ``mxnet`` is imported, the visualization module can be used in its short-form.
    For example, if we ``import mxnet`` as follows::

        import mxnet

    this method in visualization module can be used in its short-form as::

        mxnet.viz.plot_network(...)

    """
    # todo add shape support
    try:
        from graphviz import Digraph
    except:
        raise ImportError("Draw network requires graphviz library")
    if not isinstance(symbol, Symbol):
        raise TypeError("symbol must be a Symbol")
    internals = symbol.get_internals()
    draw_shape = shape is not None
    if draw_shape:
        _, out_shapes, _ = internals.infer_shape(**shape)
        if out_shapes is None:
            raise ValueError("Input shape is incomplete")
        shape_dict = dict(zip(internals.list_outputs(), out_shapes))
    draw_type = dtype is not None
    if draw_type:
        _, out_types, _ = internals.infer_type(**dtype)
        if out_types is None:
            raise ValueError("Input type is incomplete")
        type_dict = dict(zip(internals.list_outputs(), out_types))
    conf = json.loads(symbol.tojson())
    nodes = conf["nodes"]
    # check if multiple nodes have the same name
    if len(nodes) != len(set([node["name"] for node in nodes])):
        seen_nodes = set()
        # find all repeated names
        repeated = set(node['name'] for node in nodes if node['name'] in seen_nodes
                       or seen_nodes.add(node['name']))
        warning_message = "There are multiple variables with the same name in your graph, " \
                          "this may result in cyclic graph. Repeated names: " + ','.join(repeated)
        warnings.warn(warning_message, RuntimeWarning)
    # default attributes of node
    node_attr = {"shape": "box", "fixedsize": "true",
                 "width": "1.3", "height": "0.8034", "style": "filled"}
    # merge the dict provided by user and the default one
    node_attr.update(node_attrs)
    dot = Digraph(name=title, format=save_format)
    # color map
    cm = ("#8dd3c7", "#fb8072", "#ffffb3", "#bebada", "#80b1d3",
          "#fdb462", "#b3de69", "#fccde5")

    def looks_like_weight(name):
        """Internal helper to figure out if node should be hidden with `hide_weights`.
        """
        idx = name.find("_quantize")
        if idx >= 0:
            name = name[0:idx]
        weight_like = ('_weight', '_bias', '_beta', '_gamma',
                       '_moving_var', '_moving_mean', '_running_var', '_running_mean')

        return name.endswith(weight_like)

    def to_shorter_name(name):
        if name.startswith("_contrib_"):
            return name.replace("_contrib_", "")
        return name


    # make nodes
    hidden_nodes = set()
    for node in nodes:
        op = node["op"]
        name = node["name"]
        # input data
        attr = copy.deepcopy(node_attr)
        label = name

        if op == "null":
            if looks_like_weight(node["name"]):
                if hide_weights:
                    hidden_nodes.add(node["name"])
                # else we don't render a node, but
                # don't add it to the hidden_nodes set
                # so it gets rendered as an empty oval
                continue
            attr["shape"] = "oval" # inputs get their own shape
            label = node["name"]
            attr["fillcolor"] = cm[0]
        elif op == "Convolution" or op == "_contrib_quantized_conv":
            label = "{op}\n{kernel}/{stride}, {filter}".format(
                op=to_shorter_name(op),
                kernel="x".join(_str2tuple(node["attrs"]["kernel"])),
                stride="x".join(_str2tuple(node["attrs"]["stride"]))
                if "stride" in node["attrs"] else "1",
                filter=node["attrs"]["num_filter"]
            )
            attr["fillcolor"] = cm[1]
        elif op == "FullyConnected":
            label = "FullyConnected\n{hidden}".format(hidden=node["attrs"]["num_hidden"])
            attr["fillcolor"] = cm[1]
        elif op == "BatchNorm":
            attr["fillcolor"] = cm[3]
        elif op == 'Activation':
            act_type = node["attrs"]["act_type"]
            label = 'Activation\n{activation}'.format(activation=act_type)
            attr["fillcolor"] = cm[2]
        elif op == 'LeakyReLU':
            attrs = node.get("attrs")
            act_type = attrs.get("act_type", "Leaky") if attrs else "Leaky"
            label = 'LeakyReLU\n{activation}'.format(activation=act_type)
            attr["fillcolor"] = cm[2]
        elif op == "Pooling" or op == "_contrib_quantized_pooling":
            label = "{op}\n{pooltype}, {kernel}/{stride}".format(op=to_shorter_name(op),
                                                                 pooltype=node["attrs"]["pool_type"],
                                                                 kernel="x".join(_str2tuple(node["attrs"]["kernel"]))
                                                                 if "kernel" in node["attrs"] else "[]",
                                                                 stride="x".join(_str2tuple(node["attrs"]["stride"]))
                                                                 if "stride" in node["attrs"] else "1")
            attr["fillcolor"] = cm[4]
        elif op in ("Concat", "Flatten", "Reshape"):
            attr["fillcolor"] = cm[5]
        elif op == "Softmax":
            attr["fillcolor"] = cm[6]
        elif op == "_contrib_dequantize" or op == "_contrib_requantize" or op == '_contrib_quantized_act':
            label = to_shorter_name(op)
            attr["fillcolor"] = cm[2]
        else:
            attr["fillcolor"] = cm[7]
            if op == "Custom":
                label = node["attrs"]["op_type"]

        dot.node(name=name, label=label, **attr)

    def to_shape_key(input_name):
        output = input_name + '_output'
        if output in shape_dict:
            return output
        else:
            return input_name + '_output0'


    # add edges
    for node in nodes:          # pylint: disable=too-many-nested-blocks
        op = node["op"]
        name = node["name"]
        if op == "null":
            continue
        else:
            inputs = node["inputs"]

            if node['op'] == '_contrib_BilinearResize2D':
                inputs = [inputs[0]]

            if node['op'].find("quantize") > 0:
                inputs = [inputs[0]]

            for item in inputs:
                input_node = nodes[item[0]]
                input_name = input_node["name"]
                if input_name not in hidden_nodes:
                    attr = {"dir": "back", 'arrowtail':'open', 'label': ''}
                    # add shapes
                    if draw_shape:
                        if input_node["op"] != "null":
                            key = to_shape_key(input_name)
                            if "attrs" in input_node:
                                params = input_node["attrs"]
                                if "num_outputs" in params:
                                    key += str(int(params["num_outputs"]) - 1)
                            shape = shape_dict[key][1:]
                            label = "x".join([str(x) for x in shape])
                            attr["label"] = label
                        else:
                            key = input_name
                            shape = shape_dict[key][1:]
                            label = "x".join([str(x) for x in shape])
                            attr["label"] = label
                    if draw_type:
                        if input_node["op"] != "null":
                            key = to_shape_key(input_name)
                            if "attrs" in input_node:
                                params = input_node["attrs"]
                                if "num_outputs" in params:
                                    key += str(int(params["num_outputs"]) - 1)
                            dtype = type_dict[key]
                            attr["label"] += '(' + dtype.__name__ + ')'
                        else:
                            key = input_name
                            dtype = type_dict[key]
                            attr["label"] += '(' + dtype.__name__ + ')'
                    dot.edge(tail_name=name, head_name=input_name, **attr)

    return dot
