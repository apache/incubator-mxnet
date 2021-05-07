<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Exporting to ONNX format

[Open Neural Network Exchange (ONNX)](https://github.com/onnx/onnx) provides an open source format for AI models. It defines an extensible computation graph model, as well as definitions of built-in operators and standard data types. MXNet-ONNX export coverage and features are updated since MXNet 1.9.0. Visit the [ONNX operator coverage](https://github.com/apache/incubator-mxnet/tree/v1.x/python/mxnet/onnx#operator-support-matrix) page for the latest information.

In this tutorial, we will learn how to use MXNet to ONNX exporter on pre-trained models.

## Prerequisites

To run the tutorial you will need to have installed the following python modules:
- [MXNet >= 1.6.0](/get_started)
- [onnx >= 1.7.0](https://github.com/onnx/onnx#installation)

*Note:* MXNet-ONNX exporter works with ONNX opset version later than 12, which comes with ONNX v1.7.0


```python
import mxnet as mx
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
```

## Downloading a model from the MXNet model zoo

We download the pre-trained ResNet-18 [ImageNet](http://www.image-net.org/) model from the [MXNet Model Zoo](/api/python/docs/api/gluon/model_zoo/index.html).
We will also download synset file to match labels.

```python
# Download pre-trained resnet model - json and params by running following code.
path='http://data.mxnet.io/models/imagenet/'
[mx.test_utils.download(path+'resnet/18-layers/resnet-18-0000.params'),
 mx.test_utils.download(path+'resnet/18-layers/resnet-18-symbol.json'),
 mx.test_utils.download(path+'synset.txt')]
```

Now, we have downloaded ResNet-18 symbol, params and synset file on the disk.

## MXNet to ONNX exporter API

Let us describe the MXNet's `export_model` API. 

```python
help(mx.onnx.export_model)
```

Output:

```text
Help on function export_model in module mxnet.onnx.mx2onnx._export_model:

export_model(sym, params, in_shapes=None, in_types=<class 'numpy.float32'>, onnx_file_path='model.onnx', verbose=False, dynamic=False, dynamic_input_shapes=None, run_shape_inference=False, input_type=None, input_shape=None)
    Exports the MXNet model file, passed as a parameter, into ONNX model.
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
    
    Returns
    -------
    onnx_file_path : str
        Onnx file path
    
    Notes
    -----
    This method is available when you ``import mxnet.onnx``
```

`export_model` API can accept the MXNet model in one of the following ways.

1. MXNet's exported json and params files:
    * This is useful if we have pre-trained models and we want to convert them to ONNX format.
2. MXNet sym, params objects:
    * This is useful if we are training a model. At the end of training, we just need to invoke the `export_model` function and provide sym and params objects as inputs with other attributes to save the model in ONNX format. The params can be either a single object that contains both argument and auxiliary parameters, or a list that includes arg_parmas and aux_params objects


Since we have downloaded pre-trained model files, we will use the `export_model` API by passing the path for symbol and params files.

## How to use MXNet to ONNX exporter API

We will use the downloaded pre-trained model files (sym, params) and define input variables.

```python
# Downloaded input symbol and params files
sym = './resnet-18-symbol.json'
params = './resnet-18-0000.params'

# Standard Imagenet input - 3 channels, 224*224
input_shape = [(1,3,224,224)]
input_dtypes = [np.float32]

# Path of the output file
onnx_file = './mxnet_exported_resnet18.onnx'
```

We have defined the input parameters required for the `export_model` API. Now, we are ready to covert the MXNet model into ONNX format.

```python
# Invoke export model API. It returns path of the converted onnx model
converted_model_path = mx.onnx.export_model(sym, params, input_shape, input_dtypes, onnx_file)
```

This API returns path of the converted model which you can later use to import the model into other frameworks. Please refer to [mx2onnx](https://github.com/apache/incubator-mxnet/tree/v1.x/python/mxnet/onnx#apis) for more details about the API.

### Dynamic Shape Input
MXNet to ONNX export also supports dynamic input shapes. By setting up optional flags in `export_model`, users have the control of partially/fully dynamic shape input export. For example, setting the batch dimension to dynamic enables dynamic batching inference; setting the width and height dimension to dynamic allows inference on images with different shapes. Below is a code example for dynamic shape on batch dimension. The flag `dynamic` is set to switch on dynamic shape input export, and `dynamic_input_shapes` is used to specify which dimensions are dynamic. `None` or any string variable can be used to represent a dynamic shape dimension.

```python
# The first input dimension will be dynamic in this case
dynamic_input_shapes = [(None, 3, 224, 224)]
mx.onnx.export_model(mx_sym, mx_params, in_shapes, in_dtypes, onnx_file,
                     dynamic=True, dynamic_input_shapes=dynamic_input_shapes)
```

## Check validity of ONNX model

Now we can check validity of the converted ONNX model by using ONNX checker tool. The tool will validate the model by checking if the content contains valid protobuf:

```python
from onnx import checker
import onnx

# Load onnx model
model_proto = onnx.load_model(converted_model_path)

# Check if converted ONNX protobuf is valid
checker.check_graph(model_proto.graph)
```

If the converted protobuf format doesn't qualify to ONNX proto specifications, the checker will throw errors, but in this case it successfully passes. 

This method confirms exported model protobuf is valid. Now, the model is ready to be imported in other frameworks for inference! Users may consider to further optimize the ONNX model file using various tools such as [onnx-simplifier](https://github.com/daquexian/onnx-simplifier).
