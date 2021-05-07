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

# ONNX Export Support for MXNet

### Overview
[ONNX](https://onnx.ai/), or Open Neural Network Exchange, is an open source deep learning model format that acts as a framework neutral graph representation between DL frameworks or between training and inference. With the ability to export models to the ONNX format, MXNet users can enjoy faster inference and a wider range of deployment device choices, including edge and mobile devices where MXNet installation may be constrained. Popular hardware-accelerated and/or cross-platform ONNX runtime frameworks include Nvidia [TensorRT](https://github.com/onnx/onnx-tensorrt), Microsoft [ONNXRuntime](https://github.com/microsoft/onnxruntime), Apple [CoreML](https://github.com/onnx/onnx-coreml) and [TVM](https://tvm.apache.org/docs/tutorials/frontend/from_onnx.html), etc. 

### ONNX Versions Supported
ONNX 1.7 -- Fully Supported
ONNX 1.8 -- Work in Progress

### Installation
From the 1.9 release and on, the ONNX export module has become an offical, built-in module in MXNet. You can access the module at `mxnet.onnx`. 

If you are a user of earlier MXNet versions and do not want to upgrade MXNet, you can still enjoy the latest ONNX suppor by pulling the MXNet source code and building the wheel for only the mx2onnx module. Just do `cd python/mxnet/onnx` and then build the wheel with `python3 -m build`. You should be able to find the wheel under `python/mxnet/onnx/dist/mx2onnx-0.0.0-py3-none-any.whl` and install it with `pip install mx2onnx-0.0.0-py3-none-any.whl`. You should be able to access the module with `import mx2onnx` then.

### APIs
```python
mxnet.onnx.export_model(sym, params, in_shapes=None, in_types=np.float32,
                 onnx_file_path='model.onnx', verbose=False, dynamic=False,
                 dynamic_input_shapes=None, run_shape_inference=False, input_type=None,
                 input_shape=None)
```
Exports the MXNet model file into ONNX model.

Parameters:

    sym : str or symbol object
        Path to the MXNet json file or Symbol object
    params : str or dict or list of dict
        str - Path to the MXNet params file
        dict - MXNet params dictionary (Including both arg_params and aux_params)
        list - list of length 2 that contains MXNet arg_params and aux_params
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

Returns:

    onnx_file_path : str
        Onnx file path

#### Model with Multiple Input
When the model has multiple input, all the input shapes and dtypes should be provided with `in_shapes` and `in_dtypes`. Note that the shape/dtype in `in_shapes`/`in_dtypes` must follow the same order as in the MXNet model symbol file. If `in_dtypes` is provided as a single data type, the type will be applied to all input nodes.

#### Dynamic Shape Input
By setting up optional flags in export_model API, users have the control of partially/fully dynamic shape input export. The flag `dynamic` is set to switch on dynamic shape input export, and `dynamic_input_shapes` is used to specify which dimensions are dynamic `None` or any string variable can be used to represent a dynamic shape dimension.

```python
# The first input dimension will be dynamic in this case
dynamic_input_shapes = [(None, 3, 224, 224)]
mx.onnx.export_model(mx_sym, mx_params, in_shapes, in_dtypes, onnx_file,
                     dynamic=True, dynamic_input_shapes=dynamic_input_shapes)
```

### Operator Support Matrix
|MXNet Op|ONNX Version|
|:-|:-:|
|Activation|1.7 1.8 |
|BatchNorm|1.7 1.8 |
|BlockGrad|1.7 1.8 |
|Cast|1.7 1.8 |
|Concat|1.7 1.8 |
|Convolution|1.7 1.8 |
|Crop|1.7 1.8 |
|Deconvolution|1.7 1.8 |
|Dropout|1.7 1.8 |
|Embedding|1.7 1.8 |
|Flatten|1.7 1.8 |
|FullyConnected|1.7 1.8 |
|InstanceNorm|1.7 1.8 |
|L2Normalization|1.7 1.8 |
|LRN|1.7 1.8 |
|LayerNorm|1.7 1.8 |
|LeakyReLU|1.7 1.8 |
|LogisticRegressionOutput|1.7 1.8 |
|MakeLoss|1.7 1.8 |
|Pad|1.7 1.8 |
|Pooling|1.7 1.8 |
|RNN|1.7 1.8 |
|ROIPooling|1.7 1.8 |
|Reshape|1.7 1.8 |
|SequenceMask|1.7 1.8 |
|SequenceReverse|1.7 1.8 |
|SliceChannel|1.7 1.8 |
|SoftmaxOutput|1.7 1.8 |
|SwapAxis|1.7 1.8 |
|UpSampling|1.7 1.8 |
|_arange|1.7 1.8 |
|_contrib_AdaptiveAvgPooling2D|1.7 1.8 |
|_contrib_BilinearResize2D|1.7 1.8 |
|_contrib_ROIAlign|1.7 1.8 |
|_contrib_arange_like|1.7 1.8 |
|_contrib_box_decode|1.7 1.8 |
|_contrib_box_nms|1.7 1.8 |
|_contrib_div_sqrt_dim|1.7 1.8 |
|_contrib_interleaved_matmul_selfatt_qk|1.7 1.8 |
|_contrib_interleaved_matmul_selfatt_valatt|1.7 1.8 |
|_copy|1.7 1.8 |
|_div_scalar|1.7 1.8 |
|_equal_scalar|1.7 1.8 |
|_greater_scalar|1.7 1.8 |
|_lesser_scalar|1.7 1.8 |
|_linalg_gemm2|1.7 1.8 |
|_maximum|1.7 1.8 |
|_maximum_scalar|1.7 1.8 |
|_minimum|1.7 1.8 |
|_minimum_scalar|1.7 1.8 |
|_minus_scalar|1.7 1.8 |
|_mul_scalar|1.7 1.8 |
|_ones|1.7 1.8 |
|_plus_scalar|1.7 1.8 |
|_power|1.7 1.8 |
|_power_scalar|1.7 1.8 |
|_random_normal|1.7 1.8 |
|_random_uniform|1.7 1.8 |
|_random_uniform_like|1.7 1.8 |
|_rdiv_scalar|1.7 1.8 |
|_rminus_scalar|1.7 1.8 |
|_rnn_param_concat|1.7 1.8 |
|_sample_multinomial|1.7 1.8 |
|_zeros|1.7 1.8 |
|abs|1.7 1.8 |
|add_n|1.7 1.8 |
|arccos|1.7 1.8 |
|arcsin|1.7 1.8 |
|arctan|1.7 1.8 |
|argmax|1.7 1.8 |
|argmin|1.7 1.8 |
|argsort|1.7 1.8 |
|batch_dot|1.7 1.8 |
|broadcast_add|1.7 1.8 |
|broadcast_axis|1.7 1.8 |
|broadcast_div|1.7 1.8 |
|broadcast_equal|1.7 1.8 |
|broadcast_greater|1.7 1.8 |
|broadcast_greater_equal|1.7 1.8 |
|broadcast_lesser|1.7 1.8 |
|broadcast_lesser_equal|1.7 1.8 |
|broadcast_like|1.7 1.8 |
|broadcast_logical_and|1.7 1.8 |
|broadcast_logical_or|1.7 1.8 |
|broadcast_logical_xor|1.7 1.8 |
|broadcast_minimum|1.7 1.8 |
|broadcast_mod|1.7 1.8 |
|broadcast_mul|1.7 1.8 |
|broadcast_power|1.7 1.8 |
|broadcast_sub|1.7 1.8 |
|broadcast_to|1.7 1.8 |
|ceil|1.7 1.8 |
|clip|1.7 1.8 |
|cos|1.7 1.8 |
|depth_to_space|1.7 1.8 |
|dot|1.7 1.8 |
|elemwise_add|1.7 1.8 |
|elemwise_div|1.7 1.8 |
|elemwise_mul|1.7 1.8 |
|elemwise_sub|1.7 1.8 |
|exp|1.7 1.8 |
|expand_dims|1.7 1.8 |
|floor|1.7 1.8 |
|gather_nd|1.7 1.8 |
|hard_sigmoid|1.7 1.8 |
|identity|1.7 1.8 |
|log|1.7 1.8 |
|log2|1.7 1.8 |
|log_softmax|1.7 1.8 |
|logical_not|1.7 1.8 |
|max|1.7 1.8 |
|mean|1.7 1.8 |
|min|1.7 1.8 |
|negative|1.7 1.8 |
|norm|1.7 1.8 |
|null|1.7 1.8 |
|one_hot|1.7 1.8 |
|ones_like|1.7 1.8 |
|prod|1.7 1.8 |
|reciprocal|1.7 1.8 |
|relu|1.7 1.8 |
|repeat|1.7 1.8 |
|reshape_like|1.7 1.8 |
|reverse|1.7 1.8 |
|shape_array|1.7 1.8 |
|sigmoid|1.7 1.8 |
|sin|1.7 1.8 |
|size_array|1.7 1.8 |
|slice|1.7 1.8 |
|slice_axis|1.7 1.8 |
|slice_like|1.7 1.8 |
|softmax|1.7 1.8 |
|space_to_depth|1.7 1.8 |
|sqrt|1.7 1.8 |
|square|1.7 1.8 |
|squeeze|1.7 1.8 |
|stack|1.7 1.8 |
|sum|1.7 1.8 |
|take|1.7 1.8 |
|tan|1.7 1.8 |
|tanh|1.7 1.8 |
|tile|1.7 1.8 |
|topk|1.7 1.8 |
|transpose|1.7 1.8 |
|where|1.7 1.8 |
|zeros_like|1.7 1.8 |

### [GluonCV Pretrained Model Support Matrix](https://cv.gluon.ai/model_zoo/index.html)
|Image Classification|
|:-|
|alexnet|
|cifar_resnet20_v1|
|cifar_resnet56_v1|
|cifar_resnet110_v1|
|cifar_resnet20_v2|
|cifar_resnet56_v2|
|cifar_resnet110_v2|
|cifar_wideresnet16_10|
|cifar_wideresnet28_10|
|cifar_wideresnet40_8|
|cifar_resnext29_16x64d|
|darknet53|
|densenet121|
|densenet161|
|densenet169|
|densenet201|
|googlenet|
|mobilenet1.0|
|mobilenet0.75|
|mobilenet0.5|
|mobilenet0.25|
|mobilenetv2_1.0|
|mobilenetv2_0.75|
|mobilenetv2_0.5|
|mobilenetv2_0.25|
|mobilenetv3_large|
|mobilenetv3_small|
|resnest14|
|resnest26|
|resnest50|
|resnest101|
|resnest200|
|resnest269|
|resnet18_v1|
|resnet18_v1b_0.89|
|resnet18_v2|
|resnet34_v1|
|resnet34_v2|
|resnet50_v1|
|resnet50_v1d_0.86|
|resnet50_v1d_0.48|
|resnet50_v1d_0.37|
|resnet50_v1d_0.11|
|resnet50_v2|
|resnet101_v1|
|resnet101_v1d_0.76|
|resnet101_v1d_0.73|
|resnet101_v2|
|resnet152_v1|
|resnet152_v2|
|resnext50_32x4d|
|resnext101_32x4d|
|resnext101_64x4d|
|senet_154|
|se_resnext101_32x4d|
|se_resnext101_64x4d|
|se_resnext50_32x4d|
|squeezenet1.0|
|squeezenet1.1|
|vgg11|
|vgg11_bn|
|vgg13|
|vgg13_bn|
|vgg16|
|vgg16_bn|
|vgg19|
|vgg19_bn|
|xception|
|inceptionv3|

|Object Detection|
|:-|
|center_net_resnet18_v1b_voc|
|center_net_resnet50_v1b_voc|
|center_net_resnet101_v1b_voc|
|center_net_resnet18_v1b_coco|
|center_net_resnet50_v1b_coco|
|center_net_resnet101_v1b_coco|
|ssd_300_vgg16_atrous_voc|
|ssd_512_vgg16_atrous_voc|
|ssd_512_resnet50_v1_voc|
|ssd_512_mobilenet1.0_voc|
|faster_rcnn_resnet50_v1b_voc|
|yolo3_darknet53_voc|
|yolo3_mobilenet1.0_voc|
|ssd_300_vgg16_atrous_coco|
|ssd_512_vgg16_atrous_coco|
|ssd_300_resnet34_v1b_coco|
|ssd_512_resnet50_v1_coco|
|ssd_512_mobilenet1.0_coco|
|faster_rcnn_resnet50_v1b_coco|
|faster_rcnn_resnet101_v1d_coco|
|yolo3_darknet53_coco|
|yolo3_mobilenet1.0_coco|
|faster_rcnn_fpn_resnet50_v1b_coco|
|faster_rcnn_fpn_resnet101_v1d_coco|
|mask_rcnn_fpn_resnet18_v1b_coco|
|mask_rcnn_resnet18_v1b_coco|
|mask_rcnn_resnet50_v1b_coco|
|mask_rcnn_resnet101_v1d_coco|
|mask_rcnn_fpn_resnet50_v1b_coco|
|mask_rcnn_fpn_resnet101_v1d_coco|

|Image Segmentation|
|:-|
|fcn_resnet50_ade|
|fcn_resnet101_ade|
|deeplab_resnet50_ade|
|deeplab_resnet101_ade|
|deeplab_resnest50_ade|
|deeplab_resnest101_ade|
|deeplab_resnest269_ade|
|fcn_resnet101_coco|
|deeplab_resnet101_coco|
|fcn_resnet101_voc|
|deeplab_resnet101_voc|
|deeplab_resnet152_voc|
|deeplab_resnet50_citys|
|deeplab_resnet101_citys|
|deeplab_v3b_plus_wideresnet_citys|
|danet_resnet50_citys|
|danet_resnet101_citys|

|Pose Estimation|
|:-|
|simple_pose_resnet18_v1b|
|simple_pose_resnet50_v1b|
|simple_pose_resnet50_v1d|
|simple_pose_resnet101_v1b|
|simple_pose_resnet101_v1d|
|simple_pose_resnet152_v1b|
|simple_pose_resnet152_v1d|
|alpha_pose_resnet101_v1b_coco|
|mobile_pose_resnet18_v1b|
|mobile_pose_resnet50_v1b|
|mobile_pose_mobilenet1.0|
|mobile_pose_mobilenetv2_1.0|
|mobile_pose_mobilenetv3_large|
|mobile_pose_mobilenetv3_small|

|Action Estimation|
|:-|
|inceptionv1_kinetics400|
|resnet18_v1b_kinetics400|
|resnet34_v1b_kinetics400|
|resnet50_v1b_kinetics400|
|resnet101_v1b_kinetics400|
|resnet152_v1b_kinetics400|
|resnet50_v1b_hmdb51|
|resnet50_v1b_sthsthv2|
|vgg16_ucf101|
|inceptionv3_kinetics400|
|inceptionv3_ucf101|

### [GluonNLP Pretrained Model Support Matrix](https://nlp.gluon.ai/model_zoo/catalog.html)
|NLP Models|
|:-|
|awd_lstm_lm_600|
|awd_lstm_lm_1150|
|bert_12_768_12|
|bert_24_1024_16|
|distilbert_6_768_12|
|ernie_12_768_12|
|gpt2_117m|
|gpt2_345m|
|roberta_12_768_12|
|roberta_24_1024_16|
|standard_lstm_lm_200|
|standard_lstm_lm_650|
|standard_lstm_lm_1500|
|transformer_en_de_512|