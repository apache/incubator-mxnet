/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax_output.cc
 * \brief
 * \author Bing Xu, Zhang Rong A
*/
#include "./softmax_output-inl.h"
#if MXNET_USE_MKLDNN == 1
#include "./nn/mkldnn/mkldnn_ops-inl.h"
#include "./nn/mkldnn/mkldnn_base-inl.h"
#endif
namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(SoftmaxOutputParam);
struct SoftmaxOutputGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::ObjectPtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
  std::vector<nnvm::NodeEntry> out_data(n->num_outputs());
  for (uint32_t i = 0; i < out_data.size(); ++i) {
    out_data[i] = nnvm::NodeEntry{n, i, 0};
  }
  std::vector<nnvm::NodeEntry> heads;
  heads.push_back(out_data[softmaxout_enum::kOut]);
  heads.push_back(n->inputs[softmaxout_enum::kLabel]);

  nnvm::ObjectPtr gnode = nnvm::Node::Create();
  gnode->inputs = std::move(heads);
  gnode->control_deps.emplace_back(n);
  gnode->attrs = n->attrs;
  gnode->attrs.op = nnvm::Op::Get("_backward_SoftmaxOutput");
  gnode->attrs.name = n->attrs.name + "_backward";
  std::vector<nnvm::NodeEntry> in_grad(2);
  in_grad[0] = nnvm::NodeEntry{gnode, 0, 0};
  in_grad[1] = nnvm::NodeEntry{gnode, 1, 0};
  return in_grad;
  }
};

static inline std::vector<std::string> ListArguments() {
  return {"data", "label"};
}

static bool SoftmaxOutputType(const nnvm::NodeAttrs& attrs,
                              std::vector<int> *in_type,
                              std::vector<int> *out_type) {
  CHECK_EQ(in_type->size(), 2U);
  int dtype = (*in_type)[0];
  if (type_is_none(dtype)) {
    // Input type is undefined, we try backward inference
    if (out_type->size() == 0 || type_is_none((*out_type)[0])) {
      // Neither the input nor the output are defined,
      // types cannot be infered for this op
      return false;
    } else {
      // Input type is undefined but output type is: backward inference
      dtype = (*out_type)[0];
    }
  } else {
    // Input type is defined but output type is not: forward inference
    out_type->clear();
    out_type->push_back(dtype);
  }
  for (size_t i = 0; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = dtype;
    } else {
      UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments()[i]);
    }
  }
  return true;
}

static bool SoftmaxOutputShape(const nnvm::NodeAttrs& attrs,
                               mxnet::ShapeVector *in_shape,
                               mxnet::ShapeVector *out_shape) {
  using namespace mshadow;
  const SoftmaxOutputParam& param = nnvm::get<SoftmaxOutputParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 2U) << "Input:[data, label]";
  const mxnet::TShape &dshape = in_shape->at(0);
  if (!mxnet::ndim_is_known(dshape)) return false;

  // label.shape == data.shape: use probability as label
  if (dshape != (*in_shape)[softmaxout_enum::kLabel]) {
    if (param.multi_output) {
      mxnet::TShape lshape1 = Shape2(dshape[0], dshape.Size()/dshape[0]/dshape[1]);
      mxnet::TShape lshape2(dshape.ndim() - 1, -1);
      lshape2[0] = dshape[0];
      for (int i = 2; i < dshape.ndim(); ++i)
        lshape2[i-1] = dshape[i];
      mxnet::TShape lshape3 = dshape;
      lshape3[1] = 1;
      if (!mxnet::ndim_is_known(in_shape->at(softmaxout_enum::kLabel))) {
        in_shape->at(softmaxout_enum::kLabel) = lshape1;
      } else if (in_shape->at(softmaxout_enum::kLabel) == lshape1) {
      } else if (in_shape->at(softmaxout_enum::kLabel) == lshape2) {
      } else if (in_shape->at(softmaxout_enum::kLabel) == lshape3) {
      } else {
        std::ostringstream os;
        os << "Expecting " << lshape1 << " or " << lshape2
           << ". But got " << in_shape->at(softmaxout_enum::kLabel);
        throw InferShapeError(os.str(), softmaxout_enum::kLabel);
      }
    } else {
      mxnet::TShape label_shape(dshape.ndim() - 1, -1);
      for (int i = 0; i + 1 < dshape.ndim(); ++i)
        label_shape[i] = dshape[i];
      SHAPE_ASSIGN_CHECK(*in_shape, softmaxout_enum::kLabel, label_shape);
    }
  }

  out_shape->clear();
  out_shape->push_back(dshape);
  return true;
}

#if MXNET_USE_MKLDNN == 1
inline static bool SoftmaxOutputStorageType(const nnvm::NodeAttrs& attrs,
                                            const int dev_mask,
                                            DispatchMode* dispatch_mode,
                                            std::vector<int>* in_attrs,
                                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);

  return MKLDNNStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs,
                           out_attrs);
}

void SoftmaxOutputComputeExCPU(const nnvm::NodeAttrs &attrs,
                               const OpContext &ctx,
                               const std::vector<NDArray> &inputs,
                               const std::vector<OpReqType> &req,
                               const std::vector<NDArray> &outputs) {
  CHECK_EQ(inputs.size(), 2U);
  const SoftmaxOutputParam &param = nnvm::get<SoftmaxOutputParam>(attrs.parsed);
  if (SupportMKLDNN(inputs[0]) && !ctx.is_train && SupportMKLDNNSoftmaxOutput(param)) {
    MKLDNN_OPCHECK_INIT(false, outputs.size(), inputs, outputs);
    MKLDNNRun(MKLDNNSoftmaxOutputForward, attrs, ctx, inputs, req, outputs);
    MKLDNN_OPCHECK_RUN(SoftmaxOutputCompute<cpu>, attrs, ctx, inputs, req, outputs);
    return;
  }
  FallBackCompute(SoftmaxOutputCompute<cpu>, attrs, ctx, inputs, req, outputs);
}
#endif

NNVM_REGISTER_OP(SoftmaxOutput)
.describe(R"code(Computes the gradient of cross entropy loss with respect to softmax output.

- This operator computes the gradient in two steps.
  The cross entropy loss does not actually need to be computed.

  - Applies softmax function on the input array.
  - Computes and returns the gradient of cross entropy loss w.r.t. the softmax output.

- The softmax function, cross entropy loss and gradient is given by:

  - Softmax Function:

    .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}

  - Cross Entropy Function:

    .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i \log(\text{output}_i)

  - The gradient of cross entropy loss w.r.t softmax output:

    .. math:: \text{gradient} = \text{output} - \text{label}

- During forward propagation, the softmax function is computed for each instance in the input array.

  For general *N*-D input arrays with shape :math:`(d_1, d_2, ..., d_n)`. The size is
  :math:`s=d_1 \cdot d_2 \cdot \cdot \cdot d_n`. We can use the parameters `preserve_shape`
  and `multi_output` to specify the way to compute softmax:

  - By default, `preserve_shape` is ``false``. This operator will reshape the input array
    into a 2-D array with shape :math:`(d_1, \frac{s}{d_1})` and then compute the softmax function for
    each row in the reshaped array, and afterwards reshape it back to the original shape
    :math:`(d_1, d_2, ..., d_n)`.
  - If `preserve_shape` is ``true``, the softmax function will be computed along
    the last axis (`axis` = ``-1``).
  - If `multi_output` is ``true``, the softmax function will be computed along
    the second axis (`axis` = ``1``).

- During backward propagation, the gradient of cross-entropy loss w.r.t softmax output array is computed.
  The provided label can be a one-hot label array or a probability label array.

  - If the parameter `use_ignore` is ``true``, `ignore_label` can specify input instances
    with a particular label to be ignored during backward propagation. **This has no effect when
    softmax `output` has same shape as `label`**.

    Example::

      data = [[1,2,3,4],[2,2,2,2],[3,3,3,3],[4,4,4,4]]
      label = [1,0,2,3]
      ignore_label = 1
      SoftmaxOutput(data=data, label = label,\
                    multi_output=true, use_ignore=true,\
                    ignore_label=ignore_label)
      ## forward softmax output
      [[ 0.0320586   0.08714432  0.23688284  0.64391428]
       [ 0.25        0.25        0.25        0.25      ]
       [ 0.25        0.25        0.25        0.25      ]
       [ 0.25        0.25        0.25        0.25      ]]
      ## backward gradient output
      [[ 0.    0.    0.    0.  ]
       [-0.75  0.25  0.25  0.25]
       [ 0.25  0.25 -0.75  0.25]
       [ 0.25  0.25  0.25 -0.75]]
      ## notice that the first row is all 0 because label[0] is 1, which is equal to ignore_label.

  - The parameter `grad_scale` can be used to rescale the gradient, which is often used to
    give each loss function different weights.

  - This operator also supports various ways to normalize the gradient by `normalization`,
    The `normalization` is applied if softmax output has different shape than the labels.
    The `normalization` mode can be set to the followings:

    - ``'null'``: do nothing.
    - ``'batch'``: divide the gradient by the batch size.
    - ``'valid'``: divide the gradient by the number of instances which are not ignored.

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SoftmaxOutputParam>)
#if MXNET_USE_MKLDNN == 1
.set_attr<FInferStorageType>("FInferStorageType", SoftmaxOutputStorageType)
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FComputeEx>("FComputeEx<cpu>", SoftmaxOutputComputeExCPU)
#endif
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "label"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", SoftmaxOutputShape)
.set_attr<nnvm::FInferType>("FInferType", SoftmaxOutputType)
.set_attr<FCompute>("FCompute<cpu>", SoftmaxOutputCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", SoftmaxOutputGrad{"_backward_SoftmaxOutput"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs){
  return std::vector<std::pair<int, int> >{{0, 0}};
})
.add_argument("data", "NDArray-or-Symbol", "Input array.")
.add_argument("label", "NDArray-or-Symbol", "Ground truth label.")
.add_arguments(SoftmaxOutputParam::__FIELDS__());

// Softmax symbol is renamed to SoftmaxOutput and deprecated since Dec, 2015
NNVM_REGISTER_OP(SoftmaxOutput).add_alias("Softmax");

NNVM_REGISTER_OP(_backward_SoftmaxOutput)
.set_num_inputs(2)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs){
  return std::vector<std::pair<int, int> >{{0, 0}};
})
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n){
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr_parser(ParamParser<SoftmaxOutputParam>)
.set_attr<FCompute>("FCompute<cpu>", SoftmaxOutputGradCompute<cpu>);
}  // namespace op
}  // namespace mxnet
