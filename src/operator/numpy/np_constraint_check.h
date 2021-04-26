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
 * \file np_constraint_check.h
 * \brief helper function for constraint check
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_CONSTRAINT_CHECK_H_
#define MXNET_OPERATOR_NUMPY_NP_CONSTRAINT_CHECK_H_

#include <algorithm>
#include <string>
#include <vector>
#include "./np_broadcast_reduce_op.h"

namespace mxnet {
namespace op {

template<typename xpu>
void GetReduceOutput(mshadow::Stream<xpu> *s, const TBlob &output_blob, bool *red_output);

struct ConstraintCheckParam : public dmlc::Parameter<ConstraintCheckParam> {
  std::string msg;
  DMLC_DECLARE_PARAMETER(ConstraintCheckParam) {
    DMLC_DECLARE_FIELD(msg)
    .set_default("Constraint violated.")
    .describe("Error message raised when constraint violated");
  }
};

template <typename xpu>
void ConstraintCheckForward(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  const ConstraintCheckParam& param =
      nnvm::get<ConstraintCheckParam>(attrs.parsed);
#if !defined(__CUDACC__)
  ReduceAxesComputeImpl<xpu, mshadow_op::product, false, false,
                        op::mshadow_op::identity>(ctx, inputs, req, outputs,
                                                  outputs[0].shape_);
#else
  ReduceAxesRTCComputeImpl(ctx, inputs, req, outputs,
                           outputs[0].shape_, "red::product{}");
#endif
  std::string msg = param.msg;
  bool red_output = true;
  GetReduceOutput(ctx.get_stream<xpu>(), outputs[0], &red_output);
  CHECK_EQ(red_output, true) << "ValueError: " << msg;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_CONSTRAINT_CHECK_H_
