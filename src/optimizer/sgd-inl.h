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
 *  Copyright (c) 2015 by Contributors
 * \file sgd-inl.h
 * \brief Operator interface of mxnet.
 * \author Junyuan Xie
 */
#ifndef MXNET_OPTIMIZER_SGD_INL_H_
#define MXNET_OPTIMIZER_SGD_INL_H_

#include <mshadow/tensor.h>
#include <mxnet/optimizer.h>
#include <dmlc/parameter.h>
#include <string>
#include <vector>
#include <map>
#include <utility>

namespace mxnet {
namespace opt {

struct SGDParam : public dmlc::Parameter<SGDParam> {
  float momentum;
  float wd;
  float rescale_grad;
  float clip_gradient;
  DMLC_DECLARE_PARAMETER(SGDParam) {
    DMLC_DECLARE_FIELD(momentum)
    .set_range(0.0f, 1.0f)
    .set_default(0.0f)
    .describe("momentum");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("rescale gradient as grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("If greater than 0, clip gradient to "
              "grad = max(min(grad, -clip_gradient), clip_gradient). "
              "Otherwise turned off.");
  }
};


struct sgd_clip {
  MSHADOW_XINLINE static real_t Map(real_t x, real_t bound) {
    if (x > bound) {
      return bound;
    } else if (x < -bound) {
      return -bound;
    } else {
      return x;
    }
  }
};

template<typename xpu>
void sgd_mom_update(RunContext ctx, TBlob weight, const TBlob grad, TBlob mom,
                float lr, float wd, const SGDParam& param) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  Tensor<xpu, 2> weight2d = weight.FlatTo2D<xpu, real_t>(s);
  Tensor<xpu, 2> mom2d = mom.FlatTo2D<xpu, real_t>(s);
  Tensor<xpu, 2> grad2d = grad.FlatTo2D<xpu, real_t>(s);
  if (param.clip_gradient > 0.0f) {
    mom2d = param.momentum*mom2d -
            (F<sgd_clip>(param.rescale_grad * grad2d, param.clip_gradient) + wd*weight2d);
  } else {
    mom2d = param.momentum*mom2d - (param.rescale_grad*grad2d + wd*weight2d);
  }
  weight2d += lr * mom2d;
}

template<typename xpu>
void sgd_update(RunContext ctx, TBlob weight, const TBlob grad,
                float lr, float wd, const SGDParam& param) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  Tensor<xpu, 2> weight2d = weight.FlatTo2D<xpu, real_t>(s);
  Tensor<xpu, 2> grad2d = grad.FlatTo2D<xpu, real_t>(s);
  if (param.clip_gradient >= 0.0f) {
    weight2d -= lr*(F<sgd_clip>(param.rescale_grad * grad2d, param.clip_gradient) +
                wd*weight2d);
  } else {
    weight2d -= lr*(param.rescale_grad*grad2d + wd*weight2d);
  }
}

void call_sgd_mom_update_cpu(RunContext ctx, TBlob weight, const TBlob grad, TBlob mom,
                float lr, float wd, const SGDParam& param);
void call_sgd_update_cpu(RunContext ctx, TBlob weight, const TBlob grad,
                float lr, float wd, const SGDParam& param);
#if MXNET_USE_CUDA
void call_sgd_mom_update_gpu(RunContext ctx, TBlob weight, const TBlob grad, TBlob mom,
                float lr, float wd, const SGDParam& param);
void call_sgd_update_gpu(RunContext ctx, TBlob weight, const TBlob grad,
                float lr, float wd, const SGDParam& param);
#endif  // MXNET_USE_CUDA

#if DMLC_USE_CXX11

class SGDOpt : public Optimizer {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  void CreateState(const int index, const NDArray *weight) override {
    if (param_.momentum > 0.0f && mom.find(index) == mom.end()) {
      mom[index] = NDArray(weight->shape(), weight->ctx());
      mom[index] = 0.0f;
    }
  }

  void Update(const int index, NDArray *weight,
              const NDArray *grad, const float lr, const float wd) override {
    NDArray w = *weight, g = *grad;
    CreateState(index, weight);
    switch (w.ctx().dev_type) {
     case Context::kCPU:
     case Context::kCPUPinned:
      if (param_.momentum > 0.0f) {
        Engine::Get()->PushSync([this, index, w, g, lr, wd](RunContext ctx) {
          call_sgd_mom_update_cpu(ctx, w.data(), g.data(), mom[index].data(), lr, wd, param_);
        }, w.ctx(), {g.var()}, {w.var(), mom[index].var()},
        FnProperty::kNormal, 0, "SGDOptUpdate");
      } else {
        Engine::Get()->PushSync([this, index, w, g, lr, wd](RunContext ctx) {
          call_sgd_update_cpu(ctx, w.data(), g.data(), lr, wd, param_);
        }, w.ctx(), {g.var()}, {w.var()},
        FnProperty::kNormal, 0, "SGDOptUpdate");
      }
      break;
     case Context::kGPU:
#if MXNET_USE_CUDA
      if (param_.momentum > 0.0f) {
        Engine::Get()->PushSync([this, index, w, g, lr, wd](RunContext ctx) {
          call_sgd_mom_update_gpu(ctx, w.data(), g.data(), mom[index].data(), lr, wd, param_);
        }, w.ctx(), {g.var()}, {w.var(), mom[index].var()},
        FnProperty::kNormal, 0, "SGDOptUpdate");
      } else {
        Engine::Get()->PushSync([this, index, w, g, lr, wd](RunContext ctx) {
          call_sgd_update_gpu(ctx, w.data(), g.data(), lr, wd, param_);
        }, w.ctx(), {g.var()}, {w.var()},
        FnProperty::kNormal, 0, "SGDOptUpdate");
      }
      break;
#else
        LOG(FATAL) << "Please compile with CUDA enabled for cuda features";
#endif  // MXNET_USE_CUDA
     default:
      LOG(FATAL) << "Unsupported device type for sgd optimizer: " << w.ctx().dev_type;
    }
  }

 private:
  SGDParam param_;
  std::map<int, NDArray> mom;
};

#endif  // DMLC_USE_CXX11

}  // namespace opt
}  // namespace mxnet
#endif  // MXNET_OPTIMIZER_SGD_INL_H_
