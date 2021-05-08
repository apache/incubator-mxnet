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
 *  Copyright (c) 2016 by Contributors
 * \file initializer.h
 * \brief random initializer
 * \author Zhang Chen
 */

#ifndef MXNET_CPP_INITIALIZER_H_
#define MXNET_CPP_INITIALIZER_H_

#include <cmath>
#include <string>
#include <vector>
#include <random>
#include "mxnet-cpp/ndarray.h"

namespace mxnet {
namespace cpp {

class Initializer {
 public:
  static bool StringStartWith(const std::string& name,
                              const std::string& check_str) {
    return (name.size() >= check_str.size() &&
            name.substr(0, check_str.size()) == check_str);
  }
  static bool StringEndWith(const std::string& name,
                            const std::string& check_str) {
    return (name.size() >= check_str.size() &&
            name.substr(name.size() - check_str.size(), check_str.size()) ==
                check_str);
  }
  virtual void operator()(const std::string& name, NDArray* arr) {
    if (StringStartWith(name, "upsampling")) {
      InitBilinear(arr);
    } else if (StringEndWith(name, "bias")) {
      InitBias(arr);
    } else if (StringEndWith(name, "gamma")) {
      InitGamma(arr);
    } else if (StringEndWith(name, "beta")) {
      InitBeta(arr);
    } else if (StringEndWith(name, "weight")) {
      InitWeight(arr);
    } else if (StringEndWith(name, "moving_mean")) {
      InitZero(arr);
    } else if (StringEndWith(name, "moving_var")) {
      InitOne(arr);
    } else if (StringEndWith(name, "moving_inv_var")) {
      InitZero(arr);
    } else if (StringEndWith(name, "moving_avg")) {
      InitZero(arr);
    } else if (StringEndWith(name, "min")) {
      InitZero(arr);
    } else if (StringEndWith(name, "max")) {
      InitOne(arr);
    } else if (StringEndWith(name, "weight_quantize")) {
      InitQuantizedWeight(arr);
    } else if (StringEndWith(name, "bias_quantize")) {
      InitQuantizedBias(arr);
    } else {
      InitDefault(arr);
    }
  }

 protected:
  virtual void InitBilinear(NDArray* arr) {
    Shape shape(arr->GetShape());
    std::vector<float> weight(shape.Size(), 0);
    int f = std::ceil(shape[3] / 2.0);
    float c = (2 * f - 1 - f % 2) / (2. * f);
    for (size_t i = 0; i < shape.Size(); ++i) {
      int x = i % shape[3];
      int y = (i / shape[3]) % shape[2];
      weight[i] = (1 - std::abs(x / f - c)) * (1 - std::abs(y / f - c));
    }
    (*arr).SyncCopyFromCPU(weight);
  }
  virtual void InitZero(NDArray* arr) { (*arr) = 0.0f; }
  virtual void InitOne(NDArray* arr) { (*arr) = 1.0f; }
  virtual void InitBias(NDArray* arr) { (*arr) = 0.0f; }
  virtual void InitGamma(NDArray* arr) { (*arr) = 1.0f; }
  virtual void InitBeta(NDArray* arr) { (*arr) = 0.0f; }
  virtual void InitWeight(NDArray* arr) {}
  virtual void InitQuantizedWeight(NDArray* arr) {
    std::default_random_engine generator;
    std::uniform_int_distribution<int32_t> _val(-127, 127);
    (*arr) = _val(generator);
  }
  virtual void InitQuantizedBias(NDArray* arr) {
    (*arr) = 0;
  }
  virtual void InitDefault(NDArray* arr) {}
};

class Constant : public Initializer {
 public:
  explicit Constant(float value)
    : value(value) {}
  void operator()(const std::string &name, NDArray *arr) override {
    (*arr) = value;
  }
 protected:
  float value;
};

class Zero : public Constant {
 public:
  Zero(): Constant(0.0f) {}
};

class One : public Constant {
 public:
  One(): Constant(1.0f) {}
};

class Uniform : public Initializer {
 public:
  explicit Uniform(float scale)
    : Uniform(-scale, scale) {}
  Uniform(float begin, float end)
    : begin(begin), end(end) {}
  void operator()(const std::string &name, NDArray *arr) override {
    if (StringEndWith(name, "weight_quantize")) {
      InitQuantizedWeight(arr);
      return;
    }
    if (StringEndWith(name, "bias_quantize")) {
      InitQuantizedBias(arr);
      return;
    }
    NDArray::SampleUniform(begin, end, arr);
  }
 protected:
  float begin, end;
};

class Normal : public Initializer {
 public:
  Normal(float mu, float sigma)
    : mu(mu), sigma(sigma) {}
  void operator()(const std::string &name, NDArray *arr) override {
    if (StringEndWith(name, "weight_quantize")) {
      InitQuantizedWeight(arr);
      return;
    }
    if (StringEndWith(name, "bias_quantize")) {
      InitQuantizedBias(arr);
      return;
    }
    NDArray::SampleGaussian(mu, sigma, arr);
  }
 protected:
  float mu, sigma;
};

class Bilinear : public Initializer {
 public:
  Bilinear() {}
  void operator()(const std::string &name, NDArray *arr) override {
    if (StringEndWith(name, "weight_quantize")) {
      InitQuantizedWeight(arr);
      return;
    }
    if (StringEndWith(name, "bias_quantize")) {
      InitQuantizedBias(arr);
      return;
    }
    InitBilinear(arr);
  }
};

class Xavier : public Initializer {
 public:
  enum RandType {
    gaussian,
    uniform
  } rand_type;
  enum FactorType {
    avg,
    in,
    out
  } factor_type;
  float magnitude;
  Xavier(RandType rand_type = gaussian, FactorType factor_type = avg,
         float magnitude = 3)
      : rand_type(rand_type), factor_type(factor_type), magnitude(magnitude) {}

  void operator()(const std::string &name, NDArray* arr) override {
    if (StringEndWith(name, "weight_quantize")) {
      InitQuantizedWeight(arr);
      return;
    }
    if (StringEndWith(name, "bias_quantize")) {
      InitQuantizedBias(arr);
      return;
    }

    Shape shape(arr->GetShape());
    float hw_scale = 1.0f;
    if (shape.ndim() > 2) {
      for (size_t i = 2; i < shape.ndim(); ++i) {
        hw_scale *= shape[i];
      }
    }
    float fan_in = shape[1] * hw_scale, fan_out = shape[0] * hw_scale;
    float factor = 1.0f;
    switch (factor_type) {
      case avg:
        factor = (fan_in + fan_out) / 2.0;
        break;
      case in:
        factor = fan_in;
        break;
      case out:
        factor = fan_out;
    }
    float scale = std::sqrt(magnitude / factor);
    switch (rand_type) {
      case uniform:
        NDArray::SampleUniform(-scale, scale, arr);
        break;
      case gaussian:
        NDArray::SampleGaussian(0, scale, arr);
        break;
    }
  }
};

class MSRAPrelu : public Xavier {
 public:
  explicit MSRAPrelu(FactorType factor_type = avg, float slope = 0.25f)
      : Xavier(gaussian, factor_type, 2. / (1 + slope * slope)) {}
};

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_INITIALIZER_H_
