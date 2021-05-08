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
#include <gtest/gtest.h>
#include <mxnet/tensor_blob.h>
#include "../../src/operator/nn/activation-inl.h"
#include "../../src/operator/operator_tune-inl.h"
#include "../include/test_op_runner.h"
#include "../include/test_core_op.h"
#include "../include/test_tune.h"

#if MXNET_USE_OPERATOR_TUNING

using namespace mxnet;

/*!
 * \brief ActivationOp timing test for CPU
 */
TEST(OMP_TUNING, ShowAllTunedOps) {
  const std::unordered_set<std::string>& op_names =
    mxnet::op::OperatorTune<float>::TunedOperatorNames();
  for (auto iter = op_names.begin(), e_iter = op_names.end(); iter != e_iter; ++iter) {
    std::cout << *iter << std::endl;
  }
}

using kwargs_t = test::op::kwargs_t;

static std::vector<mxnet::ShapeVector> tuning_shapes() {
  std::vector<mxnet::ShapeVector> shapes;
  if (test::performance_run || test::csv) {
    shapes = {
      {{1,  1, 28,  28}},
      {{1,  3, 28,  28}},
      {{50, 1, 18,  32}},
      {{25, 3, 64,  64}},
      {{10, 3, 128, 128}},
      {{20, 3, 128, 128}},
      {{30, 3, 128, 128}},
      {{30, 3, 256, 128}},
    };
  } else {
    shapes = {
      // Non-performance dataset acts as a sanity test
      {{1,  1, 28, 28}},
      {{50, 3, 18, 32}}
    };
  }
  return shapes;
}

/*!
 * \brief Generic bidirectional sanity test
 */
TEST(OMP_TUNING, ExecuteBidirectional) {
  test::op::BasicRunCoreOpBidirectional(false, true, {}, {tuning_shapes()[0]},
                                        "elemwise_add", "_backward_add");
}

/* Some test results:
 * AWS c4.8xlarge:
  Success rate for type float: 0.90278
  Success rate for type double: 0.88889
  Success rate for type mshadow::half::half_t: 0.83333
  Success rate for type unsigned char: 0.86111
  Success rate for type int: 0.95833
  Success rate for type long: 0.88889
 * desktop: 12-core (6 real CPU cores + hyperthreading)
  Success rate for type float: 0.78125
  Success rate for type double: 0.85417
  Success rate for type mshadow::half::half_t: 0.84375
  Success rate for type unsigned char: 0.80208
  Success rate for type int: 0.94444
  Success rate for type long: 1.00000
 */

/*!
 * \brief Rune a tuning evaluation
 * \tparam DType Data type for which to evaluate tuning
 */
template<typename DType>
static float EvaluateTune(const bool verbose = true) {
  std::vector<std::pair<std::string, std::string>> binary_operators;
  if (test::csv) {
    binary_operators = {
      {"elemwise_add", COREOP_BWD_OP_NAME_VALUE_NONE}
    };
  } else if (test::performance_run) {
    binary_operators = {
      {"relu",         ""},  // Code can figure out what the backward op is for some
      {"sigmoid",      ""},
      {"sqrt",         ""},
      {"elemwise_add", "_backward_add"},
      {"elemwise_mul", ""},
      {"elemwise_div", "_backward_div"}
    };
  } else {
    binary_operators = {
      {"elemwise_add", "_backward_add"}
    };
  }
  std::vector<float> rates;
  for (size_t i = 0, n = binary_operators.size(); i < n; ++i) {
    test::tune::TuningTester<DType> tuningTester;
    tuningTester.set_calls_per_iteration(10);
    tuningTester.set_total_iterations(5);
    std::cout << "******************************" << std::endl;
    std::cout << "Operators: " << binary_operators[i].first
              << ", " << binary_operators[i].second
              << " for type: " << test::type_name<DType>()
              << std::endl;
    std::cout << "******************************" << std::endl;

    // Do the performance runs
    std::vector<mxnet::ShapeVector> shapes = tuning_shapes();

    tuningTester.TestTunedOperator({}, verbose, shapes,
                                   binary_operators[i].first.c_str(),
                                   binary_operators[i].second.c_str());
    rates.push_back(tuningTester.CalculateSuccessRate());
  }
  return std::accumulate(rates.begin(), rates.end(), 0.0f) / rates.size();
}

/*! \brief ActivationOp timing test for CPU for float */
TEST(OMP_TUNING, EvaluateTuneTestFloat) {
  typedef float DType;
  const float result = EvaluateTune<DType>();
  std::cout << "Success rate for type " << test::type_name<DType>() << ": " << result << std::endl;
}
/*! \brief ActivationOp timing test for CPU for double */
TEST(OMP_TUNING, EvaluateTuneTestDouble) {
  typedef double DType;
  const float result = EvaluateTune<DType>();
  std::cout << "Success rate for type " << test::type_name<DType>() << ": " << result << std::endl;
}
/*! \brief ActivationOp timing test for CPU for float16 */
TEST(OMP_TUNING, EvaluateTuneTestFloat16) {
  typedef mshadow::half::half_t DType;
  const float result = EvaluateTune<DType>();
  std::cout << "Success rate for type " << test::type_name<DType>() << ": " << result << std::endl;
}
/*! \brief ActivationOp timing test for CPU for int8_t */
TEST(OMP_TUNING, EvaluateTuneTestInt8) {
  typedef uint8_t DType;
  const float result = EvaluateTune<DType>();
  std::cout << "Success rate for type " << test::type_name<DType>() << ": " << result << std::endl;
}
/*! \brief ActivationOp timing test for CPU for int32_t */
TEST(OMP_TUNING, EvaluateTuneTestInt32) {
  typedef int32_t DType;
  const float result = EvaluateTune<DType>();
  std::cout << "Success rate for type " << test::type_name<DType>() << ": " << result << std::endl;
}
/*! \brief ActivationOp timing test for CPU for int64_t */
TEST(OMP_TUNING, EvaluateTuneTestInt64) {
  typedef int64_t DType;
  const float result = EvaluateTune<DType>();
  std::cout << "Success rate for type " << test::type_name<DType>() << ": " << result << std::endl;
}

#endif  // MXNET_USE_OPERATOR_TUNING

