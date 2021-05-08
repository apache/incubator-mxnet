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
 *  \file dropout_perf.cc
 *  \brief Perf/profile run of DropoutOp
 *  \author Chris Olivier
 */

#include <gtest/gtest.h>
#include <mxnet/tensor_blob.h>
#include "../include/test_op_runner.h"
#include "../include/test_core_op.h"
#include "../../src/operator/nn/dropout-inl.h"

using namespace mxnet;

typedef std::vector<std::pair<std::string, std::string> > kwargs_t;
const kwargs_t basic_dropout_args = { };

/*!
 * \brief Generic bidirectional sanity test
 */
TEST(DROPOUT_PERF, ExecuteBidirectional) {
  mxnet::TShape shape({5, 5});
  kwargs_t kwargs = basic_dropout_args;
  kwargs.push_back({"mode", "always"});
  test::op::CoreOperatorRunner<float> runner;
  kwargs = test::op::CoreOpExecutor<float>::ArgsWithOpName(kwargs, "Dropout",
                                                           "_backward_Dropout");
  runner.set_verbose(true);
  runner.RunGenericOperatorForward(false, { shape }, kwargs, 1);
}

/*!
 * \brief DropoutOp timing test for CPU
 */
TEST(DROPOUT_PERF, TimingCPU) {
  kwargs_t kwargs = basic_dropout_args;
// Which math function is arbitrary since it will have roughly constant timing among approaches
  kwargs.push_back({"mode", "always"});
  mxnet::TShape shape({10, 10, 10, 10});
  test::op::CoreOperatorRunner<float> runner;
  kwargs = test::op::CoreOpExecutor<float>::ArgsWithOpName(kwargs, "Dropout",
                                                           "_backward_Dropout");
  runner.RunGenericOperatorForward(false, { shape }, kwargs, 1);
  std::vector <mxnet::TShape> shapes;
  if (test::performance_run) {
    shapes = {
      {1,  1, 28,  28},
      {1,  3, 28,  28},
      {50, 1, 18,  32},
      {50, 3, 18,  32},
      {20, 3, 128, 128}
    };
  } else {
    shapes = {
      {1,  1, 28,  28},
      {50, 3, 18,  32},
    };
  }
  for (const mxnet::TShape &shape : shapes) {
    kwargs = test::op::CoreOpExecutor<float>::ArgsWithOpName(kwargs, "Dropout",
                                                             "_backward_Dropout");
    runner.TimingTest("Dropout Operator CPU", false, false, kwargs, 2, 10, { shape }, false);
  }
}

#if MXNET_USE_CUDA == 1
/*!
 * \brief DropoutOp timing test for GPU
 */
TEST(DROPOUT_PERF, TimingGPU) {
  kwargs_t kwargs = basic_dropout_args;
  // Which math function is arbitrary since it will have roughly constant timing among approaches
  kwargs.push_back({"mode", "always"});
  mxnet::TShape shape({10, 10, 10, 10});
  test::op::CoreOperatorRunner<float> runner;
  kwargs = test::op::CoreOpExecutor<float>::ArgsWithOpName(kwargs, "Dropout",
                                                           "_backward_Dropout");
  runner.RunGenericOperatorForward(false, { shape }, kwargs, 1);
  std::vector <mxnet::TShape> shapes = {
    {1,  1, 28,  28},
    {1,  3, 28,  28},
    {50, 1, 18,  32},
    {50, 3, 18,  32},
    {20, 3, 128, 128}
  };
  for (const mxnet::TShape &shape : shapes) {
    kwargs = test::op::CoreOpExecutor<float>::ArgsWithOpName(kwargs, "Dropout",
                                                             "_backward_Dropout");
    runner.TimingTest("Dropout Operator GPU", true, false, kwargs, 2, 10, { shape }, false);
  }
}
#endif  // MXNET_USE_CUDA == 1
