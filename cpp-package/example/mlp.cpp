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
 */

#include <iostream>
#include <vector>
#include <string>
#include "mxnet-cpp/MxNetCpp.h"
#include "utils.h"

using namespace mxnet::cpp;

/*
 * In this example,
 * we make by hand some data in 10 classes with some pattern
 * and try to use MLP to recognize the pattern.
 */

void OutputAccuracy(mx_float* pred, mx_float* target) {
  int right = 0;
  for (int i = 0; i < 128; ++i) {
    float mx_p = pred[i * 10 + 0];
    float p_y = 0;
    for (int j = 0; j < 10; ++j) {
      if (pred[i * 10 + j] > mx_p) {
        mx_p = pred[i * 10 + j];
        p_y = j;
      }
    }
    if (p_y == target[i]) right++;
  }
  std::cout << "Accuracy: " << right / 128.0 << std::endl;
}

void MLP(int max_epoch) {
  auto sym_x = Symbol::Variable("X");
  auto sym_label = Symbol::Variable("label");

  const int nLayers = 2;
  std::vector<int> layerSizes({512, 10});
  std::vector<Symbol> weights(nLayers);
  std::vector<Symbol> biases(nLayers);
  std::vector<Symbol> outputs(nLayers);

  Symbol null_sym;
  for (int i = 0; i < nLayers; i++) {
    std::string istr = std::to_string(i);
    weights[i] = Symbol::Variable(std::string("w") + istr);
    biases[i] = Symbol::Variable(std::string("b") + istr);
    Symbol fc = FullyConnected(std::string("fc") + istr,
      i == 0? sym_x : outputs[i-1],
      weights[i], biases[i], layerSizes[i]);
    outputs[i] = LeakyReLU(std::string("act") + istr, fc, null_sym, LeakyReLUActType::kLeaky);
  }
  auto sym_out = SoftmaxOutput("softmax", outputs[nLayers - 1], sym_label);

  Context ctx_dev(DeviceType::kCPU, 0);

  NDArray array_x(Shape(128, 28), ctx_dev, false);
  NDArray array_y(Shape(128), ctx_dev, false);

  mx_float* aptr_x = new mx_float[128 * 28];
  mx_float* aptr_y = new mx_float[128];

  // we make the data by hand, in 10 classes, with some pattern
  for (int i = 0; i < 128; i++) {
    for (int j = 0; j < 28; j++) {
      aptr_x[i * 28 + j] = i % 10 * 1.0f;
    }
    aptr_y[i] = i % 10;
  }
  array_x.SyncCopyFromCPU(aptr_x, 128 * 28);
  array_x.WaitToRead();
  array_y.SyncCopyFromCPU(aptr_y, 128);
  array_y.WaitToRead();

  // init the parameters
  NDArray array_w_1(Shape(512, 28), ctx_dev, false);
  NDArray array_b_1(Shape(512), ctx_dev, false);
  NDArray array_w_2(Shape(10, 512), ctx_dev, false);
  NDArray array_b_2(Shape(10), ctx_dev, false);

  // the parameters should be initialized in some kind of distribution,
  // so it learns fast
  // but here just give a const value by hand
  array_w_1 = 0.5f;
  array_b_1 = 0.0f;
  array_w_2 = 0.5f;
  array_b_2 = 0.0f;

  // the grads
  NDArray array_w_1_g(Shape(512, 28), ctx_dev, false);
  NDArray array_b_1_g(Shape(512), ctx_dev, false);
  NDArray array_w_2_g(Shape(10, 512), ctx_dev, false);
  NDArray array_b_2_g(Shape(10), ctx_dev, false);

  // Bind the symolic network with the ndarray
  // all the input args
  std::vector<NDArray> in_args;
  in_args.push_back(array_x);
  in_args.push_back(array_w_1);
  in_args.push_back(array_b_1);
  in_args.push_back(array_w_2);
  in_args.push_back(array_b_2);
  in_args.push_back(array_y);
  // all the grads
  std::vector<NDArray> arg_grad_store;
  arg_grad_store.push_back(NDArray());  // we don't need the grad of the input
  arg_grad_store.push_back(array_w_1_g);
  arg_grad_store.push_back(array_b_1_g);
  arg_grad_store.push_back(array_w_2_g);
  arg_grad_store.push_back(array_b_2_g);
  arg_grad_store.push_back(
      NDArray());  // neither do we need the grad of the loss
  // how to handle the grad
  std::vector<OpReqType> grad_req_type;
  grad_req_type.push_back(kNullOp);
  grad_req_type.push_back(kWriteTo);
  grad_req_type.push_back(kWriteTo);
  grad_req_type.push_back(kWriteTo);
  grad_req_type.push_back(kWriteTo);
  grad_req_type.push_back(kNullOp);
  std::vector<NDArray> aux_states;

  std::cout << "make the Executor" << std::endl;
  Executor* exe = new Executor(sym_out, ctx_dev, in_args, arg_grad_store,
                               grad_req_type, aux_states);

  std::cout << "Training" << std::endl;
  mx_float learning_rate = 0.0001;
  for (int epoch_num = 0; epoch_num < max_epoch; ++epoch_num) {
    exe->Forward(true);
    // print accuracy every 100 epoch
    if (epoch_num % 100 == 0) {
      std::cout << "epoch " << epoch_num << std::endl;
      std::vector<NDArray>& out = exe->outputs;
      float* cptr = new float[128 * 10];
      out[0].SyncCopyToCPU(cptr, 128 * 10);
      NDArray::WaitAll();
      OutputAccuracy(cptr, aptr_y);
      delete[] cptr;
    }

    // update the parameters
    exe->Backward();
    for (int i = 1; i < 5; ++i) {
      in_args[i] -= arg_grad_store[i] * learning_rate;
    }
    NDArray::WaitAll();
  }

  delete exe;
  delete[] aptr_x;
  delete[] aptr_y;
}

int main(int argc, char** argv) {
  int max_epoch = argc > 1 ? strtol(argv[1], nullptr, 10) : 15000;
  TRY
  MLP(max_epoch);
  MXNotifyShutdown();
  CATCH
  return 0;
}
