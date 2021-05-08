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
 * Xin Li yakumolx@gmail.com
 */
#include <chrono>
#include "utils.h"
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

Symbol mlp(const std::vector<int> &layers) {
  auto x = Symbol::Variable("X");
  auto label = Symbol::Variable("label");

  std::vector<Symbol> weights(layers.size());
  std::vector<Symbol> biases(layers.size());
  std::vector<Symbol> outputs(layers.size());

  for (size_t i = 0; i < layers.size(); ++i) {
    weights[i] = Symbol::Variable("w" + std::to_string(i));
    biases[i] = Symbol::Variable("b" + std::to_string(i));
    Symbol fc = FullyConnected(
      i == 0? x : outputs[i-1],  // data
      weights[i],
      biases[i],
      layers[i]);
    outputs[i] = i == layers.size()-1 ? fc : Activation(fc, ActivationActType::kRelu);
  }

  return SoftmaxOutput(outputs.back(), label);
}

int main(int argc, char** argv) {
  const int image_size = 28;
  const std::vector<int> layers{128, 64, 10};
  const int batch_size = 100;
  const int max_epoch = 10;
  const float learning_rate = 0.1;
  const float weight_decay = 1e-2;

  std::vector<std::string> data_files = { "./data/mnist_data/train-images-idx3-ubyte",
                                          "./data/mnist_data/train-labels-idx1-ubyte",
                                          "./data/mnist_data/t10k-images-idx3-ubyte",
                                          "./data/mnist_data/t10k-labels-idx1-ubyte"
                                        };

  auto train_iter =  MXDataIter("MNISTIter");
  if (!setDataIter(&train_iter, "Train", data_files, batch_size)) {
    return 1;
  }

  auto val_iter = MXDataIter("MNISTIter");
  if (!setDataIter(&val_iter, "Label", data_files, batch_size)) {
    return 1;
  }

  TRY
  auto net = mlp(layers);

  Context ctx = Context::gpu();  // Use GPU for training

  std::map<std::string, NDArray> args;
  args["X"] = NDArray(Shape(batch_size, image_size*image_size), ctx);
  args["label"] = NDArray(Shape(batch_size), ctx);
  // Let MXNet infer shapes of other parameters such as weights
  net.InferArgsMap(ctx, &args, args);

  // Initialize all parameters with uniform distribution U(-0.01, 0.01)
  auto initializer = Uniform(0.01);
  for (auto& arg : args) {
    // arg.first is parameter name, and arg.second is the value
    initializer(arg.first, &arg.second);
  }

  // Create sgd optimizer
  Optimizer* opt = OptimizerRegistry::Find("sgd");
  opt->SetParam("rescale_grad", 1.0/batch_size)
     ->SetParam("lr", learning_rate)
     ->SetParam("wd", weight_decay);
  std::unique_ptr<LRScheduler> lr_sch(new FactorScheduler(5000, 0.1));
  opt->SetLRScheduler(std::move(lr_sch));

  // Create executor by binding parameters to the model
  auto *exec = net.SimpleBind(ctx, args);
  auto arg_names = net.ListArguments();

  // Create metrics
  Accuracy train_acc, val_acc;

  // Start training
  for (int iter = 0; iter < max_epoch; ++iter) {
    int samples = 0;
    train_iter.Reset();
    train_acc.Reset();

    auto tic = std::chrono::system_clock::now();
    while (train_iter.Next()) {
      samples += batch_size;
      auto data_batch = train_iter.GetDataBatch();
      // Data provided by DataIter are stored in memory, should be copied to GPU first.
      data_batch.data.CopyTo(&args["X"]);
      data_batch.label.CopyTo(&args["label"]);
      // CopyTo is imperative, need to wait for it to complete.
      NDArray::WaitAll();

      // Compute gradients
      exec->Forward(true);
      exec->Backward();

      // Update parameters
      for (size_t i = 0; i < arg_names.size(); ++i) {
        if (arg_names[i] == "X" || arg_names[i] == "label") continue;
        opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
      }
      // Update metric
      train_acc.Update(data_batch.label, exec->outputs[0]);
    }
    // one epoch of training is finished
    auto toc = std::chrono::system_clock::now();
    float duration = std::chrono::duration_cast<std::chrono::milliseconds>
                     (toc - tic).count() / 1000.0;
    LG << "Epoch[" << iter << "] " << samples/duration \
       << " samples/sec " << "Train-Accuracy=" << train_acc.Get();;

    val_iter.Reset();
    val_acc.Reset();
    while (val_iter.Next()) {
      auto data_batch = val_iter.GetDataBatch();
      data_batch.data.CopyTo(&args["X"]);
      data_batch.label.CopyTo(&args["label"]);
      NDArray::WaitAll();

      // Only forward pass is enough as no gradient is needed when evaluating
      exec->Forward(false);
      val_acc.Update(data_batch.label, exec->outputs[0]);
    }
    LG << "Epoch[" << iter << "] Val-Accuracy=" << val_acc.Get();
  }

  delete exec;
  delete opt;
  MXNotifyShutdown();
  CATCH
  return 0;
}
