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
#include <map>
#include <string>
#include <fstream>
#include <cstdlib>
#include "utils.h"
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

Symbol AlexnetSymbol(int num_classes) {
  auto input_data = Symbol::Variable("data");
  auto target_label = Symbol::Variable("label");
  /*stage 1*/
  auto conv1 = Operator("Convolution")
                   .SetParam("kernel", Shape(11, 11))
                   .SetParam("num_filter", 96)
                   .SetParam("stride", Shape(4, 4))
                   .SetParam("dilate", Shape(1, 1))
                   .SetParam("pad", Shape(0, 0))
                   .SetParam("num_group", 1)
                   .SetParam("workspace", 512)
                   .SetParam("no_bias", false)
                   .SetInput("data", input_data)
                   .CreateSymbol("conv1");
  auto relu1 = Operator("Activation")
                   .SetParam("act_type", "relu") /*relu,sigmoid,softrelu,tanh */
                   .SetInput("data", conv1)
                   .CreateSymbol("relu1");
  auto pool1 = Operator("Pooling")
                   .SetParam("kernel", Shape(3, 3))
                   .SetParam("pool_type", "max") /*avg,max,sum */
                   .SetParam("global_pool", false)
                   .SetParam("stride", Shape(2, 2))
                   .SetParam("pad", Shape(0, 0))
                   .SetInput("data", relu1)
                   .CreateSymbol("pool1");
  auto lrn1 = Operator("LRN")
                  .SetParam("nsize", 5)
                  .SetParam("alpha", 0.0001)
                  .SetParam("beta", 0.75)
                  .SetParam("knorm", 1)
                  .SetInput("data", pool1)
                  .CreateSymbol("lrn1");
  /*stage 2*/
  auto conv2 = Operator("Convolution")
                   .SetParam("kernel", Shape(5, 5))
                   .SetParam("num_filter", 256)
                   .SetParam("stride", Shape(1, 1))
                   .SetParam("dilate", Shape(1, 1))
                   .SetParam("pad", Shape(2, 2))
                   .SetParam("num_group", 1)
                   .SetParam("workspace", 512)
                   .SetParam("no_bias", false)
                   .SetInput("data", lrn1)
                   .CreateSymbol("conv2");
  auto relu2 = Operator("Activation")
                   .SetParam("act_type", "relu") /*relu,sigmoid,softrelu,tanh */
                   .SetInput("data", conv2)
                   .CreateSymbol("relu2");
  auto pool2 = Operator("Pooling")
                   .SetParam("kernel", Shape(3, 3))
                   .SetParam("pool_type", "max") /*avg,max,sum */
                   .SetParam("global_pool", false)
                   .SetParam("stride", Shape(2, 2))
                   .SetParam("pad", Shape(0, 0))
                   .SetInput("data", relu2)
                   .CreateSymbol("pool2");
  auto lrn2 = Operator("LRN")
                  .SetParam("nsize", 5)
                  .SetParam("alpha", 0.0001)
                  .SetParam("beta", 0.75)
                  .SetParam("knorm", 1)
                  .SetInput("data", pool2)
                  .CreateSymbol("lrn2");
  /*stage 3*/
  auto conv3 = Operator("Convolution")
                   .SetParam("kernel", Shape(3, 3))
                   .SetParam("num_filter", 384)
                   .SetParam("stride", Shape(1, 1))
                   .SetParam("dilate", Shape(1, 1))
                   .SetParam("pad", Shape(1, 1))
                   .SetParam("num_group", 1)
                   .SetParam("workspace", 512)
                   .SetParam("no_bias", false)
                   .SetInput("data", lrn2)
                   .CreateSymbol("conv3");
  auto relu3 = Operator("Activation")
                   .SetParam("act_type", "relu") /*relu,sigmoid,softrelu,tanh */
                   .SetInput("data", conv3)
                   .CreateSymbol("relu3");
  auto conv4 = Operator("Convolution")
                   .SetParam("kernel", Shape(3, 3))
                   .SetParam("num_filter", 384)
                   .SetParam("stride", Shape(1, 1))
                   .SetParam("dilate", Shape(1, 1))
                   .SetParam("pad", Shape(1, 1))
                   .SetParam("num_group", 1)
                   .SetParam("workspace", 512)
                   .SetParam("no_bias", false)
                   .SetInput("data", relu3)
                   .CreateSymbol("conv4");
  auto relu4 = Operator("Activation")
                   .SetParam("act_type", "relu") /*relu,sigmoid,softrelu,tanh */
                   .SetInput("data", conv4)
                   .CreateSymbol("relu4");
  auto conv5 = Operator("Convolution")
                   .SetParam("kernel", Shape(3, 3))
                   .SetParam("num_filter", 256)
                   .SetParam("stride", Shape(1, 1))
                   .SetParam("dilate", Shape(1, 1))
                   .SetParam("pad", Shape(1, 1))
                   .SetParam("num_group", 1)
                   .SetParam("workspace", 512)
                   .SetParam("no_bias", false)
                   .SetInput("data", relu4)
                   .CreateSymbol("conv5");
  auto relu5 = Operator("Activation")
                   .SetParam("act_type", "relu")
                   .SetInput("data", conv5)
                   .CreateSymbol("relu5");
  auto pool3 = Operator("Pooling")
                   .SetParam("kernel", Shape(3, 3))
                   .SetParam("pool_type", "max")
                   .SetParam("global_pool", false)
                   .SetParam("stride", Shape(2, 2))
                   .SetParam("pad", Shape(0, 0))
                   .SetInput("data", relu5)
                   .CreateSymbol("pool3");
  /*stage4*/
  auto flatten =
      Operator("Flatten").SetInput("data", pool3).CreateSymbol("flatten");
  auto fc1 = Operator("FullyConnected")
                 .SetParam("num_hidden", 4096)
                 .SetParam("no_bias", false)
                 .SetInput("data", flatten)
                 .CreateSymbol("fc1");
  auto relu6 = Operator("Activation")
                   .SetParam("act_type", "relu")
                   .SetInput("data", fc1)
                   .CreateSymbol("relu6");
  auto dropout1 = Operator("Dropout")
                      .SetParam("p", 0.5)
                      .SetInput("data", relu6)
                      .CreateSymbol("dropout1");
  /*stage5*/
  auto fc2 = Operator("FullyConnected")
                 .SetParam("num_hidden", 4096)
                 .SetParam("no_bias", false)
                 .SetInput("data", dropout1)
                 .CreateSymbol("fc2");
  auto relu7 = Operator("Activation")
                   .SetParam("act_type", "relu")
                   .SetInput("data", fc2)
                   .CreateSymbol("relu7");
  auto dropout2 = Operator("Dropout")
                      .SetParam("p", 0.5)
                      .SetInput("data", relu7)
                      .CreateSymbol("dropout2");
  /*stage6*/
  auto fc3 = Operator("FullyConnected")
                 .SetParam("num_hidden", num_classes)
                 .SetParam("no_bias", false)
                 .SetInput("data", dropout2)
                 .CreateSymbol("fc3");
  auto softmax = Operator("SoftmaxOutput")
                     .SetParam("grad_scale", 1)
                     .SetParam("ignore_label", -1)
                     .SetParam("multi_output", false)
                     .SetParam("use_ignore", false)
                     .SetParam("normalization", "null") /*batch,null,valid */
                     .SetInput("data", fc3)
                     .SetInput("label", target_label)
                     .CreateSymbol("softmax");
  return softmax;
}

NDArray ResizeInput(NDArray data, const Shape new_shape) {
  NDArray pic = data.Reshape(Shape(0, 1, 28, 28));
  NDArray pic_1channel;
  Operator("_contrib_BilinearResize2D")
    .SetParam("height", new_shape[2])
    .SetParam("width", new_shape[3])
    (pic).Invoke(pic_1channel);
  NDArray output;
  Operator("tile")
    .SetParam("reps", Shape(1, 3, 1, 1))
    (pic_1channel).Invoke(output);
  return output;
}

int main(int argc, char const *argv[]) {
  /*basic config*/
  int max_epo = argc > 1 ? strtol(argv[1], nullptr, 10) : 100;
  float learning_rate = 1e-4;
  float weight_decay = 1e-4;

  /*context*/
  auto ctx = Context::cpu();
  int num_gpu;
  MXGetGPUCount(&num_gpu);
  int batch_size = 32;
#if !MXNET_USE_CPU
  if (num_gpu > 0) {
    ctx = Context::gpu();
    batch_size = 256;
  }
#endif

  TRY
  /*net symbol*/
  auto Net = AlexnetSymbol(10);

  /*args_map and aux_map is used for parameters' saving*/
  std::map<std::string, NDArray> args_map;
  std::map<std::string, NDArray> aux_map;

  /*we should tell mxnet the shape of data and label*/
  const Shape data_shape = Shape(batch_size, 3, 256, 256),
              label_shape = Shape(batch_size);
  args_map["data"] = NDArray(data_shape, ctx);
  args_map["label"] = NDArray(label_shape, ctx);

  /*with data and label, executor can be generated automatically*/
  auto *exec = Net.SimpleBind(ctx, args_map);
  auto arg_names = Net.ListArguments();
  aux_map = exec->aux_dict();
  args_map = exec->arg_dict();

  /*if fine tune from some pre-trained model, we should load the parameters*/
  // NDArray::Load("./model/alex_params_3", nullptr, &args_map);
  /*else, we should use initializer Xavier to init the params*/
  auto initializer = Uniform(0.07);
  for (auto &arg : args_map) {
    /*be careful here, the arg's name must has some specific ends or starts for
     * initializer to call*/
    initializer(arg.first, &arg.second);
  }

  /*these binary files should be generated using im2rc tools, which can be found
   * in mxnet/bin*/
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

  Optimizer* opt = OptimizerRegistry::Find("sgd");
  opt->SetParam("momentum", 0.9)
     ->SetParam("rescale_grad", 1.0 / batch_size)
     ->SetParam("clip_gradient", 10)
     ->SetParam("lr", learning_rate)
     ->SetParam("wd", weight_decay);

  Accuracy acu_train, acu_val;
  LogLoss logloss_train, logloss_val;
  for (int epoch = 0; epoch < max_epo; ++epoch) {
    LG << "Train Epoch: " << epoch;
    /*reset the metric every epoch*/
    acu_train.Reset();
    /*reset the data iter every epoch*/
    train_iter.Reset();
    int iter = 0;
    while (train_iter.Next()) {
      auto batch = train_iter.GetDataBatch();
      /*use copyto to feed new data and label to the executor*/
      ResizeInput(batch.data, data_shape).CopyTo(&args_map["data"]);
      batch.label.CopyTo(&args_map["label"]);
      exec->Forward(true);
      exec->Backward();
      for (size_t i = 0; i < arg_names.size(); ++i) {
        if (arg_names[i] == "data" || arg_names[i] == "label") continue;
        opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
      }

      NDArray::WaitAll();
      acu_train.Update(batch.label, exec->outputs[0]);
      logloss_train.Reset();
      logloss_train.Update(batch.label, exec->outputs[0]);
      ++iter;
      LG << "EPOCH: " << epoch << " ITER: " << iter
         << " Train Accuracy: " << acu_train.Get()
         << " Train Loss: " << logloss_train.Get();
    }
    LG << "EPOCH: " << epoch << " Train Accuracy: " << acu_train.Get();

    LG << "Val Epoch: " << epoch;
    acu_val.Reset();
    val_iter.Reset();
    logloss_val.Reset();
    iter = 0;
    while (val_iter.Next()) {
      auto batch = val_iter.GetDataBatch();
      ResizeInput(batch.data, data_shape).CopyTo(&args_map["data"]);
      batch.label.CopyTo(&args_map["label"]);
      exec->Forward(false);
      NDArray::WaitAll();
      acu_val.Update(batch.label, exec->outputs[0]);
      logloss_val.Update(batch.label, exec->outputs[0]);
      LG << "EPOCH: " << epoch << " ITER: " << iter << " Val Accuracy: " << acu_val.Get();
      ++iter;
    }
    LG << "EPOCH: " << epoch << " Val Accuracy: " << acu_val.Get();
    LG << "EPOCH: " << epoch << " Val LogLoss: " << logloss_val.Get();

    /*save the parameters*/
    std::stringstream ss;
    ss << epoch;
    std::string epoch_str;
    ss >> epoch_str;
    std::string save_path_param = "alex_param_" + epoch_str;
    auto save_args = args_map;
    /*we do not want to save the data and label*/
    save_args.erase(save_args.find("data"));
    save_args.erase(save_args.find("label"));
    /*the alexnet does not get any aux array, so we do not need to save
     * aux_map*/
    LG << "EPOCH: " << epoch << " Saving to..." << save_path_param;
    NDArray::Save(save_path_param, save_args);
  }
  /*don't foget to release the executor*/
  delete exec;
  delete opt;
  MXNotifyShutdown();
  CATCH
  return 0;
}
