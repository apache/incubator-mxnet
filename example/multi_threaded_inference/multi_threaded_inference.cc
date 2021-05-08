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
 * Copyright (c) 2017 by Contributors
 * \file multi_threaded_inference.cc
 * \brief Multi Threaded inference example with CachedOp
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <thread>
#include <iomanip>
#include <chrono>
#include <mxnet/ndarray.h>
#include <opencv2/opencv.hpp>
#include <mxnet/c_predict_api.h>
#include "mxnet-cpp/MxNetCpp.h"
#include <random>

const float DEFAULT_MEAN = 117.0;


// Code to load image, PrintOutput results, helper functions for the same obtained from:
// https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/predict-cpp/

static std::string trim(const std::string &input) {
  auto not_space = [](int ch) { return !std::isspace(ch); };
  auto output = input;
  output.erase(output.begin(),
               std::find_if(output.begin(), output.end(), not_space));
  output.erase(std::find_if(output.rbegin(), output.rend(), not_space).base(),
               output.end());
  return output;
}

std::vector<std::string> LoadSynset(const std::string& synset_file) {
  std::ifstream fi(synset_file.c_str());

  if (!fi.is_open()) {
    std::cerr << "Error opening synset file " << synset_file << std::endl;
    assert(false);
  }

  std::vector<std::string> output;

  std::string synset, lemma;
  while (fi >> synset) {
    getline(fi, lemma);
    output.push_back(lemma);
  }

  fi.close();

  return output;
}

void PrintOutputResult(const float* data, size_t size, const std::vector<std::string>& synset) {
  if (size != synset.size()) {
    std::cerr << "Result data and synset size do not match!" << std::endl;
  }

  float best_accuracy = 0.0;
  std::size_t best_idx = 0;

  for (std::size_t i = 0; i < size; ++i) {
    if (data[i] > best_accuracy) {
      best_accuracy = data[i];
      best_idx = i;
    }
  }

  std::cout << "Best Result: " << trim(synset[best_idx]) << " (id=" << best_idx << ", " <<
            "accuracy=" << std::setprecision(8) << best_accuracy << ")" << std::endl;
}


// Read Image data into a float array
void GetImageFile(const std::string &image_file, float *image_data,
                  int channels, cv::Size resize_size) {
  // Read all kinds of file into a BGR color 3 channels image
  cv::Mat im_ori = cv::imread(image_file, cv::IMREAD_COLOR);

  if (im_ori.empty()) {
    std::cerr << "Can't open the image. Plase check " << image_file << ". \n";
    assert(false);
  }

  cv::Mat im;
  resize(im_ori, im, resize_size);

  int size = im.rows * im.cols * channels;

  float* ptr_image_r = image_data;
  float* ptr_image_g = image_data + size / 3;
  float* ptr_image_b = image_data + size / 3 * 2;

  float mean_b, mean_g, mean_r;
  mean_b = mean_g = mean_r = DEFAULT_MEAN;

  for (int i = 0; i < im.rows; ++i) {
    auto data = im.ptr<uchar>(i);
    for (int j = 0; j < im.cols; j++) {
      if (channels > 1) {
        *ptr_image_b++ = static_cast<float>(*data++) - mean_b;
        *ptr_image_g++ = static_cast<float>(*data++) - mean_g;
      }
    }
    *ptr_image_r++ = static_cast<float>(*data++) - mean_r;
  }
}

void prepare_input_data(const mxnet::cpp::Shape& shape, const mxnet::cpp::Context& ctx,
                        int num_threads,
                        std::vector<mxnet::cpp::NDArray>* data_arr,
                        bool random_uniform = false) {
  for (size_t i = 0; i < num_threads; ++i) {
    data_arr->emplace_back(shape, ctx, false, 0);
    int begin = i * 100;
    int end = begin + 100;
    if (random_uniform) {
      mxnet::cpp::Operator("_random_uniform")(begin, end)
          .Invoke((*data_arr)[i]);
    }
    mxnet::cpp::NDArray::WaitAll();
  }
}

// Run inference on a model
void run_inference(const std::string& model_name, const std::vector<mxnet::cpp::NDArray>& input_arrs,
                   std::vector<mxnet::NDArray*> *output_mx_arr,
                   int num_inf_per_thread = 1, bool random_sleep = false,
                   int num_threads = 1, bool static_alloc = false,
                   bool static_shape = false,
                   bool is_gpu = false) {
  LOG(INFO) << "Running inference for " + model_name +
               " num_threads: " + std::to_string(num_threads) +
               " num_inf_per_thread: " + std::to_string(num_inf_per_thread) +
               " random_sleep: " + std::to_string(random_sleep) +
               " static_alloc: " + std::to_string(static_alloc) +
               " static_shape: " + std::to_string(static_shape);
  std::string json_file = model_name + "-symbol.json";
  std::string param_file = model_name + "-0000.params";
  auto out = mxnet::cpp::Symbol::Load(json_file);
  std::string static_alloc_str = static_alloc ? "true" : "false";
  std::string static_shape_str = static_shape ? "true" : "false";

  // Prepare context
# if MXNET_USE_CUDA == 1
  mxnet::Context backend_ctx;
  mxnet::cpp::Context ctx = mxnet::cpp::Context::cpu(0);
  if (is_gpu) {
    backend_ctx = mxnet::Context::GPU(0);
    ctx = mxnet::cpp::Context::gpu(0);
  } else {
    backend_ctx = mxnet::Context::CPU(0);
    ctx = mxnet::cpp::Context::cpu(0);
  }
# else
  mxnet::Context backend_ctx = mxnet::Context::CPU(0);
  mxnet::cpp::Context ctx = mxnet::cpp::Context::cpu(0);
#endif

  // Prepare input data and parameters
  std::vector<mxnet::cpp::NDArray> data_arr(num_threads);
  std::vector<mxnet::cpp::NDArray> softmax_arr;
  std::vector<mxnet::cpp::NDArray> params;
  mxnet::cpp::Shape data_shape = mxnet::cpp::Shape(1, 3, 224, 224);
  mxnet::cpp::Shape softmax_shape = mxnet::cpp::Shape(1);
  int num_inputs = out.ListInputs().size();

  for (size_t i = 0; i < data_arr.size(); ++i) {
    data_arr[i] = input_arrs[i].Copy(ctx);
  }
  prepare_input_data(softmax_shape, ctx, num_threads, &softmax_arr);
  std::map<std::string, mxnet::cpp::NDArray> parameters;
  mxnet::cpp::NDArray::Load(param_file, 0, &parameters);

  for (const std::string& name : out.ListInputs()) {
    if (name == "arg:data") {
      continue;
    }
    if (parameters.find("arg:" + name) != parameters.end()) {
      params.push_back(parameters["arg:" + name].Copy(ctx));
    } else if (parameters.find("aux:" + name) != parameters.end()) {
      params.push_back(parameters["aux:" + name].Copy(ctx));
    }
  }

  CachedOpHandle hdl = CachedOpHandle();

  std::vector<std::string> flag_keys{"data_indices", "param_indices",
                                     "static_alloc", "static_shape"};
  std::string param_indices = "[";
  for (size_t i = 1; i < num_inputs; ++i) {
    param_indices += std::to_string(i);
    param_indices += std::string(", ");
  }
  param_indices += "]";
  std::vector<std::string> flag_vals{"[0]", param_indices, static_alloc_str,
                                     static_shape_str};
  std::vector<const char*> flag_key_cstrs, flag_val_cstrs;
  flag_key_cstrs.reserve(flag_keys.size());
  for (size_t i = 0; i < flag_keys.size(); ++i) {
    flag_key_cstrs.emplace_back(flag_keys[i].c_str());
  }
  for (size_t i = 0; i < flag_vals.size(); ++i) {
    flag_val_cstrs.emplace_back(flag_vals[i].c_str());
  }

  int ret1 = MXCreateCachedOp(out.GetHandle(), flag_keys.size(),
                              flag_key_cstrs.data(), flag_val_cstrs.data(),
                              &hdl, true);
  if (ret1 < 0) {
    LOG(FATAL) << MXGetLastError();
  }

  // Prepare data structures and lambda to run in different threads
  std::vector<NDArrayHandle *> cached_op_handles(num_threads);

  std::vector<std::vector<NDArrayHandle>> arr_handles(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    arr_handles[i].reserve(num_inputs);
    arr_handles[i].emplace_back(data_arr[i].GetHandle());
    for (size_t j = 1; j < num_inputs - 1; ++j) {
      arr_handles[i].emplace_back(params[j - 1].GetHandle());
    }
    arr_handles[i].emplace_back(softmax_arr[i].GetHandle());
  }

  auto func = [&](int num) {
    unsigned next = num;
    if (random_sleep) {
      static thread_local std::mt19937 generator;
      std::uniform_int_distribution<int> distribution(0, 5);
      int sleep_time = distribution(generator);
      std::this_thread::sleep_for(std::chrono::seconds(sleep_time));
    }
    int num_output = 0;
    const int *stypes;
    int ret = MXInvokeCachedOp(hdl, arr_handles[num].size(), arr_handles[num].data(),
                               cpu::kDevMask, 0, &num_output, &(cached_op_handles[num]), &stypes);
    if (ret < 0) {
      LOG(FATAL) << MXGetLastError();
    }
    (*output_mx_arr)[num] = static_cast<mxnet::NDArray *>(*cached_op_handles[num]);
  };

  // Spawn multiple threads, join and wait for threads to complete
  std::vector<std::thread> worker_threads(num_threads);
  int count = 0;
  for (auto &&i : worker_threads) {
    i = std::thread(func, count);
    count++;
  }

  for (auto &&i : worker_threads) {
    i.join();
  }

  mxnet::cpp::NDArray::WaitAll();

  std::string synset_file = "synset.txt";
  auto synset = LoadSynset(synset_file);
  std::vector<mxnet::NDArray> tmp(num_threads);
  for (size_t i = 0; i < num_threads; i++) {
    tmp[i] = (*output_mx_arr)[i]->Copy(mxnet::Context::CPU(0));
    tmp[i].WaitToRead();
    (*output_mx_arr)[i] = &tmp[i];
  }
  for (size_t i = 0; i < num_threads; ++i) {
    PrintOutputResult(static_cast<float *>((*output_mx_arr)[i]->data().dptr_),
                      (*output_mx_arr)[i]->shape().Size(), synset);
  }
  int ret2 = MXFreeCachedOp(hdl);
  if (ret2 < 0) {
    LOG(FATAL) << MXGetLastError();
  }

  mxnet::cpp::NDArray::WaitAll();

}

int main(int argc, char *argv[]) {
  if (argc < 5) {
    std::cout << "Please provide a model name, is_gpu, test_image" << std::endl
              << "Usage: ./multi_threaded_inference [model_name] [is_gpu] [file_names]"
              << std::endl
              << "Example: ./.multi_threaded_inference imagenet1k-inception-bn 1 0 apple.jpg"
              << std::endl
              << "NOTE: Thread number ordering will be based on the ordering of file inputs" << std::endl
              << "NOTE: Epoch is assumed to be 0" << std::endl;
    return EXIT_FAILURE;
  }
  std::string model_name = std::string(argv[1]);
  //int num_threads = std::atoi(argv[2]);
  bool is_gpu = std::atoi(argv[2]);
  CHECK(argc >= 4) << "Number of files provided should be atleast 1";
  //CHECK(num_threads == argc - 3) << "Number of files provided, should be same as num_threads";
  int num_threads = argc - 3;
  std::vector<std::string> test_files;
  for (size_t i = 0; i < argc - 3; ++i) {
    test_files.emplace_back(argv[3 + i]);
  }
  int epoch = 0;
  bool static_alloc = true;
  bool static_shape = true;


  // Image size and channels
  size_t width = 224;
  size_t height = 224;
  size_t channels = 3;

  size_t image_size = width * height * channels;

  // Read Image Data
  // load into an input arr
  std::vector<std::vector<float>> files(num_threads);
  std::vector<mxnet::cpp::NDArray> input_arrs;
  mxnet::cpp::Shape input_shape = mxnet::cpp::Shape(1, 3, 224, 224);
  for (size_t i = 0; i < files.size(); i++) {
    files[i].resize(image_size);
    GetImageFile(test_files[i], files[i].data(), channels,
                 cv::Size(width, height));
    input_arrs.emplace_back(mxnet::cpp::NDArray(files[i].data(), input_shape, mxnet::cpp::Context::cpu(0)));
  }

  // load symbol
  std::string static_alloc_str = static_alloc ? "true" : "false";
  std::string static_shape_str = static_shape ? "true" : "false";
  std::vector<mxnet::NDArray*> output_mx_arr(num_threads);
  run_inference(model_name, input_arrs, &output_mx_arr, 1, false, num_threads,
                static_alloc, static_shape, is_gpu);
  mxnet::cpp::NDArray::WaitAll();

  return 0;
}
