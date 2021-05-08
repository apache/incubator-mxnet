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

/*
 * This example demonstrates image classification workflow with pre-trained models using MXNet C++ API.
 * The example performs following tasks.
 * 1. Load the pre-trained model.
 * 2. Load the parameters of pre-trained model.
 * 3. Load the inference dataset and create a new ImageRecordIter.
 * 4. Run the forward pass and obtain throughput & accuracy.
 */
#ifndef _WIN32
#include <sys/time.h>
#endif
#include <fstream>
#include <iostream>
#include <map>
#include <chrono>
#include <string>
#include <vector>
#include <random>
#include <type_traits>
#include <opencv2/opencv.hpp>
#include "mxnet/c_api.h"
#include "mxnet/tuple.h"
#include "mxnet-cpp/MxNetCpp.h"
#include "mxnet-cpp/initializer.h"

using namespace mxnet::cpp;

double ms_now() {
  double ret;
#ifdef _WIN32
  auto timePoint = std::chrono::high_resolution_clock::now().time_since_epoch();
  ret = std::chrono::duration<double, std::milli>(timePoint).count();
#else
  struct timeval time;
  gettimeofday(&time, nullptr);
  ret = 1e+3 * time.tv_sec + 1e-3 * time.tv_usec;
#endif
  return ret;
}


// define the data type for NDArray, aliged with the definition in mshadow/base.h
enum TypeFlag {
  kFloat32 = 0,
  kFloat64 = 1,
  kFloat16 = 2,
  kUint8 = 3,
  kInt32 = 4,
  kInt8  = 5,
  kInt64 = 6,
};

/*
 * class Predictor
 *
 * This class encapsulates the functionality to load the model, prepare dataset and run the forward pass.
 */

class Predictor {
 public:
    Predictor() {}
    Predictor(const std::string& model_json_file,
              const std::string& model_params_file,
              const Shape& input_shape,
              bool use_gpu,
              bool enable_tensorrt,
              const std::string& dataset,
              const int data_nthreads,
              const std::string& data_layer_type,
              const std::vector<float>& rgb_mean,
              const std::vector<float>& rgb_std,
              int shuffle_chunk_seed,
              int seed, bool benchmark);
    void BenchmarkScore(int num_inference_batches);
    void Score(int num_skipped_batches, int num_inference_batches);
    ~Predictor();

 private:
    bool CreateImageRecordIter();
    bool AdvanceDataIter(int skipped_batches);
    void LoadModel(const std::string& model_json_file);
    void LoadParameters(const std::string& model_parameters_file);
    void SplitParamMap(const std::map<std::string, NDArray> &paramMap,
        std::map<std::string, NDArray> *argParamInTargetContext,
        std::map<std::string, NDArray> *auxParamInTargetContext,
        Context targetContext);
    void ConvertParamMapToTargetContext(const std::map<std::string, NDArray> &paramMap,
        std::map<std::string, NDArray> *paramMapInTargetContext,
        Context targetContext);
    void InitParameters();

    inline bool FileExists(const std::string &name) {
      std::ifstream fhandle(name.c_str());
      return fhandle.good();
    }
    int GetDataLayerType();

    std::map<std::string, NDArray> args_map_;
    std::map<std::string, NDArray> aux_map_;
    Symbol net_;
    Executor *executor_;
    Shape input_shape_;
    Context global_ctx_ = Context::cpu();

    MXDataIter *val_iter_;
    bool use_gpu_;
    bool enable_tensorrt_;
    std::string dataset_;
    int data_nthreads_;
    std::string data_layer_type_;
    std::vector<float> rgb_mean_;
    std::vector<float> rgb_std_;
    int shuffle_chunk_seed_;
    int seed_;
    bool benchmark_;
};


/*
 * The constructor takes following parameters as input:
 * 1. model_json_file:  The model in json formatted file.
 * 2. model_params_file: File containing model parameters
 * 3. input_shape: Shape of input data to the model. Since this class will be running one inference at a time,
 *                 the input shape is required to be in format Shape(1, number_of_channels, height, width)
 *                 The input image will be resized to (height x width) size before running the inference.
 * 4. use_gpu: determine if run inference on GPU
 * 5. enable_tensorrt: determine if enable TensorRT
 * 6. dataset: data file (.rec) to be used for inference
 * 7. data_nthreads: number of threads for data loading
 * 8. data_layer_type: data type for data layer
 * 9. rgb_mean: mean value to be subtracted on R/G/B channel
 * 10. rgb_std: standard deviation on R/G/B channel
 * 11. shuffle_chunk_seed: shuffling chunk seed
 * 12. seed: shuffling seed
 * 13. benchmark: use dummy data for inference
 *
 * The constructor will:
 *  1. Create ImageRecordIter based on the given dataset file.
 *  2. Load the model and parameter files.
 *  3. Infer and construct NDArrays according to the input argument and create an executor.
 */
Predictor::Predictor(const std::string& model_json_file,
                     const std::string& model_params_file,
                     const Shape& input_shape,
                     bool use_gpu,
                     bool enable_tensorrt,
                     const std::string& dataset,
                     const int data_nthreads,
                     const std::string& data_layer_type,
                     const std::vector<float>& rgb_mean,
                     const std::vector<float>& rgb_std,
                     int shuffle_chunk_seed,
                     int seed, bool benchmark)
    : input_shape_(input_shape),
      use_gpu_(use_gpu),
      enable_tensorrt_(enable_tensorrt),
      dataset_(dataset),
      data_nthreads_(data_nthreads),
      data_layer_type_(data_layer_type),
      rgb_mean_(rgb_mean),
      rgb_std_(rgb_std),
      shuffle_chunk_seed_(shuffle_chunk_seed),
      seed_(seed),
      benchmark_(benchmark) {
  if (use_gpu) {
    global_ctx_ = Context::gpu();
  }

  // initilize data iterator
  if (!benchmark_ && !CreateImageRecordIter()) {
    LG << "Error: failed to create ImageRecordIter";
    throw std::runtime_error("ImageRecordIter cannot be created");
  }

  // Load the model
  LoadModel(model_json_file);
  // Initilize the parameters
  // benchmark=true && model_params_file.empty(), randomly initialize parameters
  // else, load parameters
  if (benchmark_ && model_params_file.empty()) {
    InitParameters();
  } else {
    LoadParameters(model_params_file);
  }

  int dtype = GetDataLayerType();
  if (dtype == -1) {
    throw std::runtime_error("Unsupported data layer type...");
  }
  args_map_["data"] = NDArray(input_shape_, global_ctx_, false, dtype);
  Shape label_shape(input_shape_[0]);
  args_map_["softmax_label"] = NDArray(label_shape, global_ctx_, false);
  std::vector<NDArray> arg_arrays;
  std::vector<NDArray> grad_arrays;
  std::vector<OpReqType> grad_reqs;
  std::vector<NDArray> aux_arrays;

  // infer and create ndarrays according to the given input ndarrays.
  net_.InferExecutorArrays(global_ctx_, &arg_arrays, &grad_arrays, &grad_reqs,
                           &aux_arrays, args_map_, std::map<std::string, NDArray>(),
                           std::map<std::string, OpReqType>(), aux_map_);
  for (auto& i : grad_reqs) i = OpReqType::kNullOp;

  // Create an executor after binding the model to input parameters.
  executor_ = new Executor(net_, global_ctx_, arg_arrays, grad_arrays, grad_reqs, aux_arrays);
}

/*
 * The following function is used to get the data layer type for input data
 */
int Predictor::GetDataLayerType() {
  int ret_type = -1;
  if (data_layer_type_ == "float32") {
    ret_type = kFloat32;
  } else if (data_layer_type_ == "int8") {
    ret_type = kInt8;
  } else if (data_layer_type_ == "uint8") {
    ret_type = kUint8;
  } else {
    LG << "Unsupported data layer type " << data_layer_type_ << "..."
       << "Please use one of {float32, int8, uint8}";
  }
  return ret_type;
}

/*
 * create a new ImageRecordIter according to the given parameters
 */
bool Predictor::CreateImageRecordIter() {
  val_iter_ = new MXDataIter("ImageRecordIter");
  if (!FileExists(dataset_)) {
    LG << "Error: " << dataset_ << " must be provided";
    return false;
  }

  std::vector<index_t> shape_vec;
  for (index_t i = 1; i < input_shape_.ndim(); i++)
    shape_vec.push_back(input_shape_[i]);
  mxnet::TShape data_shape(shape_vec.begin(), shape_vec.end());

  // set image record parser parameters
  val_iter_->SetParam("path_imgrec", dataset_);
  val_iter_->SetParam("label_width", 1);
  val_iter_->SetParam("data_shape", data_shape);
  val_iter_->SetParam("preprocess_threads", data_nthreads_);
  val_iter_->SetParam("shuffle_chunk_seed", shuffle_chunk_seed_);

  // set Batch parameters
  val_iter_->SetParam("batch_size", input_shape_[0]);

  // image record parameters
  val_iter_->SetParam("shuffle", true);
  val_iter_->SetParam("seed", seed_);

  // set normalize parameters
  val_iter_->SetParam("mean_r", rgb_mean_[0]);
  val_iter_->SetParam("mean_g", rgb_mean_[1]);
  val_iter_->SetParam("mean_b", rgb_mean_[2]);
  val_iter_->SetParam("std_r", rgb_std_[0]);
  val_iter_->SetParam("std_g", rgb_std_[1]);
  val_iter_->SetParam("std_b", rgb_std_[2]);

  // set prefetcher parameters
  if (use_gpu_) {
    val_iter_->SetParam("ctx", "gpu");
  } else {
    val_iter_->SetParam("ctx", "cpu");
  }
  val_iter_->SetParam("dtype", data_layer_type_);

  val_iter_->CreateDataIter();
  return true;
}

/*
 * The following function loads the model from json file.
 */
void Predictor::LoadModel(const std::string& model_json_file) {
  if (!FileExists(model_json_file)) {
    LG << "Model file " << model_json_file << " does not exist";
    throw std::runtime_error("Model file does not exist");
  }
  LG << "Loading the model from " << model_json_file << std::endl;
  net_ = Symbol::Load(model_json_file);
  if (enable_tensorrt_) {
    net_ = net_.GetBackendSymbol("TensorRT");
  }
}

/*
 * The following function loads the model parameters.
 */
void Predictor::LoadParameters(const std::string& model_parameters_file) {
  if (!FileExists(model_parameters_file)) {
    LG << "Parameter file " << model_parameters_file << " does not exist";
    throw std::runtime_error("Model parameters does not exist");
  }
  LG << "Loading the model parameters from " << model_parameters_file << std::endl;
  std::map<std::string, NDArray> parameters;
  NDArray::Load(model_parameters_file, 0, &parameters);
  if (enable_tensorrt_) {
    std::map<std::string, NDArray> intermediate_args_map;
    std::map<std::string, NDArray> intermediate_aux_map;
    SplitParamMap(parameters, &intermediate_args_map, &intermediate_aux_map, Context::cpu());
    contrib::InitTensorRTParams(net_, &intermediate_args_map, &intermediate_aux_map);
    ConvertParamMapToTargetContext(intermediate_args_map, &args_map_, global_ctx_);
    ConvertParamMapToTargetContext(intermediate_aux_map, &aux_map_, global_ctx_);
  } else {
    SplitParamMap(parameters, &args_map_, &aux_map_, global_ctx_);
  }
  /*WaitAll is need when we copy data between GPU and the main memory*/
  NDArray::WaitAll();
}

/*
 * The following function split loaded param map into arg parm
 *   and aux param with target context
 */
void Predictor::SplitParamMap(const std::map<std::string, NDArray> &paramMap,
    std::map<std::string, NDArray> *argParamInTargetContext,
    std::map<std::string, NDArray> *auxParamInTargetContext,
    Context targetContext) {
  for (const auto& pair : paramMap) {
    std::string type = pair.first.substr(0, 4);
    std::string name = pair.first.substr(4);
    if (type == "arg:") {
      (*argParamInTargetContext)[name] = pair.second.Copy(targetContext);
    } else if (type == "aux:") {
      (*auxParamInTargetContext)[name] = pair.second.Copy(targetContext);
    }
  }
}

/*
 * The following function copy the param map into the target context
 */
void Predictor::ConvertParamMapToTargetContext(const std::map<std::string, NDArray> &paramMap,
    std::map<std::string, NDArray> *paramMapInTargetContext,
    Context targetContext) {
  for (const auto& pair : paramMap) {
    (*paramMapInTargetContext)[pair.first] = pair.second.Copy(targetContext);
  }
}

/*
 * The following function randomly initializes the parameters when benchmark_ is true.
 */
void Predictor::InitParameters() {
  std::vector<mx_uint> data_shape;
  for (index_t i = 0; i < input_shape_.ndim(); i++) {
    data_shape.push_back(input_shape_[i]);
  }

  std::map<std::string, std::vector<mx_uint> > arg_shapes;
  std::vector<std::vector<mx_uint> > aux_shapes, in_shapes, out_shapes;
  arg_shapes["data"] = data_shape;
  net_.InferShape(arg_shapes, &in_shapes, &aux_shapes, &out_shapes);

  // initializer to call
  Xavier xavier(Xavier::uniform, Xavier::avg, 2.0f);

  auto arg_name_list = net_.ListArguments();
  for (index_t i = 0; i < in_shapes.size(); i++) {
    const auto &shape = in_shapes[i];
    const auto &arg_name = arg_name_list[i];
    int paramType = kFloat32;
    if (Initializer::StringEndWith(arg_name, "weight_quantize") ||
        Initializer::StringEndWith(arg_name, "bias_quantize")) {
      paramType = kInt8;
    }
    NDArray tmp_arr(shape, global_ctx_, false, paramType);
    xavier(arg_name, &tmp_arr);
    args_map_[arg_name] = tmp_arr.Copy(global_ctx_);
  }

  auto aux_name_list = net_.ListAuxiliaryStates();
  for (index_t i = 0; i < aux_shapes.size(); i++) {
    const auto &shape = aux_shapes[i];
    const auto &aux_name = aux_name_list[i];
    NDArray tmp_arr(shape, global_ctx_, false);
    xavier(aux_name, &tmp_arr);
    aux_map_[aux_name] = tmp_arr.Copy(global_ctx_);
  }
  /*WaitAll is need when we copy data between GPU and the main memory*/
  NDArray::WaitAll();
}

/*
 * The following function runs the forward pass on the model
 * and use dummy data for benchmark.
 */
void Predictor::BenchmarkScore(int num_inference_batches) {
  // Create dummy data
  std::vector<float> dummy_data(input_shape_.Size());
  std::default_random_engine generator;
  std::uniform_real_distribution<float> val(0.0f, 1.0f);
  for (size_t i = 0; i < static_cast<size_t>(input_shape_.Size()); ++i) {
    dummy_data[i] = static_cast<float>(val(generator));
  }
  executor_->arg_dict()["data"].SyncCopyFromCPU(
        dummy_data.data(),
        input_shape_.Size());
  NDArray::WaitAll();

  LG << "Running the forward pass on model to evaluate the performance..";

  // warm up.
  for (int i = 0; i < 5; i++) {
    executor_->Forward(false);
    NDArray::WaitAll();
  }

  // Run the forward pass.
  double ms = ms_now();
  for (int i = 0; i < num_inference_batches; i++) {
    executor_->Forward(false);
    NDArray::WaitAll();
  }
  ms = ms_now() - ms;
  LG << " benchmark completed!";
  LG << " batch size: " << input_shape_[0] << " num batch: " << num_inference_batches
     << " throughput: " << 1000.0 * input_shape_[0] * num_inference_batches / ms
     << " imgs/s latency:" << ms / input_shape_[0] / num_inference_batches << " ms";
}

/*
 * \param skipped_batches skip the first number of batches
 *
 */
bool Predictor::AdvanceDataIter(int skipped_batches) {
  assert(skipped_batches >= 0);
  if (skipped_batches == 0) return true;
  int skipped_count = 0;
  while (val_iter_->Next()) {
    if (++skipped_count >= skipped_batches) break;
  }
  if (skipped_count != skipped_batches) return false;
  return true;
}

/*
 * The following function runs the forward pass on the model
 * and use real data for testing accuracy and performance.
 */
void Predictor::Score(int num_skipped_batches, int num_inference_batches) {
  // Create metrics
  Accuracy val_acc;

  val_iter_->Reset();
  val_acc.Reset();
  int nBatch = 0;

  if (!AdvanceDataIter(num_skipped_batches)) {
    LG << "skipped batches should less than total batches!";
    return;
  }

  double ms = ms_now();
  while (val_iter_->Next()) {
    auto data_batch = val_iter_->GetDataBatch();
    data_batch.data.CopyTo(&args_map_["data"]);
    data_batch.label.CopyTo(&args_map_["softmax_label"]);
    NDArray::WaitAll();

    // running on forward pass
    executor_->Forward(false);
    NDArray::WaitAll();
    val_acc.Update(data_batch.label, executor_->outputs[0]);

    if (++nBatch >= num_inference_batches) {
      break;
    }
  }
  ms = ms_now() - ms;
  auto args_name = net_.ListArguments();
  LG << "INFO:" << "Dataset for inference: " << dataset_;
  LG << "INFO:" << "label_name = " << args_name[args_name.size()-1];
  LG << "INFO:" << "rgb_mean: " << "(" << rgb_mean_[0] << ", " << rgb_mean_[1]
     << ", " << rgb_mean_[2] << ")";
  LG << "INFO:" << "rgb_std: " << "(" << rgb_std_[0] << ", " << rgb_std_[1]
     << ", " << rgb_std_[2] << ")";
  LG << "INFO:" << "Image shape: " << "(" << input_shape_[1] << ", "
     << input_shape_[2] << ", " << input_shape_[3] << ")";
  LG << "INFO:" << "Finished inference with: " << nBatch * input_shape_[0]
     << " images ";
  LG << "INFO:" << "Batch size = " << input_shape_[0] << " for inference";
  LG << "INFO:" << "Accuracy: " << val_acc.Get();
  LG << "INFO:" << "Throughput: " << (1000.0 * nBatch * input_shape_[0] / ms)
     << " images per second";
}

Predictor::~Predictor() {
  if (executor_) {
    delete executor_;
  }
  if (!benchmark_ && val_iter_) {
    delete val_iter_;
  }
  MXNotifyShutdown();
}

/*
 * Convert the input string of number into the vector.
 */
template<typename T>
std::vector<T> createVectorFromString(const std::string& input_string) {
  std::vector<T> dst_vec;
  char *p_next;
  T elem;
  bool bFloat = std::is_same<T, float>::value;
  if (!bFloat) {
    elem = strtol(input_string.c_str(), &p_next, 10);
  } else {
    elem = strtof(input_string.c_str(), &p_next);
  }

  dst_vec.push_back(elem);
  while (*p_next) {
    if (!bFloat) {
      elem = strtol(p_next, &p_next, 10);
    } else {
      elem = strtof(p_next, &p_next);
    }
    dst_vec.push_back(elem);
  }
  return dst_vec;
}

void printUsage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "imagenet_inference --symbol_file <model symbol file in json format>" << std::endl
              << "--params_file <model params file> " << std::endl
              << "--dataset <dataset used to run inference> " << std::endl
              << "--data_nthreads <default: 60> " << std::endl
              << "--input_shape <shape of input image e.g \"3 224 224\">] " << std::endl
              << "--rgb_mean <mean value to be subtracted on RGB channel e.g \"0 0 0\">"
              << std::endl
              << "--rgb_std <standard deviation on R/G/B channel. e.g \"1 1 1\"> " << std::endl
              << "--batch_size <number of images per batch> " << std::endl
              << "--num_skipped_batches <skip the number of batches for inference> " << std::endl
              << "--num_inference_batches <number of batches used for inference> " << std::endl
              << "--data_layer_type <default: \"float32\" "
              << "choices: [\"float32\",\"int8\",\"uint8\"]>" << std::endl
              << "--gpu  <whether to run inference on GPU, default: false>" << std::endl
              << "--enableTRT  <whether to run inference with TensorRT, "
              << "default: false>" << std::endl
              << "--benchmark <whether to use dummy data to run inference, default: false>"
              << std::endl;
}

int main(int argc, char** argv) {
  std::string model_file_json;
  std::string model_file_params;
  std::string dataset("");
  std::string input_rgb_mean("0 0 0");
  std::string input_rgb_std("1 1 1");
  bool use_gpu = false;
  bool enable_tensorrt = false;
  bool benchmark = false;
  int batch_size = 64;
  int num_skipped_batches = 0;
  int num_inference_batches = 100;
  std::string data_layer_type("float32");
  std::string input_shape("3 224 224");
  int seed = 48564309;
  int shuffle_chunk_seed = 3982304;
  int data_nthreads = 60;

  int index = 1;
  while (index < argc) {
    if (strcmp("--symbol_file", argv[index]) == 0) {
      index++;
      model_file_json = (index < argc ? argv[index]:"");
    } else if (strcmp("--params_file", argv[index]) == 0) {
      index++;
      model_file_params = (index < argc ? argv[index]:"");
    } else if (strcmp("--dataset", argv[index]) == 0) {
      index++;
      dataset = (index < argc ? argv[index]:dataset);
    } else if (strcmp("--data_nthreads", argv[index]) == 0) {
      index++;
      data_nthreads = strtol(argv[index], nullptr, 10);
    } else if (strcmp("--input_shape", argv[index]) == 0) {
      index++;
      input_shape = (index < argc ? argv[index]:input_shape);
    } else if (strcmp("--rgb_mean", argv[index]) == 0) {
      index++;
      input_rgb_mean = (index < argc ? argv[index]:input_rgb_mean);
    } else if (strcmp("--rgb_std", argv[index]) == 0) {
      index++;
      input_rgb_std = (index < argc ? argv[index]:input_rgb_std);
    } else if (strcmp("--batch_size", argv[index]) == 0) {
      index++;
      batch_size = strtol(argv[index], nullptr, 10);
    }  else if (strcmp("--num_skipped_batches", argv[index]) == 0) {
      index++;
      num_skipped_batches = strtol(argv[index], nullptr, 10);
    }  else if (strcmp("--num_inference_batches", argv[index]) == 0) {
      index++;
      num_inference_batches = strtol(argv[index], nullptr, 10);
    } else if (strcmp("--data_layer_type", argv[index]) == 0) {
      index++;
      data_layer_type = (index < argc ? argv[index]:data_layer_type);
    } else if (strcmp("--gpu", argv[index]) == 0) {
      use_gpu = true;
    } else if (strcmp("--enableTRT", argv[index]) == 0) {
      use_gpu = true;
      enable_tensorrt = true;
    } else if (strcmp("--benchmark", argv[index]) == 0) {
      benchmark = true;
    } else if (strcmp("--help", argv[index]) == 0) {
      printUsage();
      return 0;
    }
    index++;
  }

  if (model_file_json.empty()
      || (!benchmark && model_file_params.empty())
      || (enable_tensorrt && model_file_params.empty())) {
    LG << "ERROR: Model details such as symbol, param files are not specified";
    printUsage();
    return 1;
  }
  std::vector<index_t> input_dimensions = createVectorFromString<index_t>(input_shape);
  input_dimensions.insert(input_dimensions.begin(), batch_size);
  Shape input_data_shape(input_dimensions);

  std::vector<float> rgb_mean = createVectorFromString<float>(input_rgb_mean);
  std::vector<float> rgb_std = createVectorFromString<float>(input_rgb_std);

  // Initialize the predictor object
  Predictor predict(model_file_json, model_file_params, input_data_shape, use_gpu, enable_tensorrt,
                    dataset, data_nthreads, data_layer_type, rgb_mean, rgb_std, shuffle_chunk_seed,
                    seed, benchmark);

  if (benchmark) {
    predict.BenchmarkScore(num_inference_batches);
  } else {
    predict.Score(num_skipped_batches, num_inference_batches);
  }
  return 0;
}
