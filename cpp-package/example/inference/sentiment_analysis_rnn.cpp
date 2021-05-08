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
 * This example demonstrates sentiment prediction workflow with pre-trained RNN model using MXNet C++ API.
 * The example performs following tasks.
 * 1. Load the pre-trained RNN model,
 * 2. Load the dictionary file that contains word to index mapping.
 * 3. Create executors for pre-determined input lengths.
 * 4. Convert each line in the input to the vector of indices.
 * 5. Predictor finds the right executor for each line.
 * 4. Run the forward pass for each line and predicts the sentiment scores.
 * The example uses a pre-trained RNN model that is trained with the IMDB dataset.
 */

#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <map>
#include <string>
#include <algorithm>
#include <vector>
#include <sstream>
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

static const int DEFAULT_BUCKET_KEYS[] = {30, 25, 20, 15, 10, 5};
static const char DEFAULT_S3_URL[] = "https://s3.amazonaws.com/mxnet-cpp/RNN_model/";


/*
 * class Predictor
 *
 * This class encapsulates the functionality to load the model, process input image and run the forward pass.
 */

class Predictor {
 public:
    Predictor() {}
    Predictor(const std::string& model_json,
              const std::string& model_params,
              const std::string& input_dictionary,
              const std::vector<int>& bucket_keys,
              bool use_gpu = false);
    float PredictSentiment(const std::string &input_review);
    ~Predictor();

 private:
    void LoadModel(const std::string& model_json_file);
    void LoadParameters(const std::string& model_parameters_file);
    void LoadDictionary(const std::string &input_dictionary);
    inline bool FileExists(const std::string& name) {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
    }
    float PredictSentimentForOneLine(const std::string &input_line);
    int ConvertToIndexVector(const std::string& input,
                      std::vector<float> *input_vector);
    int GetIndexForOutputSymbolName(const std::string& output_symbol_name);
    float GetIndexForWord(const std::string& word);
    int GetClosestBucketKey(int num_words);

    std::map<std::string, NDArray> args_map;
    std::map<std::string, NDArray> aux_map;
    std::map<std::string, int>  wordToIndex;
    Symbol net;
    std::map<int, Executor*> executor_buckets;
    Context global_ctx = Context::cpu();
    int highest_bucket_key;
};


/*
 * The constructor takes the following parameters as input:
 * 1. model_json:  The RNN model in json formatted file.
 * 2. model_params: File containing model parameters
 * 3. input_dictionary: File containing the word and associated index.
 * 4. bucket_keys: A vector of bucket keys for creating executors.
 *
 * The constructor:
 *  1. Loads the model and parameter files.
 *  2. Loads the dictionary file to create index to word and word to index maps.
 *  3. For each bucket key in the input vector of bucket keys, it creates an executor.
 *     The executors share the memory. The bucket key determines the length of input data
 *     required for that executor.
 *  4. Creates a map of bucket key to corresponding executor.
 *  5. The model is loaded only once. The executors share the memory for the parameters.
 */
Predictor::Predictor(const std::string& model_json,
                     const std::string& model_params,
                     const std::string& input_dictionary,
                     const std::vector<int>& bucket_keys,
                     bool use_gpu) {
  if (use_gpu) {
    global_ctx = Context::gpu();
  }

  /*
   * Load the dictionary file that contains the word and its index.
   * The function creates word to index and index to word map. The maps are used to create index
   * vector for the input sentence.
   */
  LoadDictionary(input_dictionary);

  // Load the model
  LoadModel(model_json);

  // Load the model parameters.
  LoadParameters(model_params);

  /*
   * Create the executors for each bucket key. The bucket key represents the shape of input data.
   * The executors will share the memory by using following technique:
   * 1. Infer the executor arrays and bind the first executor with the first bucket key.
   * 2. Then for creating the next bucket key, adjust the shape of input argument to match that key.
   * 3. Create the executor for the next bucket key by passing the inferred executor arrays and
   *    pointer to the executor created for the first key.
   */
  std::vector<NDArray> arg_arrays;
  std::vector<NDArray> grad_arrays;
  std::vector<OpReqType> grad_reqs;
  std::vector<NDArray> aux_arrays;

  /*
   * Create master executor with highest bucket key for optimizing the shared memory between the
   * executors for the remaining bucket keys.
   */
  highest_bucket_key = *(std::max_element(bucket_keys.begin(), bucket_keys.end()));
  args_map["data0"] = NDArray(Shape(highest_bucket_key, 1), global_ctx, false);
  args_map["data1"] = NDArray(Shape(1), global_ctx, false);

  net.InferExecutorArrays(global_ctx, &arg_arrays, &grad_arrays, &grad_reqs,
                          &aux_arrays, args_map, std::map<std::string, NDArray>(),
                              std::map<std::string, OpReqType>(), aux_map);
  Executor *master_executor = net.Bind(global_ctx, arg_arrays, grad_arrays, grad_reqs, aux_arrays,
                                 std::map<std::string, Context>(), nullptr);
  executor_buckets[highest_bucket_key] = master_executor;

  for (int bucket : bucket_keys) {
    if (executor_buckets.find(bucket) == executor_buckets.end()) {
      arg_arrays[0]  = NDArray(Shape(bucket, 1), global_ctx, false);
      Executor *executor = net.Bind(global_ctx, arg_arrays, grad_arrays, grad_reqs, aux_arrays,
                                    std::map<std::string, Context>(), master_executor);
      executor_buckets[bucket] = executor;
    }
  }
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
  net = Symbol::Load(model_json_file);
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
  for (const auto &k : parameters) {
    if (k.first.substr(0, 4) == "aux:") {
      auto name = k.first.substr(4, k.first.size() - 4);
      aux_map[name] = k.second.Copy(global_ctx);
    }
    if (k.first.substr(0, 4) == "arg:") {
      auto name = k.first.substr(4, k.first.size() - 4);
      args_map[name] = k.second.Copy(global_ctx);
    }
  }
  /*WaitAll is need when we copy data between GPU and the main memory*/
  NDArray::WaitAll();
}


/*
 * The following function loads the dictionary file.
 * The function constructs the word to index and index to word maps.
 * These maps will be used to represent words in the input sentence to their indices.
 * Ensure to use the same dictionary file that was used for training the network.
 */
void Predictor::LoadDictionary(const std::string& input_dictionary) {
  if (!FileExists(input_dictionary)) {
    LG << "Dictionary file " << input_dictionary << " does not exist";
    throw std::runtime_error("Dictionary file does not exist");
  }
  LG << "Loading the dictionary file.";
  std::ifstream fi(input_dictionary.c_str());
  if (!fi.is_open()) {
    std::cerr << "Error opening dictionary file " << input_dictionary << std::endl;
    assert(false);
  }

  std::string line;
  std::string word;
  int index;
  while (std::getline(fi, line)) {
    std::istringstream stringline(line);
    stringline >> word >> index;
    wordToIndex[word] = index;
  }
  fi.close();
}


/*
 * The function returns the index associated with the word in the dictionary.
 * If the word is not present, the index representing "<unk>" is returned.
 * If the "<unk>" is not present then 0 is returned.
 */
float Predictor::GetIndexForWord(const std::string& word) {
  if (wordToIndex.find(word) == wordToIndex.end()) {
    if (wordToIndex.find("<unk>") == wordToIndex.end())
      return 0;
    else
      return static_cast<float>(wordToIndex["<unk>"]);
  }
  return static_cast<float>(wordToIndex[word]);
}

/*
 * The function populates the input vector with indices from the dictionary that
 * correspond to the words in the input string.
 * The function returns the number of words in the input line.
 */
int Predictor::ConvertToIndexVector(const std::string& input, std::vector<float> *input_vector) {
  std::istringstream input_string(input);
  input_vector->clear();
  const char delimiter = ' ';
  std::string token;
  size_t words = 0;
  while (std::getline(input_string, token, delimiter) && (words <= input_vector->size())) {
    input_vector->push_back(GetIndexForWord(token));
    words++;
  }
  return words;
}


/*
 * The function returns the index at which the given symbol name will appear
 * in the output vector of NDArrays obtained after running the forward pass on the executor.
 */
int Predictor::GetIndexForOutputSymbolName(const std::string& output_symbol_name) {
  int index = 0;
  for (const std::string op : net.ListOutputs()) {
    if (op == output_symbol_name) {
      return index;
    } else {
      index++;
    }
  }
  throw std::runtime_error("The output symbol name can not be found");
}


/*
 * The function finds the closest bucket for the given num_words in the input line.
 * If the exact bucket key exists, function returns that bucket key.
 * If the matching bucket key does not exist, function looks for the next bucket key
 * that is greater than given num_words.
 * If the next larger bucket does not exist, function returns the largest bucket key.
 */
int Predictor::GetClosestBucketKey(int num_words) {
  int closest_bucket_key = highest_bucket_key;

  if (executor_buckets.lower_bound(num_words) != executor_buckets.end()) {
    closest_bucket_key = executor_buckets.lower_bound(num_words)->first;
  }
  return closest_bucket_key;
}


/*
 * The following function runs the forward pass on the model for the given line.
 *
 */
float Predictor::PredictSentimentForOneLine(const std::string& input_line) {
  /*
   * Initialize a vector of length equal to 'num_words' with index corresponding to <eos>.
   * Convert the input string to a vector of indices that represent
   * the words in the input string.
   */
  std::vector<float> index_vector(GetIndexForWord("<eos>"));
  int num_words = ConvertToIndexVector(input_line, &index_vector);
  int bucket_key = GetClosestBucketKey(num_words);

  /*
   * The index_vector has size equal to num_words. The vector needs to be padded if
   * the bucket_key is greater than num_words. The vector needs to be trimmed if
   * the bucket_key is smaller than num_words.
   */
  index_vector.resize(bucket_key, GetIndexForWord("<eos>"));

  Executor* executor = executor_buckets[bucket_key];
  executor->arg_dict()["data0"].SyncCopyFromCPU(index_vector.data(), index_vector.size());
  executor->arg_dict()["data1"] = num_words;

  // Run the forward pass.
  executor->Forward(false);

  /*
   * The output is available in executor->outputs. It is a vector of
   * NDArray. We need to find the index in that vector that
   * corresponds to the output symbol "sentimentnet0_hybridsequential0_dense0_fwd_output".
   */
  const std::string output_symbol_name = "sentimentnet0_hybridsequential0_dense0_fwd_output";
  int output_index = GetIndexForOutputSymbolName(output_symbol_name);
  std::vector<NDArray> outputs = executor->outputs;
  auto arrayout = executor->outputs[output_index].Copy(global_ctx);
  /*
   * We will run sigmoid operator to find out the sentiment score between
   * 0 and 1 where 1 represents positive.
   */
  NDArray ret;
  Operator("sigmoid")(arrayout).Invoke(ret);
  ret.WaitToRead();

  return ret.At(0, 0);
}


/*
 * The function predicts the sentiment score for the input review.
 * The function splits the input review in lines (separated by '.').
 * It finds sentiment score for each line and computes the average.
 */
float Predictor::PredictSentiment(const std::string& input_review) {
  std::istringstream input_string(input_review);
  int num_lines = 0;
  float sentiment_score = 0.0f;

  // Split the iput review in separate lines separated by '.'
  const char delimiter = '.';
  std::string line;
  while (std::getline(input_string, line, delimiter)) {
    // Predict the sentiment score for each line.
    float score = PredictSentimentForOneLine(line);
    LG << "Input Line : [" << line << "] Score : " << score;
    sentiment_score += score;
    num_lines++;
  }

  // Find the average sentiment score.
  sentiment_score = sentiment_score / num_lines;
  return sentiment_score;
}


/*
 * The destructor frees the executor and notifies MXNetEngine to shutdown.
 */
Predictor::~Predictor() {
  for (auto bucket : this->executor_buckets) {
    Executor* executor = bucket.second;
    delete executor;
  }
  MXNotifyShutdown();
}


/*
 * The function prints the usage information.
 */
void printUsage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "sentiment_analysis_rnn " << std::endl
              << "--input Input movie review. The review can be single line or multiline."
              << "e.g. \"This movie is the best.\" OR  "
              << "\"This movie is the best. The direction is awesome.\" " << std::endl
              << "[--gpu]  Specify this option if workflow needs to be run in gpu context "
              << std::endl
              << "If the review is multiline, the example predicts sentiment score for each line "
              << "and the final score is the average of scores obtained for each line."
              << std::endl;
}


/*
 * The function downloads the model files from s3 bucket.
 */
void DownloadFiles(const std::vector<std::string> model_files) {
  std::string wget_command("wget -nc ");
  std::string s3_url(DEFAULT_S3_URL);
  for (auto &file : model_files) {
    std::ostringstream oss;
    oss << wget_command << s3_url << file << " -O " << file;
    int status = system(oss.str().c_str());
    LG << "Downloading " << file << " with status " << status;
  }
  return;
}


int main(int argc, char** argv) {
  std::string model_file_json = "./sentiment_analysis-symbol.json";
  std::string model_file_params ="./sentiment_analysis-0010.params";
  std::string input_dictionary = "./sentiment_token_to_idx.txt";
  std::string input_review = "This movie is the best";
  bool use_gpu = false;

  int index = 1;
  while (index < argc) {
    if (strcmp("--input", argv[index]) == 0) {
      index++;
      input_review = (index < argc ? argv[index]:input_review);
    } else if (strcmp("--gpu", argv[index]) == 0) {
      use_gpu = true;
    } else if (strcmp("--help", argv[index]) == 0) {
      printUsage();
      return 0;
    }
    index++;
  }


  /*
   * Download the trained RNN model file, param file and dictionary file.
   * The dictionary file contains word to index mapping.
   * Each line of the dictionary file contains a word and the unique index for that word separated
   * by a space. For example:
   * snippets 11172
   * This dictionary file is created when the RNN model was trained with a particular dataset.
   * Hence the dictionary file is specific to the dataset with which model was trained.
   */
  std::vector<std::string> files;
  files.push_back(model_file_json);
  files.push_back(model_file_params);
  files.push_back(input_dictionary);

  DownloadFiles(files);

  std::vector<int> buckets(DEFAULT_BUCKET_KEYS,
                           DEFAULT_BUCKET_KEYS + sizeof(DEFAULT_BUCKET_KEYS) / sizeof(int));

  try {
    // Initialize the predictor object
    Predictor predict(model_file_json, model_file_params, input_dictionary, buckets, use_gpu);

    // Run the forward pass to predict the sentiment score for the given review.
    float sentiment_score = predict.PredictSentiment(input_review);
    LG << "The sentiment score between 0 and 1, (1 being positive)=" << sentiment_score;
  } catch (std::runtime_error &error) {
    LG << MXGetLastError();
    LG << "Execution failed with ERROR: " << error.what();
    return 1;
  } catch (...) {
    /*
     * If underlying MXNet code has thrown an exception the error message is
     * accessible through MXGetLastError() function.
     */
    LG << "Execution failed with following MXNet error";
    LG << MXGetLastError();
    return 1;
  }
  return 0;
}
