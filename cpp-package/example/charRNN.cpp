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
 * Hua Zhang mz24cn@hotmail.com
 * The code implements C++ version charRNN for mxnet\example\rnn\char-rnn.ipynb with MXNet.cpp API.
 * The generated params file is compatiable with python version.
 * train() and predict() has been verified with original data samples.
 * 2017/1/23:
 * Add faster version charRNN based on built-in cuDNN RNN operator, 10 times faster.
 * Add time major computation graph, although no substantial performance difference.
 * Support continuing training from last params file.
 * Rename params file epoch number starts from zero.
 */

#if _MSC_VER
#pragma warning(disable: 4996)  // VS2015 complains on 'std::copy' ...
#endif
#include <cstring>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <tuple>
#include <algorithm>
#include <functional>
#include <thread>
#include <chrono>
#include "mxnet-cpp/MxNetCpp.h"
#include "utils.h"

using namespace mxnet::cpp;

struct LSTMState {
  Symbol C;
  Symbol h;
};

struct LSTMParam {
  Symbol i2h_weight;
  Symbol i2h_bias;
  Symbol h2h_weight;
  Symbol h2h_bias;
};

bool TIME_MAJOR = true;

// LSTM Cell symbol
LSTMState LSTM(int num_hidden, const Symbol& indata, const LSTMState& prev_state,
    const LSTMParam& param, int seqidx, int layeridx, mx_float dropout = 0) {
  auto input = dropout > 0? Dropout(indata, dropout) : indata;
  auto prefix = std::string("t") + std::to_string(seqidx) + "_l" + std::to_string(layeridx);
  auto i2h = FullyConnected(prefix + "_i2h", input, param.i2h_weight, param.i2h_bias,
      num_hidden * 4);
  auto h2h = FullyConnected(prefix + "_h2h", prev_state.h, param.h2h_weight, param.h2h_bias,
      num_hidden * 4);
  auto gates = i2h + h2h;
  auto slice_gates = SliceChannel(prefix + "_slice", gates, 4);
  auto in_gate = Activation(slice_gates[0], ActivationActType::kSigmoid);
  auto in_transform = Activation(slice_gates[1], ActivationActType::kTanh);
  auto forget_gate = Activation(slice_gates[2], ActivationActType::kSigmoid);
  auto out_gate = Activation(slice_gates[3], ActivationActType::kSigmoid);

  LSTMState state;
  state.C = (forget_gate * prev_state.C) + (in_gate * in_transform);
  state.h = out_gate * Activation(state.C, ActivationActType::kTanh);
  return state;
}

Symbol LSTMUnroll(int num_lstm_layer, int sequence_length, int input_dim,
        int num_hidden, int num_embed, mx_float dropout = 0) {
  auto isTrain = sequence_length > 1;
  auto data = Symbol::Variable("data");
  if (TIME_MAJOR && isTrain)
    data = transpose(data);
  auto embed_weight = Symbol::Variable("embed_weight");
  auto embed = Embedding("embed", data, embed_weight, input_dim, num_embed);
  auto wordvec = isTrain? SliceChannel(embed, sequence_length, TIME_MAJOR? 0 : 1, true) : embed;

  std::vector<LSTMState> last_states;
  std::vector<LSTMParam> param_cells;
  for (int l = 0; l < num_lstm_layer; l++) {
    std::string layer = "l" + std::to_string(l);
    LSTMParam param;
    param.i2h_weight = Symbol::Variable(layer + "_i2h_weight");
    param.i2h_bias = Symbol::Variable(layer + "_i2h_bias");
    param.h2h_weight = Symbol::Variable(layer + "_h2h_weight");
    param.h2h_bias = Symbol::Variable(layer + "_h2h_bias");
    param_cells.push_back(param);
    LSTMState state;
    state.C = Symbol::Variable(layer + "_init_c");
    state.h = Symbol::Variable(layer + "_init_h");
    last_states.push_back(state);
  }

  std::vector<Symbol> hidden_all;
  for (int i = 0; i < sequence_length; i++) {
    auto hidden = wordvec[i];
    for (int layer = 0; layer < num_lstm_layer; layer++) {
      double dp_ratio = layer == 0? 0 : dropout;
      auto next_state = LSTM(num_hidden, hidden, last_states[layer], param_cells[layer],
          i, layer, dp_ratio);
      hidden = next_state.h;
      last_states[layer] = next_state;
    }
    if (dropout > 0)
      hidden = Dropout(hidden, dropout);
    hidden_all.push_back(hidden);
  }

  auto hidden_concat = isTrain? Concat(hidden_all, hidden_all.size(), 0) : hidden_all[0];
  auto cls_weight = Symbol::Variable("cls_weight");
  auto cls_bias = Symbol::Variable("cls_bias");
  auto pred = FullyConnected("pred", hidden_concat, cls_weight, cls_bias, input_dim);

  auto label = Symbol::Variable("softmax_label");
  label = transpose(label);
  label = Reshape(label, Shape(), false, Shape(0), false);  // -1: infer from graph
  auto sm = SoftmaxOutput("softmax", pred, label);
  if (isTrain)
    return sm;

  std::vector<Symbol> outputs = { sm };
  for (auto& state : last_states) {
    outputs.push_back(state.C);
    outputs.push_back(state.h);
  }
  return Symbol::Group(outputs);
}

// Currently mxnet GPU version RNN operator is implemented via *fast* NVIDIA cuDNN.
Symbol LSTMWithBuiltInRNNOp(int num_lstm_layer, int sequence_length, int input_dim,
 int num_hidden, int num_embed, mx_float dropout = 0) {
  auto isTrain = sequence_length > 1;
  auto data = Symbol::Variable("data");
  if (TIME_MAJOR && isTrain)
    data = transpose(data);

  auto embed_weight = Symbol::Variable("embed_weight");
  auto embed = Embedding("embed", data, embed_weight, input_dim, num_embed);
  auto label = Symbol::Variable("softmax_label");
  label = transpose(label);
  label = Reshape(label, Shape(), false,
                  Shape(0), false);  // FullyConnected requires one dimension
  if (!TIME_MAJOR && isTrain)
    embed = SwapAxis(embed, 0, 1);  // Change to time-major as cuDNN requires

  // We need not do the SwapAxis op as python version does. Direct and better performance in C++!
  auto rnn_h_init = Symbol::Variable("LSTM_init_h");
  auto rnn_c_init = Symbol::Variable("LSTM_init_c");
  auto rnn_params = Symbol::Variable("LSTM_parameters");  // See explanations near RNNXavier class
  auto variable_sequence_length = Symbol::Variable("sequence_length");
  auto rnn = RNN(embed, rnn_params, rnn_h_init, rnn_c_init, variable_sequence_length, num_hidden,
                 num_lstm_layer, RNNMode::kLstm, false, dropout, !isTrain);
  auto hidden = Reshape(rnn[0], Shape(), false, Shape(0, num_hidden), false);

  auto cls_weight = Symbol::Variable("cls_weight");
  auto cls_bias = Symbol::Variable("cls_bias");
  auto pred = FullyConnected("pred", hidden, cls_weight, cls_bias, input_dim);
  /*In rnn-time-major/rnn_cell_demo.py, the author claimed time-major version speeds up
   * 1.5~2 times versus batch version. I doubts on the conclusion. In my test, the performance
   * of both codes are almost same. In fact, there are no substantially differences between
   * two codes. They are both based on time major cuDNN, the computation graph only differs
   * slightly on the choices of where to put Reshape/SwapAxis/transpose operation. Here I don't
   * use Reshape on pred and keep label shape on SoftmaxOutput like time major version code,
   * but Reshape on label for simplification. It doesn't make influence on performacne. */

  auto sm = SoftmaxOutput("softmax", pred, label);
  if (isTrain)
    return sm;
  else
    return Symbol::Group({ sm, rnn[1/*RNNOpOutputs::kStateOut=1*/],
    rnn[2/*RNNOpOutputs::kStateCellOut=2*/] });
}

class Shuffler {
  std::vector<int> sequence;
 public:
  explicit Shuffler(int size) : sequence(size) {
    int* p = sequence.data();
    for (int i = 0; i < size; i++)
      *p++ = i;
  }
  void shuffle(std::function<void(int, int)> lambda = nullptr) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(sequence.begin(), sequence.end(), g);
    int n = 0;
    if (lambda != nullptr)
      for (int i : sequence)
        lambda(n++, i);
  }
  const int* data() {
    return sequence.data();
  }
};

class BucketSentenceIter : public DataIter {
  Shuffler* random;
  int batch, current, end;
  unsigned int sequence_length;
  Context device;
  std::vector<std::vector<mx_float>> sequences;
  std::vector<wchar_t> index2chars;
  std::unordered_map<wchar_t, mx_float> charIndices;

 public:
  BucketSentenceIter(std::string filename, int minibatch, Context context) : batch(minibatch),
  current(-1), device(context) {
    auto content = readContent(filename);
    buildCharIndex(content);
    sequences = convertTextToSequences(content, '\n');

    int N = sequences.size() / batch * batch;  // total used samples
    sequences.resize(N);
    sort(sequences.begin(), sequences.end(), [](const std::vector<mx_float>& a,
        const std::vector<mx_float>& b) { return a.size() < b.size(); });

    sequence_length = sequences.back().size();
    random = new Shuffler(N);
    // We still can get random results if call Reset() firstly
//    std::vector<vector<mx_float>>* target = &sequences;
//    random->shuffle([target](int n, int i) { (*target)[n].swap((*target)[i]); });
    end = N / batch;
  }
  virtual ~BucketSentenceIter() {
    delete random;
  }

  unsigned int maxSequenceLength() {
    return sequence_length;
  }

  size_t characterSize() {
    return charIndices.size();
  }

  virtual bool Next(void) {
    return ++current < end;
  }
  virtual NDArray GetData(void) {
    const int* indices = random->data();
    mx_float *data = new mx_float[sequence_length * batch], *pdata = data;

    for (int i = current * batch, end = i + batch; i < end; i++) {
      memcpy(pdata, sequences[indices[i]].data(), sequences[indices[i]].size() * sizeof(mx_float));
      if (sequences[indices[i]].size() < sequence_length)
        memset(pdata + sequences[indices[i]].size(), 0,
            (sequence_length - sequences[indices[i]].size()) * sizeof(mx_float));
      pdata += sequence_length;
    }
    NDArray array(Shape(batch, sequence_length), device, false);
    array.SyncCopyFromCPU(data, batch * sequence_length);
    return array;
  }
  virtual NDArray GetLabel(void) {
    const int* indices = random->data();
    mx_float *label = new mx_float[sequence_length * batch], *plabel = label;

    for (int i = current * batch, end = i + batch; i < end; i++) {
      memcpy(plabel, sequences[indices[i]].data() + 1,
          (sequences[indices[i]].size() - 1) * sizeof(mx_float));
      memset(plabel + sequences[indices[i]].size() - 1, 0,
          (sequence_length - sequences[indices[i]].size() + 1) * sizeof(mx_float));
      plabel += sequence_length;
    }
    NDArray array(Shape(batch, sequence_length), device, false);
    array.SyncCopyFromCPU(label, batch * sequence_length);
    return array;
  }
  virtual int GetPadNum(void) {
    return sequence_length - sequences[random->data()[current * batch]].size();
  }
  virtual std::vector<int> GetIndex(void) {
    const int* indices = random->data();
    std::vector<int> list(indices + current * batch, indices + current * batch + batch);
    return list;
  }
  virtual void BeforeFirst(void) {
    current = -1;
    random->shuffle(nullptr);
  }

  std::wstring readContent(const std::string file) {
    std::wifstream ifs(file, std::ios::binary);
    if (ifs) {
      std::wostringstream os;
      os << ifs.rdbuf();
      return os.str();
    }
    return L"";
  }

  void buildCharIndex(const std::wstring& content) {
  // This version buildCharIndex() Compatiable with python version char_rnn dictionary
    int n = 1;
    charIndices['\0'] = 0;  // padding character
    index2chars.push_back(0);  // padding character index
    for (auto c : content)
      if (charIndices.find(c) == charIndices.end()) {
        charIndices[c] = n++;
        index2chars.push_back(c);
      }
  }
//  void buildCharIndex(wstring& content) {
//    for (auto c : content)
//      charIndices[c]++; // char-frequency map; then char-index map
//    std::vector<tuple<wchar_t, mx_float>> characters;
//    for (auto& iter : charIndices)
//      characters.push_back(make_tuple(iter.first, iter.second));
//    sort(characters.begin(), characters.end(), [](const tuple<wchar_t, mx_float>& a,
//      const tuple<wchar_t, mx_float>& b) { return get<1>(a) > get<1>(b); });
//    mx_float index = 1; //0 is left for zero-padding
//    index2chars.clear();
//    index2chars.push_back(0); //zero-padding
//    for (auto& t : characters) {
//      charIndices[get<0>(t)] = index++;
//      index2chars.push_back(get<0>(t));
//    }s
//  }

  inline wchar_t character(int i) {
    return index2chars[i];
  }

  inline mx_float index(wchar_t c) {
    return charIndices[c];
  }

  void saveCharIndices(const std::string file) {
    std::wofstream ofs(file, std::ios::binary);
    if (ofs) {
      ofs.write(index2chars.data() + 1, index2chars.size() - 1);
      ofs.close();
    }
  }

  static std::tuple<std::unordered_map<wchar_t, mx_float>, std::vector<wchar_t>> loadCharIndices(
      const std::string file) {
    std::wifstream ifs(file, std::ios::binary);
    std::unordered_map<wchar_t, mx_float> map;
    std::vector<wchar_t> chars;
    if (ifs) {
      std::wostringstream os;
      os << ifs.rdbuf();
      int n = 1;
      map[L'\0'] = 0;
      chars.push_back(L'\0');
      for (auto c : os.str()) {
        map[c] = (mx_float) n++;
        chars.push_back(c);
      }
    }
    // Note: Can't use {} because this would hit the explicit constructor
    return std::tuple<std::unordered_map<wchar_t, mx_float>, std::vector<wchar_t>>(map, chars);
  }

  std::vector<std::vector<mx_float>>
  convertTextToSequences(const std::wstring& content, wchar_t spliter) {
    std::vector<std::vector<mx_float>> sequences;
    sequences.push_back(std::vector<mx_float>());
    for (auto c : content)
      if (c == spliter && !sequences.back().empty())
        sequences.push_back(std::vector<mx_float>());
      else
        sequences.back().push_back(charIndices[c]);
    return sequences;
  }
};

void OutputPerplexity(NDArray* labels, NDArray* output) {
  std::vector<mx_float> charIndices, a;
  labels->SyncCopyToCPU(&charIndices, 0L);  // 0L indicates all
  output->SyncCopyToCPU(&a, 0L)/*4128*84*/;
  mx_float loss = 0;
  int batchSize = labels->GetShape()[0]/*32*/, sequenceLength = labels->GetShape()[1]/*129*/,
      nSamples = output->GetShape()[0]/*4128*/, vocabSize = output->GetShape()[1]/*84*/;
  for (int n = 0; n < nSamples; n++) {
    int row = n % batchSize, column = n / batchSize, labelOffset = column +
        row * sequenceLength;  // Search based on column storage: labels.T
    mx_float safe_value = std::max(1e-10f, a[vocabSize * n +
                                    static_cast<int>(charIndices[labelOffset])]);
    loss += -log(safe_value);  // Calculate negative log-likelihood
  }
  loss = exp(loss / nSamples);
  std::cout << "Train-Perplexity=" << loss << std::endl;
}

void SaveCheckpoint(const std::string filepath, Symbol net, Executor* exe) {
  std::map<std::string, NDArray> params;
  for (auto iter : exe->arg_dict())
    if (iter.first.find("_init_") == std::string::npos
        && iter.first.rfind("data") != iter.first.length() - 4
        && iter.first.rfind("label") != iter.first.length() - 5)
      params.insert({"arg:" + iter.first, iter.second});
  for (auto iter : exe->aux_dict())
      params.insert({"aux:" + iter.first, iter.second});
  NDArray::Save(filepath, params);
}

void LoadCheckpoint(const std::string filepath, Executor* exe) {
  std::map<std::string, NDArray> params = NDArray::LoadToMap(filepath);
  for (auto iter : params) {
    std::string type = iter.first.substr(0, 4);
    std::string name = iter.first.substr(4);
    NDArray target;
    if (type == "arg:")
      target = exe->arg_dict()[name];
    else if (type == "aux:")
      target = exe->aux_dict()[name];
    else
      continue;
    iter.second.CopyTo(&target);
  }
}

int input_dim = 0;/*84*/
int sequence_length_max = 0;/*129*/
int num_embed = 256;
int num_lstm_layer = 3;
int num_hidden = 512;
mx_float dropout = 0.2;
void train(const std::string file, int batch_size, int max_epoch, int start_epoch) {
  Context device(DeviceType::kGPU, 0);
  BucketSentenceIter dataIter(file, batch_size, device);
  std::string prefix = file.substr(0, file.rfind("."));
  dataIter.saveCharIndices(prefix + ".dictionary");

  input_dim = static_cast<int>(dataIter.characterSize());
  sequence_length_max = dataIter.maxSequenceLength();

  auto RNN = LSTMUnroll(num_lstm_layer, sequence_length_max, input_dim, num_hidden,
      num_embed, dropout);
  std::map<std::string, NDArray> args_map;
  args_map["data"] = NDArray(Shape(batch_size, sequence_length_max), device, false);
  args_map["softmax_label"] = NDArray(Shape(batch_size, sequence_length_max), device, false);
  for (int i = 0; i < num_lstm_layer; i++) {
    std::string key = "l" + std::to_string(i) + "_init_";
    args_map[key + "c"] = NDArray(Shape(batch_size, num_hidden), device, false);
    args_map[key + "h"] = NDArray(Shape(batch_size, num_hidden), device, false);
  }
  std::vector<mx_float> zeros(batch_size * num_hidden, 0);
  // RNN.SimpleBind(device, args_map, {}, {{"data", kNullOp}});
  Executor* exe = RNN.SimpleBind(device, args_map);

  if (start_epoch == -1) {
    auto initializer = Uniform(0.07);
    for (auto &arg : exe->arg_dict()) {
      initializer(arg.first, &arg.second);
    }
  } else {
    LoadCheckpoint(prefix + "-" + std::to_string(start_epoch) + ".params", exe);
  }
  start_epoch++;

  mx_float learning_rate = 0.0002;
  mx_float weight_decay = 0.000002;
  Optimizer* opt = OptimizerRegistry::Find("sgd");
  opt->SetParam("lr", learning_rate)
     ->SetParam("wd", weight_decay);
//  opt->SetParam("momentum", 0.9)->SetParam("rescale_grad", 1.0 / batch_size)
//  ->SetParam("clip_gradient", 10);

  for (int epoch = start_epoch; epoch < max_epoch; ++epoch) {
    dataIter.Reset();
    auto tic =  std::chrono::system_clock::now();
    while (dataIter.Next()) {
      auto data_batch = dataIter.GetDataBatch();
      data_batch.data.CopyTo(&exe->arg_dict()["data"]);
      data_batch.label.CopyTo(&exe->arg_dict()["softmax_label"]);
      for (int l = 0; l < num_lstm_layer; l++) {
        std::string key = "l" + std::to_string(l) + "_init_";
        exe->arg_dict()[key + "c"].SyncCopyFromCPU(zeros);
        exe->arg_dict()[key + "h"].SyncCopyFromCPU(zeros);
      }
      NDArray::WaitAll();

      exe->Forward(true);
      exe->Backward();
      for (size_t i = 0; i < exe->arg_arrays.size(); ++i) {
        opt->Update(i, exe->arg_arrays[i], exe->grad_arrays[i]);
      }

      NDArray::WaitAll();
    }
    auto toc =  std::chrono::system_clock::now();
    std::cout << "Epoch[" << epoch << "] Time Cost:" <<
         std::chrono::duration_cast< std::chrono::seconds>(toc - tic).count() << " seconds ";
    OutputPerplexity(&exe->arg_dict()["softmax_label"], &exe->outputs[0]);
    std::string filepath = prefix + "-" + std::to_string(epoch) + ".params";
    SaveCheckpoint(filepath, RNN, exe);
  }

  delete exe;
  delete opt;
}

/*The original example, rnn_cell_demo.py, uses default Xavier as initalizer, which relies on
 * variable name, cannot initialize LSTM_parameters. Thus it was renamed to LSTM_bias,
 * which can be initialized as zero. But it cannot converge after 100 epochs in this corpus
 * example. Using RNNXavier, after 15 oscillating epochs,  it rapidly converges like old
 * LSTMUnroll version. */
class RNNXavier : public Xavier {
 public:
  RNNXavier(RandType rand_type = gaussian, FactorType factor_type = avg,
    float magnitude = 3) : Xavier(rand_type, factor_type, magnitude) {
  }
  virtual ~RNNXavier() {}
 protected:
  virtual void InitDefault(NDArray* arr) {
    Xavier::InitWeight(arr);
  }
};

void trainWithBuiltInRNNOp(const std::string file, int batch_size, int max_epoch, int start_epoch) {
  Context device(DeviceType::kGPU, 0);
  BucketSentenceIter dataIter(file, batch_size, device);
  std::string prefix = file.substr(0, file.rfind("."));
  dataIter.saveCharIndices(prefix + ".dictionary");

  input_dim = static_cast<int>(dataIter.characterSize());
  sequence_length_max = dataIter.maxSequenceLength();

  auto RNN = LSTMWithBuiltInRNNOp(num_lstm_layer, sequence_length_max, input_dim, num_hidden,
      num_embed, dropout);
  std::map<std::string, NDArray> args_map;
  args_map["data"] = NDArray(Shape(batch_size, sequence_length_max), device, false);
  // Avoiding SwapAxis, batch_size is of second dimension.
  args_map["LSTM_init_c"] = NDArray(Shape(num_lstm_layer, batch_size, num_hidden), device, false);
  args_map["LSTM_init_h"] = NDArray(Shape(num_lstm_layer, batch_size, num_hidden), device, false);
  args_map["softmax_label"] = NDArray(Shape(batch_size, sequence_length_max), device, false);
  std::vector<mx_float> zeros(batch_size * num_lstm_layer * num_hidden, 0);
  Executor* exe = RNN.SimpleBind(device, args_map);

  if (start_epoch == -1) {
    RNNXavier xavier = RNNXavier(Xavier::gaussian, Xavier::in, 2.34);
    for (auto &arg : exe->arg_dict())
      xavier(arg.first, &arg.second);
  } else {
    LoadCheckpoint(prefix + "-" + std::to_string(start_epoch) + ".params", exe);
  }
  start_epoch++;

  Optimizer* opt = OptimizerRegistry::Find("ccsgd");
//  opt->SetParam("momentum", 0.9)->SetParam("rescale_grad", 1.0 / batch_size)
//  ->SetParam("clip_gradient", 10);

  for (int epoch = start_epoch; epoch < max_epoch; ++epoch) {
    dataIter.Reset();
    auto tic =  std::chrono::system_clock::now();
    while (dataIter.Next()) {
      auto data_batch = dataIter.GetDataBatch();
      data_batch.data.CopyTo(&exe->arg_dict()["data"]);
      data_batch.label.CopyTo(&exe->arg_dict()["softmax_label"]);
      exe->arg_dict()["LSTM_init_c"].SyncCopyFromCPU(zeros);
      exe->arg_dict()["LSTM_init_h"].SyncCopyFromCPU(zeros);
      NDArray::WaitAll();

      exe->Forward(true);
      exe->Backward();
      for (size_t i = 0; i < exe->arg_arrays.size(); ++i) {
        opt->Update(i, exe->arg_arrays[i], exe->grad_arrays[i]);
      }
      NDArray::WaitAll();
    }
    auto toc =  std::chrono::system_clock::now();
    std::cout << "Epoch[" << epoch << "] Time Cost:" <<
         std::chrono::duration_cast< std::chrono::seconds>(toc - tic).count() << " seconds ";
    OutputPerplexity(&exe->arg_dict()["softmax_label"], &exe->outputs[0]);
    std::string filepath = prefix + "-" + std::to_string(epoch) + ".params";
    SaveCheckpoint(filepath, RNN, exe);
  }

  delete exe;
  delete opt;
}

void predict(std::wstring* ptext, int sequence_length, const std::string param_file,
    const std::string dictionary_file) {
  Context device(DeviceType::kGPU, 0);
  auto results = BucketSentenceIter::loadCharIndices(dictionary_file);
  auto dictionary = std::get<0>(results);
  auto charIndices = std::get<1>(results);
  input_dim = static_cast<int>(charIndices.size());
  auto RNN = LSTMUnroll(num_lstm_layer, 1, input_dim, num_hidden, num_embed, 0);

  std::map<std::string, NDArray> args_map;
  args_map["data"] = NDArray(Shape(1, 1), device, false);
  args_map["softmax_label"] = NDArray(Shape(1, 1), device, false);
  std::vector<mx_float> zeros(1 * num_hidden, 0);
  for (int l = 0; l < num_lstm_layer; l++) {
    std::string key = "l" + std::to_string(l) + "_init_";
    args_map[key + "c"] = NDArray(Shape(1, num_hidden), device, false);
    args_map[key + "h"] = NDArray(Shape(1, num_hidden), device, false);
    args_map[key + "c"].SyncCopyFromCPU(zeros);
    args_map[key + "h"].SyncCopyFromCPU(zeros);
  }
  Executor* exe = RNN.SimpleBind(device, args_map);
  LoadCheckpoint(param_file, exe);

  mx_float index;
  wchar_t next = 0;
  std::vector<mx_float> softmax;
  softmax.resize(input_dim);
  for (auto c : *ptext) {
    exe->arg_dict()["data"].SyncCopyFromCPU(&dictionary[c], 1);
    exe->Forward(false);

    exe->outputs[0].SyncCopyToCPU(softmax.data(), input_dim);
    for (int l = 0; l < num_lstm_layer; l++) {
      std::string key = "l" + std::to_string(l) + "_init_";
      exe->outputs[l * 2 + 1].CopyTo(&args_map[key + "c"]);
      exe->outputs[l * 2 + 2].CopyTo(&args_map[key + "h"]);
    }

    size_t n = max_element(softmax.begin(), softmax.end()) - softmax.begin();
    index = (mx_float) n;
    next = charIndices[n];
  }
  ptext->push_back(next);

  for (int i = 0; i < sequence_length; i++) {
    exe->arg_dict()["data"].SyncCopyFromCPU(&index, 1);
    exe->Forward(false);

    exe->outputs[0].SyncCopyToCPU(softmax.data(), input_dim);
    for (int l = 0; l < num_lstm_layer; l++) {
      std::string key = "l" + std::to_string(l) + "_init_";
      exe->outputs[l * 2 + 1].CopyTo(&args_map[key + "c"]);
      exe->outputs[l * 2 + 2].CopyTo(&args_map[key + "h"]);
    }

    size_t n = max_element(softmax.begin(), softmax.end()) - softmax.begin();
    index = (mx_float) n;
    next = charIndices[n];
    ptext->push_back(next);
  }

  delete exe;
}

void predictWithBuiltInRNNOp(std::wstring* ptext, int sequence_length, const std::string param_file,
  const std::string dictionary_file) {
  Context device(DeviceType::kGPU, 0);
  auto results = BucketSentenceIter::loadCharIndices(dictionary_file);
  auto dictionary = std::get<0>(results);
  auto charIndices = std::get<1>(results);
  input_dim = static_cast<int>(charIndices.size());
  auto RNN = LSTMWithBuiltInRNNOp(num_lstm_layer, 1, input_dim, num_hidden, num_embed, 0);

  std::map<std::string, NDArray> args_map;
  args_map["data"] = NDArray(Shape(1, 1), device, false);
  args_map["softmax_label"] = NDArray(Shape(1, 1), device, false);
  std::vector<mx_float> zeros(1 * num_lstm_layer * num_hidden, 0);
  // Avoiding SwapAxis, batch_size=1 is of second dimension.
  args_map["LSTM_init_c"] = NDArray(Shape(num_lstm_layer, 1, num_hidden), device, false);
  args_map["LSTM_init_h"] = NDArray(Shape(num_lstm_layer, 1, num_hidden), device, false);
  args_map["LSTM_init_c"].SyncCopyFromCPU(zeros);
  args_map["LSTM_init_h"].SyncCopyFromCPU(zeros);
  Executor* exe = RNN.SimpleBind(device, args_map);
  LoadCheckpoint(param_file, exe);

  mx_float index;
  wchar_t next = 0;
  std::vector<mx_float> softmax;
  softmax.resize(input_dim);
  for (auto c : *ptext) {
    exe->arg_dict()["data"].SyncCopyFromCPU(&dictionary[c], 1);
    exe->Forward(false);

    exe->outputs[0].SyncCopyToCPU(softmax.data(), input_dim);
    exe->outputs[1].CopyTo(&args_map["LSTM_init_h"]);
    exe->outputs[2].CopyTo(&args_map["LSTM_init_c"]);

    size_t n = max_element(softmax.begin(), softmax.end()) - softmax.begin();
    index = (mx_float) n;
    next = charIndices[n];
  }
  ptext->push_back(next);

  for (int i = 0; i < sequence_length; i++) {
    exe->arg_dict()["data"].SyncCopyFromCPU(&index, 1);
    exe->Forward(false);

    exe->outputs[0].SyncCopyToCPU(softmax.data(), input_dim);
    exe->outputs[1].CopyTo(&args_map["LSTM_init_h"]);
    exe->outputs[2].CopyTo(&args_map["LSTM_init_c"]);

    size_t n = max_element(softmax.begin(), softmax.end()) - softmax.begin();
    index = (mx_float) n;
    next = charIndices[n];
    ptext->push_back(next);
  }

  delete exe;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cout << "Usage for training: charRNN train[BuiltIn][TimeMajor] {corpus file}"
            " {batch size} {max epoch} [{starting epoch}]" << std::endl;
    std::cout <<"Usage for prediction: charRNN predict[BuiltIn][TimeMajor] {params file}"
            " {dictionary file} {beginning of text}" << std::endl;
    std::cout <<"Note: The {params file} of train/trainBuiltIn/trainTimeMajor/trainBuiltInTimeMajor"
            " are not compatible with each other." << std::endl;
    return 0;
  }

  std::string task = argv[1];
  bool builtIn = task.find("BuiltIn") != std::string::npos;
  TIME_MAJOR = task.find("TimeMajor") != std::string::npos;
  std::cout << "use BuiltIn cuDNN RNN: " << builtIn << std::endl
         << "use data as TimeMajor: " << TIME_MAJOR << std::endl;
  TRY
  if (task.find("train") == 0) {
    std::cout << "train batch size:      " << argv[3] << std::endl
           << "train max epoch:       " << argv[4] << std::endl;
    int start_epoch = argc > 5? atoi(argv[5]) : -1;
    // this function will generate dictionary file and params file.
    if (builtIn)
      trainWithBuiltInRNNOp(argv[2], atoi(argv[3]), atoi(argv[4]), start_epoch);
    else
      train(argv[2], atoi(argv[3]), atoi(argv[4]), start_epoch);  // ditto
  } else if (task.find("predict") == 0) {
    std::wstring text;  // = L"If there is anyone out there who still doubts ";
    // Considering of extending to Chinese samples in future, use wchar_t instead of char
    for (char c : std::string(argv[4]))
      text.push_back((wchar_t) c);
    /*Python version predicts text default to random selecltions. Here I didn't write the random
    code, always choose the 'best' character. So the text length reduced to 600. Longer size often
    leads to repeated sentances, since training sequence length is only 129 for obama corpus.*/
    if (builtIn)
      predictWithBuiltInRNNOp(&text, 600, argv[2], argv[3]);
    else
      predict(&text, 600, argv[2], argv[3]);
    std::wcout << text << std::endl;
  }

  MXNotifyShutdown();
  CATCH
  return 0;
}
