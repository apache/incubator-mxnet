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
 * Copyright (c) 2019 by Contributors
 * \file lib_api.cc
 * \brief APIs to interact with libraries
 * This API specifies function prototypes to
 * register custom ops, partitioner, and passes
 * for library authors
 * See example/extension/lib_custom_op/README.md
 * See example/extension/lib_subgraph/README.md
 * See example/extension/lib_pass/README.md
 */

#include "mxnet/lib_api.h"

mxnet::ext::MXerrorMsgs* mxnet::ext::MXerrorMsgs::get() {
    static MXerrorMsgs inst;
    return &inst;
  }

std::stringstream& mxnet::ext::MXerrorMsgs::add(const char* file, int line) {
  messages.emplace_back();
  messages.back() << file << "[" << line << "]: ";
  return messages.back();
}

int mxnet::ext::MXerrorMsgs::size() {
  return messages.size();
}

const std::string* mxnet::ext::MXerrorMsgs::get(int idx) {
  return new std::string(messages.at(idx).str());
}

mxnet::ext::MXContext::MXContext() : dev_type("error"), dev_id(-1) {}

mxnet::ext::MXContext::MXContext(std::string dev_type_, int dev_id_)
  : dev_type(std::move(dev_type_)), dev_id(dev_id_) {}

mxnet::ext::MXContext::MXContext(const char* dev_type_, int dev_id_)
  : dev_type(dev_type_), dev_id(dev_id_) {}

mxnet::ext::MXContext mxnet::ext::MXContext::CPU() { return MXContext("cpu", 0); }

mxnet::ext::MXContext mxnet::ext::MXContext::GPU() { return MXContext("gpu", 0); }

mxnet::ext::MXContext mxnet::ext::MXContext::CPU(int dev_id) { return MXContext("cpu", dev_id); }

mxnet::ext::MXContext mxnet::ext::MXContext::GPU(int dev_id) { return MXContext("gpu", dev_id); }

void mxnet::ext::MXSparse::set(void *data_ptr, const int64_t* dims, int ndims, void *idx,
                               int64_t num_idx, void *idx_ptr, int64_t num_idx_ptr) {
  data = data_ptr;
  // If CSR, num of non-zero elemets is num_idx,
  // If row sparse, num of elements is num_idx * width.
  data_len = num_idx;
  if (!idx_ptr) {
    for (int i = 1; i < ndims; ++i)
      data_len *= dims[i];
  }

  indices = reinterpret_cast<int64_t*>(idx);
  indices_len = num_idx;

  if (idx_ptr) {
    indptr = reinterpret_cast<int64_t*>(idx_ptr);
    indptr_len = num_idx_ptr;
  }
}

mxnet::ext::MXTensor::MXTensor() : data_ptr(nullptr), dtype(kUNSET), verID(0),
                                   stype(kDefaultStorage) {}
mxnet::ext::MXTensor::MXTensor(const MXTensor& oth) : data_ptr(oth.data_ptr), shape(oth.shape),
                                                      dtype(oth.dtype), verID(oth.verID),
                                                      ctx(oth.ctx), stype(oth.stype) {
  setDLTensor();
}

mxnet::ext::MXTensor::MXTensor(void *data_ptr, std::vector<int64_t> shape, MXDType dtype,
                               size_t vID, MXContext mx_ctx, MXStorageType stype)
  : data_ptr(data_ptr), shape(std::move(shape)), dtype(dtype), verID(vID), ctx(std::move(mx_ctx)),
    stype(stype) {
  setDLTensor();
}

void mxnet::ext::MXTensor::setTensor(void *dptr, MXDType type, const int64_t* dims, int ndims,
                                     size_t vID, MXContext mx_ctx, MXStorageType storage_type) {
  data_ptr = dptr; dtype = type; verID = vID; ctx = mx_ctx; stype = storage_type;
  shape.clear();
  for (int j = 0; j < ndims; j++) {
    shape.push_back(dims[j]);
  }
  setDLTensor();
}

void mxnet::ext::MXTensor::setDLTensor() {
  dltensor.data = data_ptr;
  dltensor.ndim = shape.size();
  dltensor.shape = const_cast<int64_t*>(shape.data());
  dltensor.strides = nullptr;
  dltensor.byte_offset = 0;
  dltensor.dtype.lanes = 1;
  dltensor.ctx.device_id = ctx.dev_id;
  if (ctx.dev_type == "cpu")
    dltensor.ctx.device_type = kDLCPU;
  else if (ctx.dev_type == "gpu")
    dltensor.ctx.device_type = kDLGPU;
  else if (ctx.dev_type == "opencl")
    dltensor.ctx.device_type = kDLOpenCL;
  else if (ctx.dev_type == "vulcan")
    dltensor.ctx.device_type = kDLVulkan;
  else if (ctx.dev_type == "metal")
    dltensor.ctx.device_type = kDLMetal;
  else if (ctx.dev_type == "vpi")
    dltensor.ctx.device_type = kDLVPI;
  else if (ctx.dev_type == "rocm")
    dltensor.ctx.device_type = kDLROCM;
  else
    dltensor.ctx.device_type = kDLExtDev;
  switch (dtype) {
  case kFloat32:
    dltensor.dtype.code = kDLFloat;
    dltensor.dtype.bits = 32;
    break;
  case kFloat64:
    dltensor.dtype.code = kDLFloat;
    dltensor.dtype.bits = 64;
    break;
  case kFloat16:
    dltensor.dtype.code = kDLFloat;
    dltensor.dtype.bits = 16;
    break;
  case kUint8:
    dltensor.dtype.code = kDLUInt;
    dltensor.dtype.bits = 8;
    break;
  case kInt32:
    dltensor.dtype.code = kDLInt;
    dltensor.dtype.bits = 32;
    break;
  case kInt8:
    dltensor.dtype.code = kDLInt;
    dltensor.dtype.bits = 8;
    break;
  case kUint16:
    dltensor.dtype.code = kDLUInt;
    dltensor.dtype.bits = 16;
    break;
  case kInt16:
    dltensor.dtype.code = kDLInt;
    dltensor.dtype.bits = 16;
    break;
  case kInt64:
    dltensor.dtype.code = kDLInt;
    dltensor.dtype.bits = 64;
    break;
  default:
    dltensor.dtype.code = 0;
    dltensor.dtype.bits = 0;
    throw std::runtime_error("Error! Invalid dtype flag: "
                             + std::to_string(static_cast<int>(dtype))
                             + " when constructing MXTensor");
  }
}

int64_t mxnet::ext::MXTensor::size() const {
  int64_t size = 1;
  for (auto &s : shape)
    size *= s;
  return size;
}

bool mxnet::ext::MXTensor::isSame(const MXTensor &oth) const {
  return data_ptr == oth.data_ptr &&
    dtype == oth.dtype &&
    verID == oth.verID &&
    ctx.dev_type == oth.ctx.dev_type &&
    ctx.dev_id == oth.ctx.dev_id &&
    shape == oth.shape &&
    stype == oth.stype;
}

mxnet::ext::PassResource::PassResource(std::unordered_map<std::string, MXTensor>* new_args,
                                       std::unordered_map<std::string, MXTensor>* new_aux,
                                       nd_malloc_t nd_malloc, const void* nd_alloc)
  : new_args_(new_args), new_aux_(new_aux), nd_malloc_(nd_malloc), nd_alloc_(nd_alloc) {}

mxnet::ext::MXTensor* mxnet::ext::PassResource::alloc_arg(const std::string& name,
                                                          const std::vector<int64_t>& shapes,
                                                          const mxnet::ext::MXContext &ctx,
                                                          mxnet::ext::MXDType dtype) const {
  void* data;
  nd_malloc_(nd_alloc_, shapes.data(), shapes.size(), ctx.dev_type.c_str(), ctx.dev_id,
             dtype, name.c_str(), 1, &data);
  MXTensor tensor(data, shapes, dtype, 0, ctx, kDefaultStorage);
  (*new_args_)[name] = tensor;
  return &(new_args_->at(name));
}

mxnet::ext::MXTensor* mxnet::ext::PassResource::alloc_aux(const std::string& name,
                                                          const std::vector<int64_t>& shapes,
                                                          const mxnet::ext::MXContext &ctx,
                                                          mxnet::ext::MXDType dtype) const {
  void* data;
  nd_malloc_(nd_alloc_, shapes.data(), shapes.size(), ctx.dev_type.c_str(), ctx.dev_id,
             dtype, name.c_str(), 0, &data);
  MXTensor tensor(data, shapes, dtype, 0, ctx, kDefaultStorage);
  (*new_aux_)[name] = tensor;
  return &(new_aux_->at(name));
}

mxnet::ext::OpResource::OpResource(xpu_malloc_t cpu_malloc_fp, void* cpu_alloc_fp,
                                   xpu_malloc_t gpu_malloc_fp, void* gpu_alloc_fp, void* stream,
                                   sparse_malloc_t sparse_malloc_fp, void* sparse_alloc_fp,
                                   void* rng_cpu_states, void* rng_gpu_states)
  : cpu_malloc(cpu_malloc_fp), gpu_malloc(gpu_malloc_fp),
    cpu_alloc(cpu_alloc_fp), gpu_alloc(gpu_alloc_fp), cuda_stream(stream),
    sparse_malloc(sparse_malloc_fp), sparse_alloc(sparse_alloc_fp),
    rand_cpu_states(rng_cpu_states), rand_gpu_states(rng_gpu_states) {}

void* mxnet::ext::OpResource::alloc_cpu(int size) const {
  return cpu_malloc(cpu_alloc, size);
}

void* mxnet::ext::OpResource::alloc_gpu(int size) const {
  return gpu_malloc(gpu_alloc, size);
}

void mxnet::ext::OpResource::alloc_sparse(mxnet::ext::MXSparse* sparse, int index,
                                          int indices_len, int indptr_len) const {
  sparse_malloc(sparse_alloc, index, indices_len, indptr_len,
                &(sparse->data), &(sparse->indices), &(sparse->indptr));
}

mxnet::ext::mx_cpu_rand_t* mxnet::ext::OpResource::get_cpu_rand_states() const {
  return static_cast<mx_cpu_rand_t*>(rand_cpu_states);
}

std::string mxnet::ext::getShapeAt(const std::string& shape, unsigned index) {
  int idx = 1;  // start at 1 to skip the first square bracket [
  // find the beginning of the output shape for the particular output index
  for (unsigned x=0; x < index; x++)
    idx = shape.find('[', idx+1);
  int stop = shape.find(']', idx);  // find stop index for this output shape
  // add this shape to the list
  return shape.substr(idx, stop-idx+1);
}

std::string mxnet::ext::getDtypeAt(const std::string& dtype, unsigned index) {
  // find the beginning of the output dtype for the particular output index
  int idx = 0;
  for (unsigned x=0; x < index; x++)
    idx = dtype.find(',', idx+1);
  int stop = dtype.find(',', idx+1);  // find stop index for this output dtype
  if (stop == -1) stop = dtype.find(']', idx+1);
  return dtype.substr(idx+1, stop-idx-1);
}

mxnet::ext::JsonVal::JsonVal() : type(ERR), num(-1), str("") {}
mxnet::ext::JsonVal::JsonVal(mxnet::ext::JsonType t) : type(t), num(-1), str("") {}
mxnet::ext::JsonVal::JsonVal(std::string s) : type(STR), num(-1), str(std::move(s)) {}
mxnet::ext::JsonVal::JsonVal(int n) : type(NUM), num(n), str(std::to_string(n)) {}
mxnet::ext::JsonVal::JsonVal(JsonType t, int n, std::string s) : type(t), num(n),
                                                                 str(std::move(s)) {}

bool mxnet::ext::JsonVal::operator<(const mxnet::ext::JsonVal &o) const {
  // for string JSON objects compare the string
  if (type == STR) return type == o.type && str < o.str;
  // for number JSON objects compare the number
  if (type == NUM) return type == o.type && num < o.num;
  // for list JSON objects, compare the size of list, and then each object in the list
  if (type == LIST) {
    if (list.size() != o.list.size()) return false;
    for (unsigned int i=0; i< list.size(); i++)
      if (list[i] < o.list[i])
        return false;  // if we find an object that doesnt match return
    return true;  // all objects in lists matched
  }
  // for map JSON objects, compare the size of map, and then each key/value in the maps
  if (type == MAP) {
    if (map.size() != o.map.size()) return false;
    for (auto &item : map) {
      // if one map is missing a key in another return
      if (o.map.find(item.first) == o.map.end()) return false;
      if (item.second < o.map.at(item.first)) return false;
    }
    return true;
  }
  return type < o.type;
}

std::string mxnet::ext::JsonVal::dump() const {
  std::string ret;
  switch (type) {
  case ERR:
    ret = "json(Error)";
    break;
  case STR:
    ret = "\"" + str + "\"";
    break;
  case NUM:
    ret = str;
    break;
  case LIST:
    ret = "[";
    for (unsigned i=0; i < list.size(); i++) {
      auto &item = list[i];
      ret += item.dump();
      if (i < list.size()-1)
        ret += ",";
    }
    ret += "]";
    break;
  case MAP:
    ret = "{";
    unsigned cnt = 0;
    for (auto &item : map) {
      ret += item.first.dump() + " : " + item.second.dump();
      if (cnt++ < map.size()-1)
        ret += ",";
    }
    ret += "}";
    break;
  }
  return ret;
}

mxnet::ext::JsonVal mxnet::ext::JsonVal::parse(const std::string& json) {
  unsigned int idx = 0;
  return JsonVal::parse(json, &idx);
}

mxnet::ext::JsonVal mxnet::ext::JsonVal::parse_string(const std::string& json, unsigned int* idx) {
  JsonVal ret(STR);
  while (*idx < json.size()) {
    if (json[*idx] == '"' && (ret.str.size() == 0 ||
                              (ret.str.size() > 0 && ret.str.back() != '\\'))) {
      ++(*idx);
      return ret;
    } else {
      ret.str += json[*idx];
      ++(*idx);
    }
  }
  MX_ERROR_MSG << "Error! Unable to parse string: '" << json.substr(*idx) << "'" << std::endl;
  return JsonVal();
}

mxnet::ext::JsonVal mxnet::ext::JsonVal::parse_num(const std::string& json, unsigned int* idx) {
  JsonVal ret(NUM);
  while (*idx < json.size()) {
    if (json[*idx] >= '0' && json[*idx] <= '9') {
      ret.str += json[*idx];
      ++(*idx);
    } else {
      break;
    }
  }
  ret.num = std::stoi(ret.str);
  return ret;
}

mxnet::ext::JsonVal mxnet::ext::JsonVal::parse_list(const std::string& json, unsigned int* idx) {
  JsonVal ret(LIST);
  while (*idx < json.size()) {
    if (json[*idx] == ']') {
      ++(*idx);
      return ret;
    } else {
      JsonVal item = JsonVal::parse(json, idx);
      if (item.type != ERR)
        ret.list.push_back(item);
    }
  }
  MX_ERROR_MSG << "Error! Unable to parse list: '" << json.substr(*idx) << "'" << std::endl;
  return JsonVal();
}

mxnet::ext::JsonVal mxnet::ext::JsonVal::parse_map(const std::string& json, unsigned int* idx) {
  JsonVal ret(MAP), key;
  while (*idx < json.size()) {
    if (json[*idx] == '}') {
      ++(*idx);
      return ret;
    } else {
      JsonVal item = JsonVal::parse(json, idx);
      if (key.type == ERR) {
        key = item;
      } else {
        ret.map[key] = item;
        key.type = ERR;
      }
    }
  }
  MX_ERROR_MSG << "Error! Unable to parse map: '" << json.substr(*idx) << "'" << std::endl;
  return mxnet::ext::JsonVal();
}

mxnet::ext::JsonVal mxnet::ext::JsonVal::parse(const std::string& json, unsigned int *idx) {
  JsonVal ret;
  while (*idx < json.size()) {
    if (json[*idx] == '"') {
      ++(*idx);
      ret = JsonVal::parse_string(json, idx);
    } else if (json[*idx] >= '0' && json[*idx] <= '9') {
      ret = JsonVal::parse_num(json, idx);
    } else if (json[*idx] == '[') {
      ++(*idx);
      ret = JsonVal::parse_list(json, idx);
    } else if (json[*idx] == '{') {
      ++(*idx);
      ret = JsonVal::parse_map(json, idx);
    } else if (json[*idx] == ']' || json[*idx] == '}') {return ret;}
    if (ret.type != ERR) return ret;
    ++(*idx);
  }
  return ret;
}

std::string mxnet::ext::JsonVal::toString() const {
  std::string ret;
  switch (type) {
  case ERR:
    ret = "json(Error)";
    break;
  case STR:
    ret = "json(STR:" + str + ")";
    break;
  case NUM:
    ret = "json(INT:" + str + ")";
    break;
  case LIST:
    ret = "json(LIST:[";
    for (auto &item : list)
      ret += item.toString() + ",";
    ret += "])";
    break;
  case MAP:
    ret = "json(MAP:{";
    for (auto &item : map)
      ret += item.first.toString() + " : " + item.second.toString() + ",";
    ret += "})";
    break;
  }
  return ret;
}

mxnet::ext::Node::Node() {tensor = nullptr;}

void mxnet::ext::Node::_setPassResource(mxnet::ext::PassResource* res_) {res = res_;}

void mxnet::ext::Node::alloc_arg(const std::vector<int64_t>& shapes,
                                 const mxnet::ext::MXContext &ctx, mxnet::ext::MXDType dtype) {
  if (!res)
    throw std::runtime_error("Node not initialized. Cannot use alloc_arg outside of graph passes.");
  tensor = res->alloc_arg(name, shapes, ctx, dtype);
}

void mxnet::ext::Node::alloc_aux(const std::vector<int64_t>& shapes,
                                 const mxnet::ext::MXContext &ctx, mxnet::ext::MXDType dtype) {
  if (!res)
    throw std::runtime_error("Node not initialized. Cannot use alloc_aux outside of graph passes.");
  tensor = res->alloc_aux(name, shapes, ctx, dtype);
}

mxnet::ext::Graph::Graph() : res(nullptr) {}

mxnet::ext::Graph::~Graph() {
  for (auto &node : nodes)
    delete node;
}

mxnet::ext::Graph* mxnet::ext::Graph::fromString(const std::string& json) {
  JsonVal val = JsonVal::parse(json);
  return fromJson(val);
}

mxnet::ext::Graph* mxnet::ext::Graph::fromJson(mxnet::ext::JsonVal val) {
  // get nodes list
  JsonVal nodes = val.map[JsonVal("nodes")];
  Graph *g = new Graph();

  std::map<int, Node*> nodeMap;
  // loop over nodes
  for (int i = 0; i < nodes.list.size(); i++) {
    Node* n = new Node();
    g->nodes.push_back(n);
    JsonVal node = nodes.list[i];

    // set the op info
    n->op = node.map[JsonVal("op")].str;
    n->name = node.map[JsonVal("name")].str;

    // if op is null it is an input to the graph
    if (n->op.compare("null") == 0)
      g->inputs.push_back(n);

    // set attrs
    JsonVal attributes = node.map[JsonVal("attrs")];
    for (auto& kv : attributes.map) {
      n->attrs[kv.first.str] = kv.second.str;
    }

    // set subgraphs, parsing each into a graph
    if (node.map.count(JsonVal("subgraphs")) > 0) {
      JsonVal subgraphs = node.map[JsonVal("subgraphs")];
      for (auto &subgraph : subgraphs.list) {
        n->subgraphs.push_back(fromJson(subgraph));
      }
    }

    // set node inputs
    JsonVal node_inputs = node.map[JsonVal("inputs")];
    n->inputs.resize(node_inputs.list.size());
    for (int j = 0; j < node_inputs.list.size(); j++) {
      JsonVal input = node_inputs.list[j];
      NodeEntry& entry = n->inputs[j];
      // get pointer to other node
      entry.node = nodeMap[input.list[0].num];
      // get the other node's output index
      entry.entry = input.list[1].num;
      // set other nodes output as connected to this node
      entry.node->outputs.push_back({n, j});
    }
    nodeMap[i] = n;
  }

  // set graph level outputs
  JsonVal& heads = val.map[JsonVal("heads")];
  g->outputs.resize(heads.list.size());
  for (int i = 0; i < heads.list.size(); i++) {
    JsonVal head = heads.list[i];
    g->outputs[i].node = nodeMap[head.list[0].num];
    g->outputs[i].entry = head.list[1].num;
  }

  // add all attributes to the graph
  for (auto& kv : val.map) {
    if (kv.first.str.compare("nodes") != 0 &&
        kv.first.str.compare("heads") != 0 &&
        kv.first.str.compare("node_row_ptr") != 0 &&
        kv.first.str.compare("arg_nodes") != 0) {
      g->attrs[kv.first.str] = kv.second;
    }
  }
  return g;
}

/* \brief convert graph object back to JSON object */
mxnet::ext::JsonVal mxnet::ext::Graph::toJson() const {
  // top level object is a map
  JsonVal val(MAP);

  // add attributes
  for (auto& kv : attrs) {
    val.map[JsonVal(kv.first)] = kv.second;
  }

  // sort graph nodes in topological order, create mapping of node to index
  std::map<Node*, int> nodeMap;
  std::vector<Node*> sorted = topological_sort();
  // nodes are in reverse topological order in the vector (back is first)
  // so loop from end to front over the vector 'sorted'
  for (int i = sorted.size()-1; i >= 0; i--) {
    nodeMap[sorted[i]] = sorted.size()-1-i;
  }

  // create node_row_ptr entry
  val.map[JsonVal("node_row_ptr")] = JsonVal(LIST);
  JsonVal& node_row_ptr = val.map[JsonVal("node_row_ptr")];
  for (int i = 0; i < nodes.size(); i++)
    node_row_ptr.list.emplace_back(i);

  // add all input nodes
  val.map[JsonVal("arg_nodes")] = JsonVal(LIST);
  JsonVal& arg_nodes = val.map[JsonVal("arg_nodes")];
  for (auto &input : inputs)
    arg_nodes.list.emplace_back(nodeMap[input]);

  // add all output nodes
  val.map[JsonVal("heads")] = JsonVal(LIST);
  JsonVal& heads = val.map[JsonVal("heads")];
  for (int i = 0; i < outputs.size(); i++) {
    heads.list.emplace_back(LIST);
    JsonVal& out = heads.list[i];
    out.list.emplace_back(nodeMap[outputs[i].node]);
    out.list.emplace_back(outputs[i].entry);
    out.list.emplace_back(0);
  }

  // add all graph nodes
  val.map[JsonVal("nodes")] = JsonVal(LIST);
  JsonVal& nodes_ = val.map[JsonVal("nodes")];
  for (int i = sorted.size()-1; i >= 0; i--) {
    // each node is a map
    nodes_.list.emplace_back(MAP);
    Node* n = sorted[i];
    JsonVal& n_ = nodes_.list[nodes_.list.size()-1];

    n_.map[JsonVal("op")] = JsonVal(n->op);
    n_.map[JsonVal("name")] = JsonVal(n->name);
    n_.map[JsonVal("inputs")] = JsonVal(LIST);

    // add inputs for this node
    JsonVal& inputs_ = n_.map[JsonVal("inputs")];
    for (int j = 0; j < n->inputs.size(); j++) {
      inputs_.list.emplace_back(LIST);
      NodeEntry& entry = n->inputs[j];
      JsonVal& in = inputs_.list[j];
      in.list.emplace_back(nodeMap[entry.node]);
      in.list.emplace_back(entry.entry);
      in.list.emplace_back(0);
    }

    // add subgraphs for this node, convert each back to JSON
    if (n->subgraphs.size() > 0) {
      n_.map[JsonVal("subgraphs")] = JsonVal(LIST);
      JsonVal &subgraphs_ = n_.map[JsonVal("subgraphs")];
      for (Graph *subgraph : n->subgraphs) {
        subgraphs_.list.push_back(subgraph->toJson());
      }
    }

    // add attributes for this node
    n_.map[JsonVal("attrs")] = JsonVal(MAP);
    JsonVal& attrs_ = n_.map[JsonVal("attrs")];
    for (auto& kv : n->attrs) {
      attrs_.map[JsonVal(kv.first)] = JsonVal(kv.second);
    }
  }
  return val;
}

/* \brief convert graph object to JSON string */
std::string mxnet::ext::Graph::toString() const {
  return toJson().dump();
}

  /* \brief visits a node "n" */
void mxnet::ext::Graph::_dfs_util(Node* n, std::unordered_set<mxnet::ext::Node*>* to_visit,
                                  std::function<void(mxnet::ext::Node*)> handler) const {
  to_visit->erase(n);  // remove node now that we're visiting it
  for (NodeEntry& e : n->outputs) {
    Node* o = e.node;
    if (to_visit->count(o) != 0) {
      _dfs_util(o, to_visit, handler);  // visit neighbor
    }
  }
  handler(n);  // post-order visit this node
}

/* \brief post-order DFS graph traversal */
void mxnet::ext::Graph::DFS(std::function<void(Node*)> handler) const {
  std::unordered_set<Node*> to_visit;
  // put all nodes in set to visit
  for (auto& n : nodes)
    to_visit.insert(n);
  // visit all inputs first
  for (auto& i : inputs)
    if (to_visit.count(i) != 0)
      _dfs_util(i, &to_visit, handler);
  // visit any nodes left
  while (to_visit.size() > 0)
    _dfs_util(*(to_visit.begin()), &to_visit, handler);
}

/* \brief sort graph nodes in topological order */
std::vector<mxnet::ext::Node*> mxnet::ext::Graph::topological_sort() const {
  std::vector<mxnet::ext::Node*> sorted;
  auto handler = [&](mxnet::ext::Node* n) {
    sorted.push_back(n);  // when visiting each node, add it in order to the vector
  };
  DFS(handler);
  return sorted;
}

/* \brief print out graph details */
void mxnet::ext::Graph::print(int indent) const {
  std::string space = "";
  for (int i = 0; i < indent; i++) space+=" ";

  std::cout << space << "########### Graph #############" << std::endl;
  std::cout << space << "attributes: " << std::endl;
  for (auto &kv : attrs)
    std::cout << space << "\t" << kv.first << " : " << kv.second.str << std::endl;
  std::cout << space << "inputs: " << inputs.size() << std::endl;
  std::cout << space << "outputs: " << outputs.size() << std::endl;
  std::cout << space << "nodes: " << nodes.size() << std::endl;
  std::vector<mxnet::ext::Node*> sorted = topological_sort();
  // loop over each node and print out its inputs/outputs
  for (int i = sorted.size()-1; i >= 0; i--) {
    std::cout << space << "Node: " << sorted[i]->name << std::endl;
    for (auto &input : sorted[i]->inputs) {
      std::cout << space << "\tInput: " << input.node->name << " "
                << input.entry << std::endl;
    }
    for (auto &output : sorted[i]->outputs) {
      std::cout << space << "\tOutput: " << output.node->name << " "
                << output.entry << std::endl;
    }
    if (sorted[i]->subgraphs.size() > 0) {
      for (auto &subgraph : sorted[i]->subgraphs) {
        std::cout << space << "\tSubgraph:" << std::endl;
        subgraph->print(indent+2);
      }
    }
  }
  std::cout << space << "###############################" << std::endl;
}

/* \brief add a new node to this graph */
mxnet::ext::Node* mxnet::ext::Graph::addNode(const std::string& name, const std::string& op) {
  Node* n = new Node();
  nodes.push_back(n);
  n->name = name;
  n->op = op;
  if (res)
    n->_setPassResource(res);
  return n;
}

/* \brief get node at index in graph */
mxnet::ext::Node* mxnet::ext::Graph::getNode(size_t idx) {
  return nodes[idx];
}

/* \brief get const node at index in const graph */
const mxnet::ext::Node* mxnet::ext::Graph::getNode(size_t idx) const {
  return nodes.at(idx);
}

/* \brief get attribute on graph */
const mxnet::ext::JsonVal& mxnet::ext::Graph::getAttr(const std::string& key) const {
  return attrs.at(key);
}

/* \brief get number of nodes in the graph */
size_t mxnet::ext::Graph::size() const {
  return nodes.size();
}

// internally set passResource to enable tensor allocation for graph passes
void mxnet::ext::Graph::_setPassResource(PassResource* res_) {
  res = res_;
  // set passResource for each node
  for (Node* node : nodes) {
    node->_setPassResource(res);
  }
}

// internally set arg/aux params when available
void mxnet::ext::Graph::_setParams(std::unordered_map<std::string, mxnet::ext::MXTensor>* args,
                                   std::unordered_map<std::string, mxnet::ext::MXTensor>* aux) {
  // set params for each input node
  for (Node* node : inputs) {
    std::string name = node->name;
    if (node->attrs.count("isArg") > 0 && node->attrs["isArg"].compare("True") == 0)
      // mapping name back to original node name from subgraph input name
      name = node->attrs["argName"];
    if (args->count(name) > 0)
      node->tensor = &args->at(name);
    else if (aux->count(name) > 0)
      node->tensor = &aux->at(name);
  }
}

mxnet::ext::CustomOp::CustomOp(const char* op_name)
  : name(op_name), parse_attrs(nullptr), infer_type(nullptr), infer_storage_type(nullptr),
    infer_shape(nullptr), mutate_inputs(nullptr), isSGop(false) {}

mxnet::ext::CustomOp& mxnet::ext::CustomOp::setForward(mxnet::ext::fcomp_t fcomp, const char* ctx) {
  if (forward_ctx_map.count(ctx) > 0)
    raiseDuplicateContextError();
  forward_ctx_map[ctx] = fcomp;
  return *this;
}

mxnet::ext::CustomOp& mxnet::ext::CustomOp::setBackward(mxnet::ext::fcomp_t fgrad,
                                                        const char* ctx) {
  if (backward_ctx_map.count(ctx) > 0)
    raiseDuplicateContextError();
  backward_ctx_map[ctx] = fgrad;
  return *this;
}

mxnet::ext::CustomOp& mxnet::ext::CustomOp::setParseAttrs(mxnet::ext::parseAttrs_t func) {
  parse_attrs = func;
  return *this;
}

mxnet::ext::CustomOp& mxnet::ext::CustomOp::setInferType(mxnet::ext::inferType_t func) {
  infer_type = func;
  return *this;
}

mxnet::ext::CustomOp& mxnet::ext::CustomOp::setInferSType(mxnet::ext::inferSType_t func) {
  infer_storage_type = func;
  return *this;
}

mxnet::ext::CustomOp& mxnet::ext::CustomOp::setInferShape(mxnet::ext::inferShape_t func) {
  infer_shape = func;
  return *this;
}

mxnet::ext::CustomOp& mxnet::ext::CustomOp::setMutateInputs(mxnet::ext::mutateInputs_t func) {
  mutate_inputs = func;
  return *this;
}

mxnet::ext::CustomOp& mxnet::ext::CustomOp::setCreateOpState(mxnet::ext::createOpState_t func,
                                                             const char* ctx) {
  if (create_op_ctx_map.count(ctx) > 0)
    raiseDuplicateContextError();
  create_op_ctx_map[ctx] = func;
  return *this;
}

mxnet::ext::CustomOp& mxnet::ext::CustomOp::setIsSubgraphOp() {
  isSGop = true;
  return *this;
}

void mxnet::ext::CustomOp::mapToVector() {
  for (auto kv : forward_ctx_map) {
    forward_ctx_cstr.push_back(kv.first);
    forward_fp.push_back(kv.second);
  }
  for (auto kv : backward_ctx_map) {
    backward_ctx_cstr.push_back(kv.first);
    backward_fp.push_back(kv.second);
  }
  for (auto kv : create_op_ctx_map) {
    create_op_ctx_cstr.push_back(kv.first);
    create_op_fp.push_back(kv.second);
  }
}

void mxnet::ext::CustomOp::raiseDuplicateContextError() {
  std::string op_name_str(name);
  throw std::runtime_error(
    "Error! Error! Cannot register multiple functions under same context for operator '"
    + op_name_str + "'");
}

mxnet::ext::CustomStatefulOp::CustomStatefulOp() : ignore_warn(false), created(false) {}
mxnet::ext::CustomStatefulOp::~CustomStatefulOp() = default;

mxnet::ext::CustomStatefulOpWrapper::~CustomStatefulOpWrapper() {
  destroy_(instance);
}

mxnet::ext::CustomPass::CustomPass() : name("ERROR") {}
mxnet::ext::CustomPass::CustomPass(const char* pass_name)
  : name(pass_name) {}
mxnet::ext::CustomPass& mxnet::ext::CustomPass::setBody(graphPass_t fn) {
  pass = fn;
  return *this;
}

mxnet::ext::CustomPartitioner::CustomPartitioner() : name("ERROR") {}
mxnet::ext::CustomPartitioner::CustomPartitioner(const char* backend_name) :
  name(backend_name) {}

mxnet::ext::CustomPartitioner& mxnet::ext::CustomPartitioner::addStrategy(const char* prop_name,
                                                                          const char* sg_name) {
  strategies.push_back(prop_name);
  op_names.push_back(sg_name);
  return *this;
}

mxnet::ext::CustomPartitioner& mxnet::ext::CustomPartitioner::setSupportedOps(const char* prop_name,
                                                                    mxnet::ext::supportedOps_t fn) {
  supported_map[std::string(prop_name)] = fn;
  return *this;
}

mxnet::ext::CustomPartitioner& mxnet::ext::CustomPartitioner::setCreateSelector(
                                           const char* prop_name, mxnet::ext::createSelector_t fn) {
  selector_map[std::string(prop_name)] = fn;
  return *this;
}

mxnet::ext::CustomPartitioner& mxnet::ext::CustomPartitioner::setReviewSubgraph(
                                           const char* prop_name, mxnet::ext::reviewSubgraph_t fn) {
  review_map[std::string(prop_name)] = fn;
  return *this;
}

mxnet::ext::supportedOps_t mxnet::ext::CustomPartitioner::getSupportedOps(int stg_id) {
  std::string prop(strategies[stg_id]);
  if (supported_map.count(prop) > 0)
    return supported_map[prop];
  else
    return nullptr;
}

mxnet::ext::createSelector_t mxnet::ext::CustomPartitioner::getCreateSelector(int stg_id) {
  std::string prop(strategies[stg_id]);
  if (selector_map.count(prop) > 0)
    return selector_map[prop];
  else
    return nullptr;
}

mxnet::ext::reviewSubgraph_t mxnet::ext::CustomPartitioner::getReviewSubgraph(int stg_id) {
  std::string prop(strategies[stg_id]);
  if (review_map.count(prop) > 0)
    return review_map[prop];
  else
    return nullptr;
}

/*! \brief returns MXNet library version */
MX_INT_RET _opVersion() {
  return MX_LIBRARY_VERSION;
}

/*! \brief returns number of ops registered in this library */
MX_INT_RET _opRegSize() {
  return mxnet::ext::Registry<mxnet::ext::CustomOp>::get()->size();
}

/*! \brief returns operator registration at specified index */
MX_VOID_RET _opRegGet(int idx, const char** name, int *isSGop,
                      const char*** forward_ctx, mxnet::ext::fcomp_t** forward_fp,
                      int* forward_count, const char*** backward_ctx,
                      mxnet::ext::fcomp_t** backward_fp, int* backward_count,
                      const char*** create_op_ctx, mxnet::ext::createOpState_t** create_op_fp,
                      int* create_op_count, mxnet::ext::parseAttrs_t* parse,
                      mxnet::ext::inferType_t* type, mxnet::ext::inferSType_t* stype,
                      mxnet::ext::inferShape_t* shape, mxnet::ext::mutateInputs_t* mutate) {
  mxnet::ext::CustomOp &op = mxnet::ext::Registry<mxnet::ext::CustomOp>::get()->get(idx);
  *name = op.name;
  *parse = op.parse_attrs;
  *type = op.infer_type;
  *stype = op.infer_storage_type;
  *shape = op.infer_shape;
  *mutate = op.mutate_inputs;
  *isSGop = op.isSGop;
  op.mapToVector();
  *forward_ctx = op.forward_ctx_cstr.data();
  *forward_fp = op.forward_fp.data();
  *forward_count = op.forward_fp.size();
  *backward_ctx = op.backward_ctx_cstr.data();
  *backward_fp = op.backward_fp.data();
  *backward_count = op.backward_fp.size();
  *create_op_ctx = op.create_op_ctx_cstr.data();
  *create_op_fp = op.create_op_fp.data();
  *create_op_count = op.create_op_fp.size();
}

/*! \brief calls free from the external library for library allocated arrays */
MX_VOID_RET _opCallFree(void* ptr) {
  free(ptr);
}

/*! \brief returns status of calling parse attributes function for operator from library */
MX_INT_RET _opCallParseAttrs(mxnet::ext::parseAttrs_t parseAttrs, const char* const* keys,
                             const char* const* vals, int num,
                             int* num_in, int* num_out) {
  // create map of attributes from list
  std::unordered_map<std::string, std::string> attrs;
  for (int i = 0; i < num; i++) {
    attrs[std::string(keys[i])] = std::string(vals[i]);
  }
  return parseAttrs(attrs, num_in, num_out);
}

/*! \brief returns status of calling inferShape function for operator from library */
MX_INT_RET _opCallInferShape(mxnet::ext::inferShape_t inferShape, const char* const* keys,
                             const char* const* vals, int num,
                             unsigned int** inshapes, int* indims, int num_in,
                             unsigned int*** mod_inshapes, int** mod_indims,
                             unsigned int*** outshapes, int** outdims, int num_out) {
  // create map of attributes from list
  std::unordered_map<std::string, std::string> attrs;
  for (int i = 0; i < num; i++) {
    attrs[std::string(keys[i])] = std::string(vals[i]);
  }

  // create a vector of shapes for inputs
  std::vector<std::vector<unsigned int> > in_shapes(num_in);
  for (int i = 0; i < num_in; i++) {
    for (int j = 0; j < indims[i]; j++) {
      in_shapes[i].push_back(inshapes[i][j]);
    }
  }

  // create a vector of shapes for outputs
  std::vector<std::vector<unsigned int> > out_shapes(num_out);

  int retval = inferShape(attrs, &in_shapes, &out_shapes);
  if (!retval) return retval;

  // allocate space for modified input dims, shape
  *mod_indims = static_cast<int*>(malloc (num_in * sizeof(int)));
  *mod_inshapes = static_cast<unsigned**>(malloc (num_in * sizeof(unsigned*)));

  // copy modified input shapes
  for (int i = 0; i < num_in; i++) {
    (*mod_indims)[i] = in_shapes[i].size();
    (*mod_inshapes)[i] = static_cast<unsigned*>(malloc ((*mod_indims)[i] * sizeof(unsigned)));
    for (int j = 0; j < (*mod_indims)[i]; j++) {
      (*mod_inshapes)[i][j] = in_shapes[i][j];
    }
  }

  // allocate space for output dims, shape
  *outdims = static_cast<int*>(malloc (num_out * sizeof(int)));
  *outshapes = static_cast<unsigned**>(malloc (num_out * sizeof(unsigned*)));

  // copy output shapes
  for (int i = 0; i < num_out; i++) {
    (*outdims)[i] = out_shapes[i].size();
    (*outshapes)[i] = static_cast<unsigned*>(malloc ((*outdims)[i] * sizeof(unsigned)));
    for (int j = 0; j < (*outdims)[i]; j++) {
      (*outshapes)[i][j] = out_shapes[i][j];
    }
  }
  return retval;
}

/*! \brief returns status of calling inferType function for operator from library */
MX_INT_RET _opCallInferType(mxnet::ext::inferType_t inferType, const char* const* keys,
                            const char* const* vals, int num,
                            int* intypes, int num_in, int* outtypes, int num_out) {
  // create map of attributes from list
  std::unordered_map<std::string, std::string> attrs;
  for (int i = 0; i < num; i++) {
    attrs[std::string(keys[i])] = std::string(vals[i]);
  }

  // create a vector of types for inputs
  std::vector<int> in_types(num_in);
  for (int i = 0; i < num_in; i++) {
    in_types[i] = intypes[i];
  }

  // create a vector of types for outputs
  std::vector<int> out_types(num_out, -1);

  int retval = inferType(attrs, &in_types, &out_types);
  if (!retval)
    return retval;

  // copy modified input types
  for (int i = 0; i < num_in; i++) {
    intypes[i] = in_types[i];
  }
  // copy output types
  for (int i = 0; i < num_out; i++) {
    outtypes[i] = out_types[i];
  }

  return retval;
}

/*! \brief returns status of calling inferSType function for operator from library */
MX_INT_RET _opCallInferSType(mxnet::ext::inferSType_t inferSType, const char* const* keys,
                             const char* const* vals, int num,
                             int* instypes, int num_in, int* outstypes, int num_out) {
  // create map of attributes from list
  std::unordered_map<std::string, std::string> attrs;
  for (int i = 0; i < num; i++) {
    attrs[std::string(keys[i])] = std::string(vals[i]);
  }

  // create a vector of types for inputs
  std::vector<int> in_stypes(num_in);
  for (int i = 0; i < num_in; i++) {
    in_stypes[i] = instypes[i];
  }

  // create a vector of types for outputs
  std::vector<int> out_stypes(num_out, -1);

  int retval = inferSType(attrs, &in_stypes, &out_stypes);

  if (!retval)
    return retval;

  // copy modified input storage types
  for (int i = 0; i < num_in; i++) {
    instypes[i] = in_stypes[i];
  }
  // copy output storage types
  for (int i = 0; i < num_out; i++) {
    outstypes[i] = out_stypes[i];
  }

  return retval;
}

/*! \brief returns status of calling Forward/Backward function for operator from library */
MX_INT_RET _opCallFCompute(mxnet::ext::fcomp_t fcomp, const char* const* keys,
                           const char* const* vals,
                           int num, const int64_t** inshapes, int* indims, void** indata,
                           int* intypes, size_t* inIDs, const char** indev_type, int* indev_id,
                           int num_in, const int64_t** outshapes, int* outdims, void** outdata,
                           int* outtypes, size_t* outIDs, const char** outdev_type,
                           int* outdev_id, int num_out, mxnet::ext::xpu_malloc_t cpu_malloc,
                           void* cpu_alloc,
                           mxnet::ext::xpu_malloc_t gpu_malloc, void* gpu_alloc,
                           void* cuda_stream,
                           mxnet::ext::sparse_malloc_t sparse_malloc, void* sparse_alloc,
                           int* instypes, int* outstypes, void** in_indices, void** out_indices,
                           void** in_indptr, void** out_indptr,
                           int64_t* in_indices_shapes, int64_t* out_indices_shapes,
                           int64_t* in_indptr_shapes, int64_t* out_indptr_shapes,
                           void* rng_cpu_states, void* rng_gpu_states) {
  // create map of attributes from list
  std::unordered_map<std::string, std::string> attrs;
  for (int i = 0; i < num; i++) {
    attrs[std::string(keys[i])] = std::string(vals[i]);
  }

  // create a vector of tensors for inputs
  std::vector<mxnet::ext::MXTensor> inputs(num_in);
  // create a vector for sparse inputs
  std::vector<mxnet::ext::MXSparse> in_sparse(num_in);

  for (int i = 0; i < num_in; i++) {
    // Dense representation.
    if (instypes[i] == 0) {
      inputs[i].setTensor(indata[i], (mxnet::ext::MXDType)intypes[i], inshapes[i], indims[i],
                          inIDs[i], mxnet::ext::MXContext(indev_type[i], indev_id[i]),
                          mxnet::ext::kDefaultStorage);
    } else {
      // Sparse representation.
      mxnet::ext::MXStorageType type;
      if (instypes[i] == 1) {
        type = mxnet::ext::kRowSparseStorage;
        in_sparse[i].set(indata[i], inshapes[i], indims[i], in_indices[i], in_indices_shapes[i]);
      } else {
        type = mxnet::ext::kCSRStorage;
        in_sparse[i].set(indata[i], inshapes[i], indims[i], in_indices[i],
                         in_indices_shapes[i], in_indptr[i], in_indptr_shapes[i]);
      }
      inputs[i].setTensor(reinterpret_cast<void*>(&in_sparse[i]), (mxnet::ext::MXDType)intypes[i],
                          inshapes[i], indims[i], inIDs[i],
                          mxnet::ext::MXContext(indev_type[i], indev_id[i]), type);
    }
  }

  // create a vector of tensors for outputs
  std::vector<mxnet::ext::MXTensor> outputs(num_out);
  std::vector<mxnet::ext::MXSparse> out_sparse(num_out);

  for (int i = 0; i < num_out; i++) {
    // Dense representation.
    if (outstypes[i] == 0) {
      outputs[i].setTensor(outdata[i], (mxnet::ext::MXDType)outtypes[i], outshapes[i], outdims[i],
                           outIDs[i], mxnet::ext::MXContext(outdev_type[i], outdev_id[i]),
                           mxnet::ext::kDefaultStorage);
    } else {
      // Sparse representation.
      mxnet::ext::MXStorageType type;
      if (outstypes[i] == 1) {
        type = mxnet::ext::kRowSparseStorage;
        out_sparse[i].set(outdata[i], outshapes[i], outdims[i],
                          out_indices[i], out_indices_shapes[i]);
      } else {
        type = mxnet::ext::kCSRStorage;
        out_sparse[i].set(outdata[i], outshapes[i], outdims[i], out_indices[i],
                          out_indices_shapes[i], out_indptr[i], out_indptr_shapes[i]);
      }
      outputs[i].setTensor(reinterpret_cast<void*>(&out_sparse[i]),
                           (mxnet::ext::MXDType)outtypes[i],
                           outshapes[i], outdims[i], outIDs[i],
                           mxnet::ext::MXContext(outdev_type[i], outdev_id[i]), type);
    }
  }

  mxnet::ext::OpResource res(cpu_malloc, cpu_alloc, gpu_malloc, gpu_alloc,
                             cuda_stream, sparse_malloc, sparse_alloc,
                             rng_cpu_states, rng_gpu_states);
  return fcomp(attrs, &inputs, &outputs, res);
}

/*! \brief returns status of calling mutateInputs function for operator from library */
MX_INT_RET _opCallMutateInputs(mxnet::ext::mutateInputs_t mutate, const char* const* keys,
                               const char* const* vals, int num,
                               int** mutate_indices, int* indices_size) {
  // create map of attributes from list
  std::unordered_map<std::string, std::string> attrs;
  for (int i = 0; i < num; i++) {
    attrs[std::string(keys[i])] = std::string(vals[i]);
  }

  // create a vector of mutate input indices
  std::vector<int> mut_ind;

  int retval = mutate(attrs, &mut_ind);
  if (!retval)
    return retval;

  // output the input indices
  *indices_size = mut_ind.size();
  *mutate_indices = static_cast<int*>(malloc (*indices_size * sizeof(int)));
  for (int i = 0; i < *indices_size; i++) {
    (*mutate_indices)[i] = mut_ind[i];
  }

  return retval;
}

/*! \brief returns status of calling createStatefulOp function for operator from library */
MX_INT_RET _opCallCreateOpState(mxnet::ext::createOpState_t create_op, const char* const* keys,
                                     const char* const* vals, int num, const char* dev_type,
                                     int dev_id, unsigned int** inshapes, int* indims,
                                     int num_in, const int* intypes, void** state_op) {
  // create map of attributes from list
  std::unordered_map<std::string, std::string> attrs;
  for (int i = 0; i < num; i++) {
    attrs[std::string(keys[i])] = std::string(vals[i]);
  }

  mxnet::ext::MXContext ctx(dev_type, dev_id);

  // create a vector of shapes for inputs
  std::vector<std::vector<unsigned int> > in_shapes(num_in);
  for (int i = 0; i < num_in; i++) {
    for (int j = 0; j < indims[i]; j++) {
      in_shapes[i].push_back(inshapes[i][j]);
    }
  }

  // create a vector of types for inputs
  std::vector<int> in_types(num_in);
  for (int i = 0; i < num_in; i++) {
    in_types[i] = intypes[i];
  }

  // void pointer to hold custom state op instance created in custom library
  // eventually state_op pointer is populated by instance from custom library
  mxnet::ext::CustomStatefulOp** op_ptr =
    reinterpret_cast<mxnet::ext::CustomStatefulOp**>(state_op);
  return create_op(attrs, ctx, in_shapes, in_types, op_ptr);
}

/*! \brief calls StatefulOp destructor for operator from library */
MX_VOID_RET _opCallDestroyOpState(void* state_op) {
  mxnet::ext::CustomStatefulOp* op_ptr =
    reinterpret_cast<mxnet::ext::CustomStatefulOp*>(state_op);
  delete op_ptr;
}

/*! \brief returns status of calling Stateful Forward/Backward for operator from library */
MX_INT_RET _opCallFStatefulCompute(int is_forward, void* state_op, const int64_t** inshapes,
                                   int* indims, void** indata, int* intypes, size_t* inIDs,
                                   const char** indev_type, int* indev_id, int num_in,
                                   const int64_t** outshapes, int* outdims, void** outdata,
                                   int* outtypes, size_t* outIDs, const char** outdev_type,
                                   int* outdev_id, int num_out,
                                   mxnet::ext::xpu_malloc_t cpu_malloc,
                                   void* cpu_alloc, mxnet::ext::xpu_malloc_t gpu_malloc,
                                   void* gpu_alloc,
                                   void* stream, mxnet::ext::sparse_malloc_t sparse_malloc,
                                   void* sparse_alloc, int* instypes, int* outstypes,
                                   void** in_indices, void** out_indices, void** in_indptr,
                                   void** out_indptr, int64_t* in_indices_shapes,
                                   int64_t* out_indices_shapes, int64_t* in_indptr_shapes,
                                   int64_t* out_indptr_shapes,
                                   void* rng_cpu_states, void* rng_gpu_states) {
  // create a vector of tensors for inputs
  std::vector<mxnet::ext::MXTensor> inputs(num_in);
  // create a vector for sparse inputs
  std::vector<mxnet::ext::MXSparse> in_sparse(num_in);

  for (int i = 0; i < num_in; i++) {
    if (instypes[i] == 0) {
      // Dense representation.
      inputs[i].setTensor(indata[i], (mxnet::ext::MXDType)intypes[i], inshapes[i], indims[i],
                          inIDs[i], mxnet::ext::MXContext(indev_type[i], indev_id[i]),
                          mxnet::ext::kDefaultStorage);
    } else {
      // Sparse representation.
      mxnet::ext::MXStorageType type;
      if (instypes[i] == 1) {
        type = mxnet::ext::kRowSparseStorage;
        in_sparse[i].set(indata[i], inshapes[i], indims[i], in_indices[i], in_indices_shapes[i]);
      } else {
        type = mxnet::ext::kCSRStorage;
        in_sparse[i].set(indata[i], inshapes[i], indims[i], in_indices[i],
                         in_indices_shapes[i], in_indptr[i], in_indptr_shapes[i]);
      }
      inputs[i].setTensor(reinterpret_cast<void*>(&in_sparse[i]), (mxnet::ext::MXDType)intypes[i],
                          inshapes[i], indims[i], inIDs[i],
                          mxnet::ext::MXContext(indev_type[i], indev_id[i]), type);
    }
  }

  // create a vector of tensors for outputs
  std::vector<mxnet::ext::MXTensor> outputs(num_out);
  // create a vector for sparse outputs
  std::vector<mxnet::ext::MXSparse> out_sparse(num_out);

  for (int i = 0; i < num_out; i++) {
    if (outstypes[i] == 0) {
      // Dense representation.
      outputs[i].setTensor(outdata[i], (mxnet::ext::MXDType)outtypes[i], outshapes[i], outdims[i],
                           outIDs[i], mxnet::ext::MXContext(outdev_type[i], outdev_id[i]),
                           mxnet::ext::kDefaultStorage);
    } else {
      // Sparse representation.
      mxnet::ext::MXStorageType type;
      if (outstypes[i] == 1) {
        type = mxnet::ext::kRowSparseStorage;
        out_sparse[i].set(outdata[i], outshapes[i], outdims[i], out_indices[i],
                          out_indices_shapes[i]);
      } else {
        type = mxnet::ext::kCSRStorage;
        out_sparse[i].set(outdata[i], outshapes[i], outdims[i], out_indices[i],
                          out_indices_shapes[i], out_indptr[i], out_indptr_shapes[i]);
      }
      outputs[i].setTensor(reinterpret_cast<void*>(&out_sparse[i]),
                           (mxnet::ext::MXDType)outtypes[i],
                           outshapes[i], outdims[i], outIDs[i],
                           mxnet::ext::MXContext(outdev_type[i], outdev_id[i]), type);
    }
  }

  mxnet::ext::OpResource res(cpu_malloc, cpu_alloc, gpu_malloc, gpu_alloc,
                             stream, sparse_malloc, sparse_alloc, rng_cpu_states, rng_gpu_states);

  mxnet::ext::CustomStatefulOp* op_ptr =
    reinterpret_cast<mxnet::ext::CustomStatefulOp*>(state_op);
  if (is_forward) {
    return op_ptr->Forward(&inputs, &outputs, res);
  }
  return op_ptr->Backward(&inputs, &outputs, res);
}

/*! \brief returns number of partitioners registered in this library */
MX_INT_RET _partRegSize() {
  return mxnet::ext::Registry<mxnet::ext::CustomPartitioner>::get()->size();
}

/* returns number of strategies registered for partitioner
 * at specified index */
MX_INT_RET _partRegGetCount(int idx, const char** name) {
  mxnet::ext::CustomPartitioner part =
    mxnet::ext::Registry<mxnet::ext::CustomPartitioner>::get()->get(idx);
  *name = part.name;
  return part.strategies.size();
}

/*! \brief returns partitioner registration at specified index */
MX_VOID_RET _partRegGet(int part_idx, int stg_idx, const char** strategy,
                        mxnet::ext::supportedOps_t* supportedOps,
                        mxnet::ext::createSelector_t* createSelector,
                        mxnet::ext::reviewSubgraph_t* reviewSubgraph, const char** op_name) {
  mxnet::ext::CustomPartitioner part =
    mxnet::ext::Registry<mxnet::ext::CustomPartitioner>::get()->get(part_idx);
  *strategy = part.strategies[stg_idx];
  *op_name = part.op_names[stg_idx];
  *supportedOps = part.getSupportedOps(stg_idx);
  *createSelector = part.getCreateSelector(stg_idx);
  *reviewSubgraph = part.getReviewSubgraph(stg_idx);
}

/*! \brief returns status of calling supported ops function from library */
MX_INT_RET _partCallSupportedOps(mxnet::ext::supportedOps_t supportedOps, const char *json,
                                 int num_ids, int *ids, const char* const* opt_keys,
                                 const char* const* opt_vals, int num_opts) {
  mxnet::ext::Graph *graph = mxnet::ext::Graph::fromString(json);
  // create map of options from list
  std::unordered_map<std::string, std::string> opts;
  for (int i = 0; i < num_opts; i++)
    opts[std::string(opt_keys[i])] = std::string(opt_vals[i]);

  // create array of subgraph IDs for operator support
  std::vector<int> _ids(num_ids, -2);
  // call user's supportedOps function
  mxnet::ext::MXReturnValue retval = supportedOps(graph, &_ids, opts);
  if (!retval) return retval;

  // copy bools in ids to ints
  for (int i = 0; i < num_ids; i++)
    ids[i] = _ids[i];

  return retval;
}

/*! \brief returns status of calling create selector function from library */
MX_INT_RET _partCallCreateSelector(mxnet::ext::createSelector_t createSelector, const char *json,
                                   void** selector, const char* const* opt_keys,
                                   const char* const* opt_vals, int num_opts) {
  mxnet::ext::Graph *graph = mxnet::ext::Graph::fromString(json);
  // create map of options from list
  std::unordered_map<std::string, std::string> opts;
  for (int i = 0; i < num_opts; i++)
    opts[std::string(opt_keys[i])] = std::string(opt_vals[i]);

  // void pointer to hold selector instance created in custom library
  // eventually pointer is populated by instance from custom library
  mxnet::ext::CustomOpSelector** sel_ptr =
    reinterpret_cast<mxnet::ext::CustomOpSelector**>(selector);

  // call user's createSelector function
  return createSelector(graph, sel_ptr, opts);
}

/*! \brief returns status of calling select function from library */
MX_VOID_RET _partCallSelect(void* sel_inst, int nodeID, int* selected) {
  mxnet::ext::CustomOpSelector* sel_ptr =
    reinterpret_cast<mxnet::ext::CustomOpSelector*>(sel_inst);
  *selected = sel_ptr->Select(nodeID);
}

/*! \brief returns status of calling select input function from library */
MX_VOID_RET _partCallSelectInput(void* sel_inst, int nodeID,
                                 int input_nodeID, int* selected) {
  mxnet::ext::CustomOpSelector* sel_ptr =
    reinterpret_cast<mxnet::ext::CustomOpSelector*>(sel_inst);
  *selected = sel_ptr->SelectInput(nodeID, input_nodeID);
}

/*! \brief returns status of calling select output function from library */
MX_VOID_RET _partCallSelectOutput(void* sel_inst, int nodeID,
                                  int output_nodeID, int* selected) {
  mxnet::ext::CustomOpSelector* sel_ptr =
    reinterpret_cast<mxnet::ext::CustomOpSelector*>(sel_inst);
  *selected = sel_ptr->SelectOutput(nodeID, output_nodeID);
}

/*! \brief returns status of calling filter function from library */
MX_VOID_RET _partCallFilter(void* sel_inst, int* candidates, int num_candidates,
                            int** keep, int* num_keep) {
  mxnet::ext::CustomOpSelector* sel_ptr =
    reinterpret_cast<mxnet::ext::CustomOpSelector*>(sel_inst);
  std::vector<int> candidates_(num_candidates);
  for (int i=0; i < num_candidates; i++) {
    candidates_[i] = candidates[i];
  }
  std::vector<int> keep_;

  sel_ptr->Filter(candidates_, &keep_);

  *num_keep = keep_.size();
  *keep = static_cast<int*>(malloc(keep_.size() * sizeof(int)));
  for (unsigned i=0; i < keep_.size(); i++)
    (*keep)[i] = keep_[i];
}

/*! \brief returns status of calling reset selector function from library */
MX_VOID_RET _partCallReset(void* sel_inst) {
  mxnet::ext::CustomOpSelector* sel_ptr =
    reinterpret_cast<mxnet::ext::CustomOpSelector*>(sel_inst);
    sel_ptr->Reset();
}

/*! \brief returns status of calling review subgraph function from library */
MX_INT_RET _partCallReviewSubgraph(mxnet::ext::reviewSubgraph_t reviewSubgraph, const char *json,
                                   int subgraph_id, int *accept, const char* const* opt_keys,
                                   const char* const* opt_vals, int num_opts,
                                   char*** attr_keys, char*** attr_vals, int *num_attrs,
                                   const char* const* arg_names, int num_args,
                                   void* const* arg_data, const int64_t* const* arg_shapes,
                                   const int* arg_dims, const int* arg_types,
                                   const size_t* arg_IDs, const char* const* arg_dev_type,
                                   const int* arg_dev_id,
                                   const char* const* aux_names, int num_aux,
                                   void* const* aux_data, const int64_t* const* aux_shapes,
                                   const int* aux_dims, const int* aux_types,
                                   const size_t* aux_IDs, const char* const* aux_dev_type,
                                   const int* aux_dev_id) {
  mxnet::ext::Graph *subgraph = mxnet::ext::Graph::fromString(json);
  bool accept_bool = false;
  // create map of attributes from list
  std::unordered_map<std::string, std::string> opts;
  for (int i = 0; i < num_opts; i++)
    opts[std::string(opt_keys[i])] = std::string(opt_vals[i]);

  // create a map of named tensors for args
  std::unordered_map<std::string, mxnet::ext::MXTensor> args;
  for (int i = 0; i < num_args; i++) {
    std::vector<int64_t> shapes;
    shapes.reserve(arg_dims[i]);
    for (int j = 0; j < arg_dims[i]; j++)
      shapes.push_back(arg_shapes[i][j]);

    mxnet::ext::MXTensor tensor(arg_data[i], shapes, (mxnet::ext::MXDType)arg_types[i],
                                arg_IDs[i], mxnet::ext::MXContext(arg_dev_type[i], arg_dev_id[i]));
    args[arg_names[i]] = tensor;
  }
  // create a map of named tensors for aux
  std::unordered_map<std::string, mxnet::ext::MXTensor> aux;
  for (int i = 0; i < num_aux; i++) {
    std::vector<int64_t> shapes;
    shapes.reserve(aux_dims[i]);
    for (int j = 0; j < aux_dims[i]; j++)
      shapes.push_back(aux_shapes[i][j]);

    mxnet::ext::MXTensor tensor(aux_data[i], shapes, (mxnet::ext::MXDType)aux_types[i],
                                aux_IDs[i], mxnet::ext::MXContext(aux_dev_type[i],
                                                                  aux_dev_id[i]));
    aux[aux_names[i]] = tensor;
  }

  subgraph->_setParams(&args, &aux);

  std::unordered_map<std::string, std::string> attrs;
  mxnet::ext::MXReturnValue retval = reviewSubgraph(subgraph, subgraph_id, &accept_bool,
                                                    opts, &attrs);
  if (!retval) return retval;

  *accept = accept_bool;

  if (attrs.size() > 0) {
    *num_attrs = attrs.size();
    // allocate space for attributes
    *attr_keys = static_cast<char**>(malloc (*num_attrs * sizeof(char*)));
    *attr_vals = static_cast<char**>(malloc (*num_attrs * sizeof(char*)));

    // copy attributes
    int i = 0;
    for (auto kv : attrs) {
      (*attr_keys)[i] = static_cast<char*>(malloc ((kv.first.size()+1) * sizeof(char)));  // NOLINT
      (*attr_vals)[i] = static_cast<char*>(malloc ((kv.second.size()+1) * sizeof(char)));  // NOLINT
      snprintf((*attr_keys)[i], kv.first.size()+1, "%s", kv.first.c_str());
      snprintf((*attr_vals)[i], kv.second.size()+1, "%s", kv.second.c_str());
      i++;
    }
  }

  return retval;
}

/*! \brief returns number of graph passes registered in this library */
MX_INT_RET _passRegSize() {
  return mxnet::ext::Registry<mxnet::ext::CustomPass>::get()->size();
}

/*! \brief returns pass registration at specified index */
MX_VOID_RET _passRegGet(int pass_idx, mxnet::ext::graphPass_t* graphPass,
                        const char** pass_name) {
  mxnet::ext::CustomPass pass =
    mxnet::ext::Registry<mxnet::ext::CustomPass>::get()->get(pass_idx);
  *graphPass = pass.pass;
  *pass_name = pass.name;
}

/*! \brief returns status of calling graph pass function from library */
MX_INT_RET _passCallGraphPass(mxnet::ext::graphPass_t graphPass, const char *json,
                              char** out_graph, const char* const* opt_keys,
                              const char* const* opt_vals, int num_opts,
                              const char* pass_name, const char* const* arg_names, int num_args,
                              void* const* arg_data, const int64_t* const* arg_shapes,
                              const int* arg_dims, const int* arg_types,
                              const size_t* arg_IDs, const char* const* arg_dev_type,
                              const int* arg_dev_id, const char* const* aux_names, int num_aux,
                              void* const* aux_data, const int64_t* const* aux_shapes,
                              const int* aux_dims, const int* aux_types,
                              const size_t* aux_IDs, const char* const* aux_dev_type,
                              const int* aux_dev_id, mxnet::ext::nd_malloc_t nd_malloc,
                              const void* nd_alloc) {
  mxnet::ext::Graph *graph = mxnet::ext::Graph::fromString(json);
  // create map of attributes from list
  std::unordered_map<std::string, std::string> opts;
  for (int i = 0; i < num_opts; i++)
    opts[std::string(opt_keys[i])] = std::string(opt_vals[i]);

  // create a map of named tensors for args
  std::unordered_map<std::string, mxnet::ext::MXTensor> args;
  for (int i = 0; i < num_args; i++) {
    std::vector<int64_t> shapes;
    shapes.reserve(arg_dims[i]);
    for (int j = 0; j < arg_dims[i]; j++)
      shapes.push_back(arg_shapes[i][j]);

    mxnet::ext::MXTensor tensor(arg_data[i], shapes, (mxnet::ext::MXDType)arg_types[i],
                                arg_IDs[i], mxnet::ext::MXContext(arg_dev_type[i],
                                                                  arg_dev_id[i]));
    args[arg_names[i]] = tensor;
  }
  // create a map of named tensors for aux
  std::unordered_map<std::string, mxnet::ext::MXTensor> aux;
  for (int i = 0; i < num_aux; i++) {
    std::vector<int64_t> shapes;
    shapes.reserve(aux_dims[i]);
    for (int j = 0; j < aux_dims[i]; j++)
      shapes.push_back(aux_shapes[i][j]);

    mxnet::ext::MXTensor tensor(aux_data[i], shapes, (mxnet::ext::MXDType)aux_types[i],
                                aux_IDs[i], mxnet::ext::MXContext(aux_dev_type[i],
                                                                  aux_dev_id[i]));
    aux[aux_names[i]] = tensor;
  }

  std::unordered_map<std::string, mxnet::ext::MXTensor> new_args, new_aux;
  mxnet::ext::PassResource res(&new_args, &new_aux, nd_malloc, nd_alloc);
  graph->_setParams(&args, &aux);
  graph->_setPassResource(&res);
  mxnet::ext::MXReturnValue retval = graphPass(graph, opts);
  if (!retval) return retval;

  std::string tmp = graph->toString();
  *out_graph = static_cast<char*>(malloc ((tmp.size()+1) * sizeof(char)));  // NOLINT
  snprintf((*out_graph), tmp.size()+1, "%s", tmp.c_str());
  return retval;
}

/*!
 * \brief Checks if the MXNet version is supported by the library.
 * If supported, initializes the library.
 * \param version MXNet version number passed to library and defined as:
 *                MXNET_VERSION = (MXNET_MAJOR*10000 + MXNET_MINOR*100 + MXNET_PATCH)
 * \return Non-zero value on error i.e. library incompatible with passed MXNet version
 */
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
__declspec(dllexport) mxnet::ext::MXReturnValue __cdecl
#else
mxnet::ext::MXReturnValue
#endif
initialize(int version);

MX_INT_RET _msgSize() {
  return mxnet::ext::MXerrorMsgs::get()->size();
}

/*! \brief returns operator registration at specified index */
MX_VOID_RET _msgGet(int idx, const char** msg) {
  *msg = mxnet::ext::MXerrorMsgs::get()->get(idx)->c_str();
}
