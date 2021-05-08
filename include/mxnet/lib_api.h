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
 * \file lib_api.h
 * \brief APIs to interact with libraries
 * This API specifies function prototypes to
 * register custom ops, partitioner, and passes
 * for library authors
 * See example/extension/lib_custom_op/README.md
 * See example/extension/lib_subgraph/README.md
 * See example/extension/lib_pass/README.md
 */

#ifndef MXNET_LIB_API_H_
#define MXNET_LIB_API_H_

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <iostream>
#include <utility>
#include <stdexcept>
#include <functional>
#include <random>
#include <sstream>

#if defined(__NVCC__)
  #include <cuda_runtime.h>
  #include <curand_kernel.h>
#endif

/* Make sure to update the version number everytime you make changes */
#define MX_LIBRARY_VERSION 11

/*!
 * \brief For loading multiple custom op libraries in Linux, exporting same symbol multiple
 * times may lead to undefined behaviour, so we need to set symbol visibility to hidden
 * see https://labjack.com/news/simple-cpp-symbol-visibility-demo for details
 */
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  #define PRIVATE_SYMBOL
#else
  #define PRIVATE_SYMBOL  __attribute__ ((visibility ("hidden")))
#endif

/*
 * Import from DLPack https://github.com/dmlc/dlpack/blob/master/include/dlpack/dlpack.h
 */
#ifndef DLPACK_VERSION
#ifdef __cplusplus
#define DLPACK_EXTERN_C extern "C"
#else
#define DLPACK_EXTERN_C
#endif

/*! \brief The current version of dlpack */
#define DLPACK_VERSION 020

/*! \brief DLPACK_DLL prefix for windows */
#ifdef _WIN32
#ifdef DLPACK_EXPORTS
#define DLPACK_DLL __declspec(dllexport)
#else
#define DLPACK_DLL __declspec(dllimport)
#endif
#else
#define DLPACK_DLL
#endif

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
  #endif
  /*!
   * \brief The device type in DLContext.
   */
  typedef enum {
    /*! \brief CPU device */
    kDLCPU = 1,
    /*! \brief CUDA GPU device */
    kDLGPU = 2,
    /*!
     * \brief Pinned CUDA GPU device by cudaMallocHost
     * \note kDLCPUPinned = kDLCPU | kDLGPU
     */
    kDLCPUPinned = 3,
    /*! \brief OpenCL devices. */
    kDLOpenCL = 4,
    /*! \brief Vulkan buffer for next generation graphics. */
    kDLVulkan = 7,
    /*! \brief Metal for Apple GPU. */
    kDLMetal = 8,
    /*! \brief Verilog simulator buffer */
    kDLVPI = 9,
    /*! \brief ROCm GPUs for AMD GPUs */
    kDLROCM = 10,
    /*!
     * \brief Reserved extension device type,
     * used for quickly test extension device
     * The semantics can differ depending on the implementation.
     */
    kDLExtDev = 12,
  } DLDeviceType;

  /*!
   * \brief A Device context for Tensor and operator.
   */
  typedef struct {
    /*! \brief The device type used in the device. */
    DLDeviceType device_type;
    /*! \brief The device index */
    int device_id;
  } DLContext;

  /*!
   * \brief The type code options DLDataType.
   */
  typedef enum {
    kDLInt = 0U,
    kDLUInt = 1U,
    kDLFloat = 2U,
  } DLDataTypeCode;

  /*!
   * \brief The data type the tensor can hold.
   *
   *  Examples
   *   - float: type_code = 2, bits = 32, lanes=1
   *   - float4(vectorized 4 float): type_code = 2, bits = 32, lanes=4
   *   - int8: type_code = 0, bits = 8, lanes=1
   */
  typedef struct {
    /*!
     * \brief Type code of base types.
     * We keep it uint8_t instead of DLDataTypeCode for minimal memory
     * footprint, but the value should be one of DLDataTypeCode enum values.
     * */
    uint8_t code;
    /*!
     * \brief Number of bits, common choices are 8, 16, 32.
     */
    uint8_t bits;
    /*! \brief Number of lanes in the type, used for vector types. */
    uint16_t lanes;
  } DLDataType;

  /*!
   * \brief Plain C Tensor object, does not manage memory.
   */
  typedef struct {
    /*!
     * \brief The opaque data pointer points to the allocated data. This will be
     * CUDA device pointer or cl_mem handle in OpenCL. This pointer is always
     * aligns to 256 bytes as in CUDA.
     *
     * For given DLTensor, the size of memory required to store the contents of
     * data is calculated as follows:
     *
     * \code{.c}
     * static inline size_t GetDataSize(const DLTensor* t) {
     *   size_t size = 1;
     *   for (tvm_index_t i = 0; i < t->ndim; ++i) {
     *     size *= t->shape[i];
     *   }
     *   size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
     *   return size;
     * }
     * \endcode
     */
    void* data;
    /*! \brief The device context of the tensor */
    DLContext ctx;
    /*! \brief Number of dimensions */
    int ndim;
    /*! \brief The data type of the pointer*/
    DLDataType dtype;
    /*! \brief The shape of the tensor */
    int64_t* shape;
    /*!
     * \brief strides of the tensor (in number of elements, not bytes)
     *  can be nullptr, indicating tensor is compact and row-majored.
     */
    int64_t* strides;
    /*! \brief The offset in bytes to the beginning pointer to data */
    uint64_t byte_offset;
  } DLTensor;
#ifdef __cplusplus
}  // DLPACK_EXTERN_C
#endif
#endif

namespace mxnet {
namespace ext {

/* \brief Class to store error messages from extensions to pass to MXNet */
class MXerrorMsgs {
 public:
  /* \brief get singleton pointer to class */
  static MXerrorMsgs* get();

  /* \brief add a new error message */
  std::stringstream& add(const char* file, int line);

  /* \brief return number of error messages */
  int size();

  /* \brief get error message at index */
  const std::string* get(int idx);

 private:
  /*! \brief constructor */
  MXerrorMsgs() {}
  /*! \brief destructor */
  ~MXerrorMsgs() {}
  /*! \brief map of entries in registry */
  std::vector<std::stringstream> messages;
};

// Add a new error message, example: MX_ERROR_MSG << "my error msg";
#define MX_ERROR_MSG mxnet::ext::MXerrorMsgs::get()->add(__FILE__, __LINE__)

/*!
 * \brief Tensor data type, consistent with mshadow data type
 */
enum MXDType {
  kFloat32 = 0,
  kFloat64 = 1,
  kFloat16 = 2,
  kUint8 = 3,
  kInt32 = 4,
  kInt8  = 5,
  kInt64 = 6,
  kUint16 = 7,
  kInt16 = 8,
  kUNSET = 100,
};

/*
 * MXTensor storage type.
 */
enum MXStorageType {
  // dense
  kDefaultStorage = 0,
  // row sparse
  kRowSparseStorage = 1,
  // csr
  kCSRStorage = 2,
};

/*!
 * \brief Context info passing from MXNet OpContext
 * dev_type is string repr of supported context, currently only "cpu" and "gpu"
 * dev_id is the device index where the tensor locates
 */
struct MXContext {
  MXContext();
  explicit MXContext(std::string dev_type_, int dev_id_);
  explicit MXContext(const char* dev_type_, int dev_id_);
  static MXContext CPU();
  static MXContext GPU();
  static MXContext CPU(int dev_id);
  static MXContext GPU(int dev_id);

  std::string dev_type;
  int dev_id;
};

enum MXReturnValue {
  MX_FAIL = 0,
  MX_SUCCESS = 1,
};

// For sparse tensors, read/write the data from NDarray via pointers.
struct MXSparse {
  // Pointer to data.
  void *data{nullptr};
  // length of (non-zero) data.
  int64_t data_len;

  // To store aux data for sparse.
  // For CSR, indices stores the col index of non-zero elements.
  // For row sparse, indices store row index of rows which have non-zero elements.
  int64_t* indices;
  int64_t indices_len;

  // For CSR, indptr gives the start and end index of data for each row.
  // For row sparse, indptr is not used.
  int64_t* indptr = nullptr;
  int64_t indptr_len;

  void set(void *data_ptr, const int64_t* dims, int ndims, void *idx,
           int64_t num_idx, void *idx_ptr = nullptr, int64_t num_idx_ptr = 0);
};

/*!
 * \brief Tensor data structure used by custom operator
 */
struct MXTensor {
  MXTensor();
  MXTensor(const MXTensor& oth);
  MXTensor(void *data_ptr, std::vector<int64_t> shape, MXDType dtype,
           size_t vID, MXContext mx_ctx, MXStorageType stype = kDefaultStorage);

  /*! \brief populate internal tensor fields */
  void setTensor(void *dptr, MXDType type, const int64_t* dims, int ndims,
                 size_t vID, MXContext mx_ctx, MXStorageType storage_type);

  /*! \brief populate DLTensor fields */
  void setDLTensor();

  /*! \brief helper function to cast data pointer */
  template<typename data_type>
  inline data_type* data() {
    return reinterpret_cast<data_type*>(data_ptr);
  }

  /*! \brief helper function to get data size */
  int64_t size() const;

  /*! \brief helper function to compare two MXTensors */
  bool isSame(const MXTensor &oth) const;

  // For dense, data_ptr points to 1D flattened tensor data
  // For sparse, data_ptr points to MXSparse
  void *data_ptr;

  // shape is in [2,3,4] format to represent high-dim tensor
  std::vector<int64_t> shape;

  // type can only be MXDType enum types
  MXDType dtype;

  // version number updated if the tensor has changed since the last use by custom op
  size_t verID;

  // context of MXTensor representing which device the tensor data is located
  MXContext ctx;

  // corresponding DLTensor repr of MXTensor
  // easy way to reuse functions taking DLTensor
  DLTensor dltensor;

  // storage type
  MXStorageType stype;
};

/*! \brief resource malloc function to allocate memory inside Forward/Backward functions */
typedef void* (*xpu_malloc_t)(void*, int);
/*! \brief sparse alloc function to allocate memory inside Forward/Backward functions */
typedef void (*sparse_malloc_t)(void*, int, int, int, void**, int64_t**, int64_t**);
/*! \brief resource malloc function to allocate ndarrays for graph passes */
typedef void (*nd_malloc_t)(const void* _ndarray_alloc, const int64_t* shapes, int num_shapes,
                            const char* dev_str, int dev_id, int dtype, const char* name,
                            int isArg, void** data);
/*! \brief GPU stream pointer, is void* when not compiled with CUDA */
#if defined(__NVCC__)
  typedef cudaStream_t mx_stream_t;
  typedef curandStatePhilox4_32_10_t mx_gpu_rand_t;
#else
  typedef void* mx_stream_t;
  typedef void* mx_gpu_rand_t;
#endif
typedef std::mt19937 mx_cpu_rand_t;

/*! \brief MXNet initialized random states for each device, used for parallelism */
/* Each thread should generate random number unique sequence out of different states */
#define MX_NUM_CPU_RANDOM_STATES 1024
#define MX_NUM_GPU_RANDOM_STATES 32768

/* \brief Class to help allocate new args/aux params in graph passes */
class PassResource {
 public:
  PassResource(std::unordered_map<std::string, MXTensor>* new_args,
               std::unordered_map<std::string, MXTensor>* new_aux,
               nd_malloc_t nd_malloc, const void* nd_alloc);

  // allocate new arg param, adds to args map, returns newly allocated tensor
  MXTensor* alloc_arg(const std::string& name, const std::vector<int64_t>& shapes,
                      const MXContext &ctx, MXDType dtype) const;

  // allocate new aux param, adds to aux map, returns newly allocated tensor
  MXTensor* alloc_aux(const std::string& name, const std::vector<int64_t>& shapes,
                      const MXContext &ctx, MXDType dtype) const;

 private:
  std::unordered_map<std::string, MXTensor>* new_args_;
  std::unordered_map<std::string, MXTensor>* new_aux_;
  nd_malloc_t nd_malloc_;
  const void* nd_alloc_;
};

/*!
 * \brief provide resource APIs memory allocation mechanism to Forward/Backward functions
 */
class OpResource {
 public:
  OpResource(xpu_malloc_t cpu_malloc_fp, void* cpu_alloc_fp,
             xpu_malloc_t gpu_malloc_fp, void* gpu_alloc_fp, void* stream,
             sparse_malloc_t sparse_malloc_fp, void* sparse_alloc_fp,
             void* rng_cpu_states, void* rng_gpu_states);

  /*! \brief allocate cpu memory controlled by MXNet */
  void* alloc_cpu(int size) const;

  /*! \brief allocate gpu memory controlled by MXNet */
  void* alloc_gpu(int size) const;

  /*! \brief return the cuda stream object with correct type */
  inline mx_stream_t get_cuda_stream() const {
    return static_cast<mx_stream_t>(cuda_stream);
  }

  /*! \brief allocate sparse memory controlled by MXNet */
  void alloc_sparse(MXSparse* sparse, int index, int indices_len, int indptr_len = 0) const;

  /*! \brief get pointer to initialized and seeded random number states located on CPU */
  /* Access each state by states[id], but this id should be <= MX_NUM_CPU_RANDOM_STATES */
  mx_cpu_rand_t* get_cpu_rand_states() const;

  /*! \brief get pointer to initialized and seeded random number states located on GPU */
  /* Access each state by states[id], but this id should be <= MX_NUM_GPU_RANDOM_STATES */
  /* Note that if you are using cpu build, it will return a nullptr */
  inline mx_gpu_rand_t* get_gpu_rand_states() const {
    return static_cast<mx_gpu_rand_t*>(rand_gpu_states);
  }

 private:
  /*! \brief allocation lambda function */
  xpu_malloc_t cpu_malloc, gpu_malloc;
  /*! \brief lambda function to return allocated memory handle */
  void *cpu_alloc, *gpu_alloc;
  /*! \brief cuda stream passed from MXNet */
  void *cuda_stream;
  /*! \brief sparse allocation lambda function */
  sparse_malloc_t sparse_malloc;
  /*! \brief lambda function to return allocated sparse memory handle */
  void *sparse_alloc;
  /*! \brief cpu and gpu rng fully inited and seeded states */
  void *rand_cpu_states, *rand_gpu_states;
};

/*! \brief attribute key to help passing serialized subgraph through subgraph op attribute */
#define MX_STR_SUBGRAPH_SYM_JSON "subgraph_sym_json"
/*! \brief dtype attribute key for ops after type propagation */
#define MX_STR_DTYPE "__ext_dtype__"
/*! \brief shape attribute key for ops after shape propagation */
#define MX_STR_SHAPE "__ext_shape__"
/*! \brief extra input attribute key for ops */
#define MX_STR_EXTRA_INPUTS "__ext_extra_inputs__"

/* \brief get shape value from list of shapes string
 *
 * Examples:
 *
 * getShapeAt("[[1]]", 0) returns "[1]"
 * getShapeAt("[[1],[2,3]]", 1) returns "[2,3]"
 */
std::string getShapeAt(const std::string& shape, unsigned index);

/* \brief get dtype value from list of dtypes string
 *
 * Examples:
 *
 * getDtypeAt("[1]", 0) returns "1"
 * getDtypeAt("[1,2]", 1) returns "2" 
 */
std::string getDtypeAt(const std::string& dtype, unsigned index);

/*!
 * \brief Json utility to parse serialized subgraph symbol
 */
/*! \brief Types of JSON objects */
enum JsonType {ERR, STR, NUM, LIST, MAP};

/*! \brief definition of JSON objects */
struct JsonVal {
  JsonVal();  // default constructor
  // construct a JSON object by type
  explicit JsonVal(JsonType t);
  // construct a string JSON object
  explicit JsonVal(std::string s);
  // construct a number JSON object
  explicit JsonVal(int n);
  // complex constructor
  JsonVal(JsonType t, int n, std::string s);
  bool operator<(const JsonVal &o) const;

  // convert JSON object back to JSON-compatible string
  std::string dump() const;

  // convert JSON-compatible string to JSON object
  static JsonVal parse(const std::string& json);

  // parse a string JSON object
  static JsonVal parse_string(const std::string& json, unsigned int* idx);

  // parse a number JSON object
  static JsonVal parse_num(const std::string& json, unsigned int* idx);

  // parse a list of JSON objects
  static JsonVal parse_list(const std::string& json, unsigned int* idx);

  // parse a map of JSON objects
  static JsonVal parse_map(const std::string& json, unsigned int* idx);

  // generic parse function
  static JsonVal parse(const std::string& json, unsigned int *idx);

  // debug function to convert data structure to a debugstring
  std::string toString() const;

  JsonType type;
  int num;
  std::string str;
  std::vector<JsonVal> list;
  std::map<JsonVal, JsonVal> map;
};

/*!
 * \brief Graph utility to parse serialized subgraph symbol
 */
class Node;
class Graph;

// Representation of an input/output to a node
struct NodeEntry {
  Node* node;  // other node thats producing/consuming inputs/outputs
  int entry;  // entry index from other node (ie. output index from producing node)
};

// Representation of a node in the graph
class Node {
 public:
  Node();

  // internally set passResource to enable tensor allocation for graph passes
  void _setPassResource(PassResource* res_);

  /* \brief allocate an arg tensor for this node */
  void alloc_arg(const std::vector<int64_t>& shapes,
                 const MXContext &ctx, MXDType dtype);

  /* \brief allocate an aux tensor for this node */
  void alloc_aux(const std::vector<int64_t>& shapes,
                 const MXContext &ctx, MXDType dtype);

  std::string op;  // operator name (ie. Convolution)
  std::string name;  // unique node name (ie. conv_0 or conv_1)
  MXTensor* tensor;  // tensor data for input nodes
  std::vector<NodeEntry> inputs;  // set of inputs to the node
  std::vector<NodeEntry> outputs;  // set of outputs from the node
  std::vector<Graph*> subgraphs;  // set of subgraphs within this node
  std::unordered_map<std::string, std::string> attrs;  // node attributes

 private:
  PassResource* res;
};

// Representation of the graph
class Graph {
 public:
  Graph();

  /* \brief deleted nodes when deleting the graph */
  ~Graph();

  /* \brief create a graph object from an unparsed string */
  static Graph* fromString(const std::string& json);

  /* \brief create a graph object from a parsed JSON object */
  static Graph* fromJson(JsonVal val);

  /* \brief convert graph object back to JSON object */
  JsonVal toJson() const;

  /* \brief convert graph object to JSON string */
  std::string toString() const;

  /* \brief visits a node "n" */
  void _dfs_util(Node* n, std::unordered_set<Node*>* to_visit,
                 std::function<void(Node*)> handler) const;

  /* \brief post-order DFS graph traversal */
  void DFS(std::function<void(Node*)> handler) const;

  /* \brief sort graph nodes in topological order */
  std::vector<Node*> topological_sort() const;

  /* \brief print out graph details */
  void print(int indent = 0) const;

  /* \brief add a new node to this graph */
  Node* addNode(const std::string& name, const std::string& op);

  /* \brief get node at index in graph */
  Node* getNode(size_t idx);

  /* \brief get const node at index in const graph */
  const Node* getNode(size_t idx) const;

  /* \brief get attribute on graph */
  const JsonVal& getAttr(const std::string& key) const;

  /* \brief get number of nodes in the graph */
  size_t size() const;

  // internally set passResource to enable tensor allocation for graph passes
  void _setPassResource(PassResource* res_);

  // internally set arg/aux params when available
  void _setParams(std::unordered_map<std::string, mxnet::ext::MXTensor>* args,
                  std::unordered_map<std::string, mxnet::ext::MXTensor>* aux);

  std::vector<Node*> inputs;
  std::vector<NodeEntry> outputs;
  std::map<std::string, JsonVal> attrs;

 private:
  std::vector<Node*> nodes;
  PassResource* res;
};

/* \brief An abstract class for library authors creating custom
 * partitioners. Optional, can just implement supportedOps instead
 */
class CustomOpSelector {
 public:
  /* \brief Select a node to include in subgraph, return true to include node
   * nodeID - index of node in graph
   */
  virtual bool Select(int nodeID) = 0;
  /* \brief Select an input node from current node to include in subgraph
   * return true to include node
   * nodeID - index of node in graph
   * input_nodeID - index of input node in graph
   */
  virtual bool SelectInput(int nodeID, int input_nodeID) = 0;
  /* \brief Select an output node from current node to include in subgraph
   * return true to include node
   * nodeID - index of node in graph
   * output_nodeID - index of output node in graph
   */
  virtual bool SelectOutput(int nodeID, int output_nodeID) = 0;
  /* \brief Review nodes to include in subgraph
   * return set of candidate nodes to keep in subgraph
   * candidates - indices of nodes to include in subgraph
   * keep - indices of nodes to keep in subgraph
   */
  virtual void Filter(const std::vector<int>& candidates,
                      std::vector<int>* keep) {
    keep->insert(keep->end(), candidates.begin(), candidates.end());
  }
  /* \brief Reset any selector state, called after growing subgraph, before filter
   * Called after finished calling SelectInput/SelectOutput and growing subgraph
   */
  virtual void Reset() {}
};

/*!
 * \brief An abstract class for library authors creating stateful op
 * custom library should override Forward and destructor, and has an
 * option to implement Backward
 */
class CustomStatefulOp {
 public:
  CustomStatefulOp();
  virtual ~CustomStatefulOp();

  template<class A, typename ...Ts>
  static CustomStatefulOp* create(Ts...args) {
    CustomStatefulOp* op = new A(args...);
    op->created = true;
    return op;
  }

  bool wasCreated() { return created; }

  virtual MXReturnValue Forward(std::vector<MXTensor>* inputs,
                                std::vector<MXTensor>* outputs,
                                const OpResource& op_res) = 0;
  virtual MXReturnValue Backward(std::vector<MXTensor>* inputs,
                                 std::vector<MXTensor>* outputs,
                                 const OpResource& op_res) {
    MX_ERROR_MSG << "Error! Operator does not support backward" << std::endl;
    return MX_FAIL;
  }

  bool ignore_warn;

 private:
  bool created;
};

/*! \brief Custom Operator function templates */
typedef MXReturnValue (*fcomp_t)(const std::unordered_map<std::string,
                                 std::string>& attributes,
                                 std::vector<MXTensor>* inputs,
                                 std::vector<MXTensor>* outputs,
                                 const OpResource& res);
typedef MXReturnValue (*parseAttrs_t)(const std::unordered_map<std::string,
                                      std::string>& attributes,
                                      int* num_inputs, int* num_outputs);
typedef MXReturnValue (*inferType_t)(const std::unordered_map<std::string,
                                     std::string>& attributes,
                                     std::vector<int>* in_types,
                                     std::vector<int>* out_types);
typedef MXReturnValue (*inferSType_t)(const std::unordered_map<std::string,
                                      std::string>& attributes,
                                      std::vector<int>* in_storage_types,
                                      std::vector<int>* out_storage_types);
typedef MXReturnValue (*inferShape_t)(const std::unordered_map<std::string,
                                      std::string>& attributes,
                                      std::vector<std::vector<unsigned int> >* in_shapes,
                                      std::vector<std::vector<unsigned int> >* out_shapes);
typedef MXReturnValue (*mutateInputs_t)(const std::unordered_map<std::string,
                                        std::string>& attributes,
                                        std::vector<int>* input_indices);
typedef MXReturnValue (*createOpState_t)(const std::unordered_map<std::string,
                                         std::string>& attributes,
                                         const MXContext& ctx,
                                         const std::vector<std::vector<unsigned int> >& in_shapes,
                                         const std::vector<int> in_types,
                                         CustomStatefulOp**);

/*!
 * \brief Class to hold custom operator registration
 */
class CustomOp {
 public:
  explicit CustomOp(const char* op_name);

  CustomOp& setForward(fcomp_t fcomp, const char* ctx);

  CustomOp& setBackward(fcomp_t fgrad, const char* ctx);

  CustomOp& setParseAttrs(parseAttrs_t func);

  CustomOp& setInferType(inferType_t func);

  CustomOp& setInferSType(inferSType_t func);

  CustomOp& setInferShape(inferShape_t func);

  CustomOp& setMutateInputs(mutateInputs_t func);

  CustomOp& setCreateOpState(createOpState_t func, const char* ctx);

  CustomOp& setIsSubgraphOp();

  void mapToVector();

  /*! \brief operator name */
  const char* name;

  /*! \brief operator functions */
  parseAttrs_t parse_attrs;
  inferType_t infer_type;
  inferSType_t infer_storage_type;
  inferShape_t infer_shape;
  mutateInputs_t mutate_inputs;
  bool isSGop;

  /*! \brief vector repr of ctx map to be easily loaded from c_api */
  std::vector<const char*> forward_ctx_cstr, backward_ctx_cstr, create_op_ctx_cstr;
  std::vector<fcomp_t> forward_fp, backward_fp;
  std::vector<createOpState_t> create_op_fp;

 private:
  void raiseDuplicateContextError();

  /*! \brief dedup context maps - static string ctx to custom function */
  std::unordered_map<const char*, fcomp_t> forward_ctx_map, backward_ctx_map;
  std::unordered_map<const char*, createOpState_t> create_op_ctx_map;
};

/*! \brief Custom Pass Create function template */
typedef MXReturnValue (*graphPass_t)(mxnet::ext::Graph* graph,
                                     const std::unordered_map<std::string, std::string>& options);

/*!
 * \brief An abstract class for graph passes
 */
class CustomPass {
 public:
  CustomPass();
  explicit CustomPass(const char* pass_name);

  CustomPass& setBody(graphPass_t fn);

  /*! \brief pass name */
  const char* name;
  /*! \brief pass function */
  graphPass_t pass;
};

/*! \brief Custom Subgraph Create function template */
typedef MXReturnValue (*supportedOps_t)(const mxnet::ext::Graph *graph, std::vector<int>* ids,
                                        const std::unordered_map<std::string,
                                                                 std::string>& options);
typedef MXReturnValue (*createSelector_t)(const mxnet::ext::Graph *graph,
                                          CustomOpSelector** sel_inst,
                                          const std::unordered_map<std::string,
                                                                   std::string>& options);
typedef MXReturnValue (*reviewSubgraph_t)(const mxnet::ext::Graph *subgraph, int subgraph_id,
                                          bool* accept,
                                          const std::unordered_map<std::string,
                                                                   std::string>& options,
                                          std::unordered_map<std::string,
                                                             std::string>* attrs);

/*!
 * \brief An abstract class for subgraph property
 */
class CustomPartitioner {
 public:
  CustomPartitioner();

  explicit CustomPartitioner(const char* backend_name);

  CustomPartitioner& addStrategy(const char* prop_name,
                                 const char* sg_name);

  CustomPartitioner& setSupportedOps(const char* prop_name, supportedOps_t fn);

  CustomPartitioner& setCreateSelector(const char* prop_name, createSelector_t fn);

  CustomPartitioner& setReviewSubgraph(const char* prop_name, reviewSubgraph_t fn);

  supportedOps_t getSupportedOps(int stg_id);

  createSelector_t getCreateSelector(int stg_id);

  reviewSubgraph_t getReviewSubgraph(int stg_id);

  /*! \brief partitioner name */
  const char* name;
  std::map<std::string, supportedOps_t> supported_map;
  std::map<std::string, createSelector_t> selector_map;
  std::map<std::string, reviewSubgraph_t> review_map;
  /*! \brief strategy names */
  std::vector<const char*> strategies;
  /*! \brief subgraph operator name */
  std::vector<const char*> op_names;
};

/*!
 * \brief Registry class to registers things (ops, properties)
 *        Singleton class
 */
template <class T>
class Registry {
 public:
  /*!
   * \brief get singleton pointer to class
   * \returns pointer to class
   */
  static Registry* get() PRIVATE_SYMBOL {
    static Registry inst;
    return &inst;
  }
  /*!
   * \brief add a new entry
   * \returns new object associated with registered name
   */
  T& add(const char* name) {
    T *entry = new T(name);
    entries.push_back(entry);
    return *entry;
  }
  int size() {
    return entries.size();
  }
  T& get(int idx) {
    return *(entries.at(idx));
  }

 private:
  /*! \brief constructor */
  Registry() {}
  /*! \brief destructor */
  ~Registry() {}
  /*! \brief map of entries in registry */
  std::vector<T*> entries;
};

/*!
 * \brief Macros to help with string concat
 * Annoyingly, the concat_ and concat macros are necessary to
 * be able to use __COUNTER__ in an identifier name
 */
#define MX_STR_CONCAT_(__a, __b) __a ## __b
#define MX_STR_CONCAT(__a, __b) MX_STR_CONCAT_(__a, __b)

/*! \brief convert a token to a string */
#define MX_STRINGIFY(x) #x
#define MX_TOSTRING(x) MX_STRINGIFY(x)

/*! \brief declare a variable with custom name */
#define MX_REGISTER_NAME_(Name) MXNet ## _CustomOp ## _ ## Name
#define MX_REGISTER_DEF_(Name) mxnet::ext::CustomOp MX_REGISTER_NAME_(Name)

#define MX_REGISTER_PROP_NAME_(Name) MXNet ## _CustomSubProp ## _ ## Name
#define MX_REGISTER_PROP_DEF_(Name) mxnet::ext::CustomPartitioner MX_REGISTER_PROP_NAME_(Name)

#define MX_REGISTER_PASS_NAME_(Name) MXNet ## _CustomPass ## _ ## Name
#define MX_REGISTER_PASS_DEF_(Name) mxnet::ext::CustomPass MX_REGISTER_PASS_NAME_(Name)

/*! \brief assign a var to a value */
#define REGISTER_OP(Name) MX_STR_CONCAT(MX_REGISTER_DEF_(Name), __COUNTER__) = \
    mxnet::ext::Registry<mxnet::ext::CustomOp>::get()->add(MX_TOSTRING(Name))

#define REGISTER_PARTITIONER(Name) \
  MX_STR_CONCAT(MX_REGISTER_PROP_DEF_(Name), __COUNTER__) = \
    mxnet::ext::Registry<mxnet::ext::CustomPartitioner>::get()->add(MX_TOSTRING(Name))

#define REGISTER_PASS(Name) \
  MX_STR_CONCAT(MX_REGISTER_PASS_DEF_(Name), __COUNTER__) = \
    mxnet::ext::Registry<mxnet::ext::CustomPass>::get()->add(MX_TOSTRING(Name))

/* -------------- BELOW ARE CTYPE FUNCTIONS PROTOTYPES --------------- */

/*!
 * \brief Following are the C type APIs implemented in the external library
 * Each API has a #define string that is used to lookup the function in the library
 * Followed by the function declaration
 */
#define MXLIB_OPREGSIZE_STR "_opRegSize"
typedef int (*opRegSize_t)(void);

#define MXLIB_OPREGGET_STR "_opRegGet"
typedef int (*opRegGet_t)(int idx, const char** name, int *isSGop,
                          const char*** forward_ctx, mxnet::ext::fcomp_t** forward_fp,
                          int* forward_count, const char*** backward_ctx,
                          mxnet::ext::fcomp_t** backward_fp, int* backward_count,
                          const char*** create_op_ctx, mxnet::ext::createOpState_t** create_op_fp,
                          int* create_op_count, mxnet::ext::parseAttrs_t* parse,
                          mxnet::ext::inferType_t* type, mxnet::ext::inferSType_t* stype,
                          mxnet::ext::inferShape_t* shape, mxnet::ext::mutateInputs_t* mutate);

#define MXLIB_OPCALLFREE_STR "_opCallFree"
typedef int (*opCallFree_t)(void* ptr);

#define MXLIB_OPCALLPARSEATTRS_STR "_opCallParseAttrs"
typedef int (*opCallParseAttrs_t)(parseAttrs_t parseAttrs, const char* const* keys,
                                  const char* const* vals, int num,
                                  int* num_in, int* num_out);

#define MXLIB_OPCALLINFERSHAPE_STR "_opCallInferShape"
typedef int (*opCallInferShape_t)(inferShape_t inferShape, const char* const* keys,
                                  const char* const* vals, int num,
                                  unsigned int** inshapes, int* indims, int num_in,
                                  unsigned int*** mod_inshapes, int** mod_indims,
                                  unsigned int*** outshapes, int** outdims, int num_out);

#define MXLIB_OPCALLINFERTYPE_STR "_opCallInferType"
typedef int (*opCallInferType_t)(inferType_t inferType, const char* const* keys,
                                 const char* const* vals, int num,
                                 int* intypes, int num_in, int* outtypes, int num_out);

#define MXLIB_OPCALLINFERSTYPE_STR "_opCallInferSType"
typedef int (*opCallInferSType_t)(inferSType_t inferSType, const char* const* keys,
                                 const char* const* vals, int num,
                                 int* intypes, int num_in, int* outtypes, int num_out);

#define MXLIB_OPCALLFCOMP_STR "_opCallFCompute"
typedef int (*opCallFComp_t)(fcomp_t fcomp, const char* const* keys,
                             const char* const* vals, int num,
                             const int64_t** inshapes, int* indims,
                             void** indata, int* intypes,
                             size_t* inIDs, const char** indev_type,
                             int* indev_id, int num_in,
                             const int64_t** outshapes, int* outdims,
                             void** outdata, int* outtypes,
                             size_t* outIDs, const char** outdev_type,
                             int* outdev_id, int num_out,
                             xpu_malloc_t cpu_malloc, void* cpu_alloc,
                             xpu_malloc_t gpu_malloc, void* gpu_alloc, void* cuda_stream,
                             sparse_malloc_t sparse_malloc, void* sparse_alloc,
                             int* instypes, int* outstypes,
                             void** in_indices, void** out_indices,
                             void** in_indptr, void** out_indptr,
                             int64_t* in_indices_shapes, int64_t* out_indices_shapes,
                             int64_t* in_indptr_shapes, int64_t* out_indptr_shapes,
                             void* rng_cpu_states, void* rng_gpu_states);

#define MXLIB_OPCALLMUTATEINPUTS_STR "_opCallMutateInputs"
typedef int (*opCallMutateInputs_t)(mutateInputs_t mutate, const char* const* keys,
                                    const char* const* vals, int num,
                                    int** mutate_indices, int* indices_size);

#define MXLIB_OPCALLCREATEOPSTATE_STR "_opCallCreateOpState"
typedef int (*opCallCreateOpState_t)(createOpState_t create_op, const char* const* keys,
                                     const char* const* vals, int num, const char* dev_type,
                                     int dev_id, unsigned int** inshapes, int* indims,
                                     int num_in, const int* intypes, void** state_op);

#define MXLIB_OPCALLDESTROYOPSTATE_STR "_opCallDestroyOpState"
typedef int (*opCallDestroyOpState_t)(void* state_op);

#define MXLIB_OPCALLFSTATEFULCOMP_STR "_opCallFStatefulCompute"
typedef int (*opCallFStatefulComp_t)(int is_forward, void* state_op,
                                     const int64_t** inshapes, int* indims,
                                     void** indata, int* intypes,
                                     size_t* inIDs, const char** indev_type,
                                     int* indev_id, int num_in,
                                     const int64_t** outshapes, int* outdims,
                                     void** outdata, int* outtypes,
                                     size_t* outIDs, const char** outdev_type,
                                     int* outdev_id, int num_out,
                                     xpu_malloc_t cpu_malloc, void* cpu_alloc,
                                     xpu_malloc_t gpu_malloc, void* gpu_alloc, void* stream,
                                     sparse_malloc_t sparse_malloc, void* sparse_alloc,
                                     int* instypes, int* outstypes,
                                     void** in_indices, void** out_indices,
                                     void** in_indptr, void** out_indptr,
                                     int64_t* in_indices_shapes, int64_t* out_indices_shapes,
                                     int64_t* in_indptr_shapes, int64_t* out_indptr_shapes,
                                     void* rng_cpu_states, void* rng_gpu_states);

#define MXLIB_PARTREGSIZE_STR "_partRegSize"
typedef int (*partRegSize_t)(void);

#define MXLIB_PARTREGGETCOUNT_STR "_partRegGetCount"
typedef int (*partRegGetCount_t)(int idx, const char** name);

#define MXLIB_PARTREGGET_STR "_partRegGet"
typedef void (*partRegGet_t)(int part_idx, int stg_idx, const char** strategy,
                             supportedOps_t* supportedOps, createSelector_t* createSelector,
                             reviewSubgraph_t* reviewSubgraph, const char** op_name);

#define MXLIB_PARTCALLSUPPORTEDOPS_STR "_partCallSupportedOps"
typedef int (*partCallSupportedOps_t)(supportedOps_t supportedOps, const char *json,
                                      int num_ids, int *ids, const char* const* opt_keys,
                                      const char* const* opt_vals, int num_opts);

#define MXLIB_PARTCALLCREATESELECTOR_STR "_partCallCreateSelector"
typedef int (*partCallCreateSelector_t)(createSelector_t createSelector, const char *json,
                                        void** selector, const char* const* opt_keys,
                                        const char* const* opt_vals, int num_opts);

#define MXLIB_PARTCALLSELECT_STR "_partCallSelect"
typedef void (*partCallSelect_t)(void* sel_inst, int nodeID, int* selected);

#define MXLIB_PARTCALLSELECTINPUT_STR "_partCallSelectInput"
typedef void (*partCallSelectInput_t)(void* sel_inst, int nodeID, int input_nodeID,
                                  int* selected);

#define MXLIB_PARTCALLSELECTOUTPUT_STR "_partCallSelectOutput"
typedef void (*partCallSelectOutput_t)(void* sel_inst, int nodeID, int output_nodeID,
                                   int* selected);

#define MXLIB_PARTCALLFILTER_STR "_partCallFilter"
typedef void (*partCallFilter_t)(void* sel_inst, int* candidates, int num_candidates,
                             int** keep, int* num_keep);

#define MXLIB_PARTCALLRESET_STR "_partCallReset"
typedef void (*partCallReset_t)(void* sel_inst);

#define MXLIB_PARTCALLREVIEWSUBGRAPH_STR "_partCallReviewSubgraph"
typedef int (*partCallReviewSubgraph_t)(reviewSubgraph_t reviewSubgraph, const char *json,
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
                                        const int* aux_dev_id);

#define MXLIB_PASSREGSIZE_STR "_passRegSize"
typedef int (*passRegSize_t)(void);

#define MXLIB_PASSREGGET_STR "_passRegGet"
typedef void (*passRegGet_t)(int pass_idx, graphPass_t* graphPass, const char** pass_name);

#define MXLIB_PASSCALLGRAPHPASS_STR "_passCallGraphPass"
typedef int (*passCallGraphPass_t)(graphPass_t graphPass, const char *in_graph,
                                   char** out_graph, const char* const* opt_keys,
                                   const char* const* opt_vals, int num_opts,
                                   const char* pass_name, const char* const* arg_names,
                                   int num_args, void* const* arg_data,
                                   const int64_t* const* arg_shapes, const int* arg_dims,
                                   const int* arg_types, const size_t* arg_IDs,
                                   const char* const* arg_dev_type, const int* arg_dev_id,
                                   const char* const* aux_names, int num_aux,
                                   void* const* aux_data, const int64_t* const* aux_shapes,
                                   const int* aux_dims, const int* aux_types,
                                   const size_t* aux_IDs, const char* const* aux_dev_type,
                                   const int* aux_dev_id, nd_malloc_t nd_malloc,
                                   const void* nd_alloc);

#define MXLIB_INITIALIZE_STR "initialize"
typedef int (*initialize_t)(int version);

#define MXLIB_OPVERSION_STR "_opVersion"
typedef int (*opVersion_t)();

#define MXLIB_MSGSIZE_STR "_msgSize"
typedef int (*msgSize_t)(void);

#define MXLIB_MSGGET_STR "_msgGet"
typedef int (*msgGet_t)(int idx, const char** msg);

/*! \brief StatefulOp wrapper class to pass to backend OpState */
class CustomStatefulOpWrapper {
 public:
  ~CustomStatefulOpWrapper();
  explicit CustomStatefulOpWrapper(CustomStatefulOp* inst, opCallDestroyOpState_t destroy)
    : instance(inst), destroy_(destroy) {}
  CustomStatefulOp* get_instance() { return instance; }
 private:
  CustomStatefulOp* instance;
  opCallDestroyOpState_t destroy_;
};

#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
#define MX_INT_RET  __declspec(dllexport) int __cdecl
#define MX_VOID_RET __declspec(dllexport) void __cdecl
#else
#define MX_INT_RET  int
#define MX_VOID_RET void
#endif

}  // namespace ext
}  // namespace mxnet

extern "C" {
  /*! \brief returns MXNet library version */
  MX_INT_RET _opVersion();

  /*! \brief returns number of ops registered in this library */
  MX_INT_RET _opRegSize();

  /*! \brief returns operator registration at specified index */
  MX_VOID_RET _opRegGet(int idx, const char** name, int *isSGop,
                        const char*** forward_ctx, mxnet::ext::fcomp_t** forward_fp,
                        int* forward_count, const char*** backward_ctx,
                        mxnet::ext::fcomp_t** backward_fp, int* backward_count,
                        const char*** create_op_ctx, mxnet::ext::createOpState_t** create_op_fp,
                        int* create_op_count, mxnet::ext::parseAttrs_t* parse,
                        mxnet::ext::inferType_t* type, mxnet::ext::inferSType_t* stype,
                        mxnet::ext::inferShape_t* shape, mxnet::ext::mutateInputs_t* mutate);

  /*! \brief calls free from the external library for library allocated arrays */
  MX_VOID_RET _opCallFree(void* ptr);

  /*! \brief returns status of calling parse attributes function for operator from library */
  MX_INT_RET _opCallParseAttrs(mxnet::ext::parseAttrs_t parseAttrs, const char* const* keys,
                               const char* const* vals, int num,
                               int* num_in, int* num_out);

  /*! \brief returns status of calling inferShape function for operator from library */
  MX_INT_RET _opCallInferShape(mxnet::ext::inferShape_t inferShape, const char* const* keys,
                               const char* const* vals, int num,
                               unsigned int** inshapes, int* indims, int num_in,
                               unsigned int*** mod_inshapes, int** mod_indims,
                               unsigned int*** outshapes, int** outdims, int num_out);

  /*! \brief returns status of calling inferType function for operator from library */
  MX_INT_RET _opCallInferType(mxnet::ext::inferType_t inferType, const char* const* keys,
                              const char* const* vals, int num,
                              int* intypes, int num_in, int* outtypes, int num_out);

  /*! \brief returns status of calling inferSType function for operator from library */
  MX_INT_RET _opCallInferSType(mxnet::ext::inferSType_t inferSType, const char* const* keys,
                               const char* const* vals, int num,
                               int* instypes, int num_in, int* outstypes, int num_out);

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
                             void* rng_cpu_states, void* rng_gpu_states);

  /*! \brief returns status of calling mutateInputs function for operator from library */
  MX_INT_RET _opCallMutateInputs(mxnet::ext::mutateInputs_t mutate, const char* const* keys,
                                 const char* const* vals, int num,
                                 int** mutate_indices, int* indices_size);

  /*! \brief returns status of calling createStatefulOp function for operator from library */
  MX_INT_RET _opCallCreateOpState(mxnet::ext::createOpState_t create_op, const char* const* keys,
                                  const char* const* vals, int num, const char* dev_type,
                                  int dev_id, unsigned int** inshapes, int* indims,
                                  int num_in, const int* intypes, void** state_op);

  /*! \brief returns status of deleting StatefulOp instance for operator from library */
  MX_VOID_RET _opCallDestroyOpState(void* state_op);

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
                                     void* rng_cpu_states, void* rng_gpu_states);

  /*! \brief returns number of partitioners registered in this library */
  MX_INT_RET _partRegSize();

  /* returns number of strategies registered for partitioner
   * at specified index */
  MX_INT_RET _partRegGetCount(int idx, const char** name);

  /*! \brief returns partitioner registration at specified index */
  MX_VOID_RET _partRegGet(int part_idx, int stg_idx, const char** strategy,
                          mxnet::ext::supportedOps_t* supportedOps,
                          mxnet::ext::createSelector_t* createSelector,
                          mxnet::ext::reviewSubgraph_t* reviewSubgraph, const char** op_name);

  /*! \brief returns status of calling supported ops function from library */
  MX_INT_RET _partCallSupportedOps(mxnet::ext::supportedOps_t supportedOps, const char *json,
                                   int num_ids, int *ids, const char* const* opt_keys,
                                   const char* const* opt_vals, int num_opts);

  /*! \brief returns status of calling create selector function from library */
  MX_INT_RET _partCallCreateSelector(mxnet::ext::createSelector_t createSelector, const char *json,
                                     void** selector, const char* const* opt_keys,
                                     const char* const* opt_vals, int num_opts);

  /*! \brief returns status of calling select function from library */
  MX_VOID_RET _partCallSelect(void* sel_inst, int nodeID, int* selected);

  /*! \brief returns status of calling select input function from library */
  MX_VOID_RET _partCallSelectInput(void* sel_inst, int nodeID,
                                   int input_nodeID, int* selected);

  /*! \brief returns status of calling select output function from library */
  MX_VOID_RET _partCallSelectOutput(void* sel_inst, int nodeID,
                                    int output_nodeID, int* selected);

  /*! \brief returns status of calling filter function from library */
  MX_VOID_RET _partCallFilter(void* sel_inst, int* candidates, int num_candidates,
                              int** keep, int* num_keep);

  /*! \brief returns status of calling reset selector function from library */
  MX_VOID_RET _partCallReset(void* sel_inst);

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
                                     const int* aux_dev_id);

  /*! \brief returns number of graph passes registered in this library */
  MX_INT_RET _passRegSize();

  /*! \brief returns pass registration at specified index */
  MX_VOID_RET _passRegGet(int pass_idx, mxnet::ext::graphPass_t* graphPass,
                          const char** pass_name);

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
                                const void* nd_alloc);

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

  MX_INT_RET _msgSize();

  /*! \brief returns operator registration at specified index */
  MX_VOID_RET _msgGet(int idx, const char** msg);
}  // extern "C"

#endif  // MXNET_LIB_API_H_
