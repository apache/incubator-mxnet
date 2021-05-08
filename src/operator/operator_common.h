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
 * Copyright (c) 2015 by Contributors
 * \file  operator_common.h
 * \brief common internal header of most operators
 *   this header includes utility functions operator can use
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_OPERATOR_COMMON_H_
#define MXNET_OPERATOR_OPERATOR_COMMON_H_

#include <dmlc/json.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <mxnet/operator.h>
#include <mxnet/ndarray.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/base.h>
#include <istream>
#include <ostream>
#include <string>
#include <vector>
#include <algorithm>
#include "../common/cuda/utils.h"
#include "../common/utils.h"

namespace mxnet {
namespace op {
/*!
 * \brief assign the expression to out according to request
 * \param out the data to be assigned
 * \param req the assignment request
 * \param exp the expression
 * \tparam OType output type
 * \tparam Exp expression type
 */
#define Assign(out, req, exp)           \
  {                                     \
    switch (req) {                      \
      case kNullOp:                     \
        break;                          \
      case kWriteTo:                    \
      case kWriteInplace:               \
        (out) = (exp);                  \
        break;                          \
      case kAddTo:                      \
        (out) += (exp);                 \
        break;                          \
      default:                          \
        LOG(FATAL) << "not reached";    \
    }                                   \
  }


/*! \brief exception throwed by InferShape error */
struct InferShapeError : public dmlc::Error {
  /*! \brief analyze message */
  std::string msg;
  /*! \brief corresponding input index */
  int index;
  // constructor
  InferShapeError(const std::string& msg_, int index)
    : dmlc::Error(msg_), msg(msg_), index(index) {}
};

/*! \brief exception throwed by InferShape error */
struct InferTypeError : public dmlc::Error {
  /*! \brief analyze message */
  std::string msg;
  /*! \brief corresponding input index */
  int index;
  // constructor
  InferTypeError(const std::string& msg_, int index)
    : dmlc::Error(msg_), msg(msg_), index(index) {}
};

/*! \brief exception throwed by InferStorageType error */
struct InferStorageTypeError : public dmlc::Error {
  /*! \brief analyze message */
  std::string msg;
  /*! \brief corresponding input index */
  int index;
  // constructor
  InferStorageTypeError(const std::string& msg_, int index)
    : dmlc::Error(msg_), msg(msg_), index(index) {}
};

/*! \brief check if shape is empty or contains unknown (0) dim.
 * DEPRECATED. */
inline bool shape_is_none(const mxnet::TShape& x) {
  return !mxnet::shape_is_known(x);
}

/*! \brief check if type is none (-1) */
inline bool type_is_none(const int& x) {
  return x == -1;
}

/*! \brief check if type is none (-1) */
inline bool storage_type_is_none(const int& x) {
  return x == -1;
}

/*! \brief check if shape is scalar({1}). */
inline bool shape_is_scalar(const mxnet::TShape& x) {
  return x.ndim() == 0;
}

/*! \brief get string representation of shape */
inline std::string shape_string(const mxnet::TShape& x) {
  std::ostringstream os;
  os << x;
  return os.str();
}

/*! \brief get string representation of data type */
inline std::string type_string(const int& x) {
  switch (x) {
    case mshadow::kFloat32:
      return "float32";
    case mshadow::kFloat64:
      return "float64";
    case mshadow::kFloat16:
      return "float16";
    case mshadow::kBfloat16:
      return "bfloat16";
    case mshadow::kInt8:
      return "int8";
    case mshadow::kUint8:
      return "uint8";
    case mshadow::kInt16:
      return "int8";
    case mshadow::kUint16:
      return "uint8";
    case mshadow::kInt32:
      return "int32";
    case mshadow::kInt64:
      return "int64";
  }
  return "unknown";
}

/*!
 * \brief Assign x to y. Checks for compatiblity when y is not empty.
 *  Allow missing dim in both x and y (as 0).
 * \param y target shape.
 * \param x source shape.
 * \return whether x and y are compatible.
 */
inline bool shape_assign(mxnet::TShape *y, const mxnet::TShape& x) {
  if (!mxnet::ndim_is_known(*y)) {
    *y = x;
    return true;
  } else if (y->ndim() != x.ndim()) {
    return !mxnet::ndim_is_known(x);
  } else {
    for (int i = 0; i < y->ndim(); ++i) {
      if (!mxnet::dim_size_is_known(*y, i)) {
        (*y)[i] = x[i];
      } else if ((*y)[i] != x[i] && x[i] >= 0) {
        return false;
      }
    }
    return true;
  }
}

/*!
 * \brief Assign x to y. Checks for compatiblity when y is not -1.
 * \param y target type.
 * \param x source type.
 * \return whether x and y are compatible.
 */
inline bool type_assign(int *y, const int& x) {
  if (*y == -1) {
    *y = x;
    return true;
  } else if (*y != x && x != -1) {
    return false;
  }
  return true;
}

/*!
 * \brief Assign x to y. Checks for compatiblity when y is not DispatchMode::kUndefined.
 * \param y target mode.
 * \param x source mode.
 * \return whether x and y are compatible.
 */
inline bool dispatch_mode_assign(DispatchMode *y, const DispatchMode& x) {
  if (*y == DispatchMode::kUndefined) {
    *y = x;
    return true;
  } else if (*y != x && x != DispatchMode::kUndefined) {
    return false;
  }
  return true;
}

/*! \brief Register op name as an alias */
#define MXNET_ADD_SPARSE_OP_ALIAS(__name$) \
  .add_alias("_sparse_" #__name$)

/*!
 * \brief macro assign shape to out if out is unknown otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param shape_array the shape array to store the result
 * \param index the index of in the array
 * \param shape the inferred shape
 */
#define SHAPE_ASSIGN_CHECK(shape_array, index, shape)                       \
  {                                                                         \
    if (!::mxnet::op::shape_assign(&(shape_array)[index], mxnet::TShape(shape))) { \
      std::ostringstream os;                                                \
      os << "Shape inconsistent, Provided = " << (shape_array)[index] << ','\
         << " inferred shape=" << shape;                                    \
      throw ::mxnet::op::InferShapeError(os.str(), index);                  \
    }                                                                       \
  }

/*!
 * \brief macro assign type to out if out is unknown (-1) otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param type_array the type array to store the result
 * \param index the index of in the array
 * \param type the inferred type
 */
#define TYPE_ASSIGN_CHECK(type_array, index, type)                          \
  {                                                                         \
    if (!::mxnet::op::type_assign(&(type_array)[index], type)) {            \
      std::ostringstream os;                                                \
      os << "Type inconsistent, Provided = "                                \
         << ::mxnet::op::type_string((type_array)[index]) << ','            \
         << " inferred type = " << ::mxnet::op::type_string(type);          \
      throw ::mxnet::op::InferTypeError(os.str(), index);                   \
    }                                                                       \
  }

/*!
 * \brief macro assign storage type to out if out is unknown (-1) otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param type_array the type array to store the result
 * \param index the index of in the array
 * \param type the inferred storage type
 */
#define STORAGE_TYPE_ASSIGN_CHECK(type_array, index, type)                  \
  {                                                                         \
    if (!::mxnet::op::type_assign(&(type_array)[index], type)) {            \
      std::ostringstream os;                                                \
      os << "Storage type inconsistent, Provided = "                        \
         << common::stype_string((type_array)[index]) << ','                \
         << " inferred storage type = " << common::stype_string(type);      \
      throw ::mxnet::op::InferStorageTypeError(os.str(), index);            \
    }                                                                       \
  }

/*!
 * \brief macro assign type to out if out is unknown (-1) otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param type_array the type array to store the result
 * \param index the index of in the array
 * \param type the inferred dispatch type
 */
#define DISPATCH_MODE_ASSIGN_CHECK(type_array, index, type)                 \
  {                                                                         \
    if (!::mxnet::op::dispatch_mode_assign(&(type_array)[index], type)) {   \
      std::ostringstream os;                                                \
      os << "Dispatch mode inconsistent, Provided = "                       \
         << common::dispatch_mode_string((type_array)[index]) << ','        \
         << " inferred mode = " << common::dispatch_mode_string(type);      \
      throw ::mxnet::op::InferStorageTypeError(os.str(), index);            \
    }                                                                       \
  }

/*!
 * \brief macro check if type is the same as expected.
 * \param type the type to be checked
 * \param expected the expected type
 */
#define UNIFORM_TYPE_CHECK(type, expected, arg)                         \
  {                                                                     \
    CHECK_EQ(type, expected) << "This layer requires uniform type. "    \
                             << "Expected '" << ::mxnet::op::type_string(expected)   \
                             << "' v.s. given '" << ::mxnet::op::type_string(type)   \
                             << "' at '" << arg << "'";                 \
  }

// helper macro to implement bind dispatch
#if MXNET_USE_CUDA
#define DO_BIND_DISPATCH(Method, ...)                                \
  if (ctx.dev_mask() == cpu::kDevMask) {                             \
      return Method<cpu>(__VA_ARGS__);                               \
    } else {                                                         \
      return Method<gpu>(__VA_ARGS__);                               \
    }
#else
#define DO_BIND_DISPATCH(Method, ...)                                \
  if (ctx.dev_mask() == cpu::kDevMask) {                             \
    return Method<cpu>(__VA_ARGS__);                                 \
  } else {                                                           \
    LOG(FATAL) << "GPU is not enabled";                              \
    return nullptr;                                                  \
  }
#endif

/*! \brief assign stype to target_stype, if successful,
 *         assign dispatch_mode to target_dispatch
 */
inline bool storage_type_assign(int* stype,
                                const NDArrayStorageType target_stype,
                                DispatchMode* dispatch,
                                const DispatchMode target_dispatch) {
  if (type_assign(stype, target_stype)) {
    DISPATCH_MODE_ASSIGN_CHECK(dispatch, 0, target_dispatch);
    return true;
  }
  return false;
}

/*! \brief assign the stype vector to target_stype, if successful,
 *         assign dispatch_mode to target_dispatch
 */
inline bool storage_type_assign(StorageTypeVector* stypes,
                                const NDArrayStorageType target_stype,
                                DispatchMode* dispatch,
                                const DispatchMode target_dispatch) {
  CHECK_GT(stypes->size(), 0);
  bool success = true;
  for (int& stype : *stypes) {
    if (!type_assign(&stype, target_stype)) {
      success = false;
    }
  }
  if (success) {
    DISPATCH_MODE_ASSIGN_CHECK(dispatch, 0, target_dispatch);
  }
  return success;
}

/*! \brief update the stype vector to default storage and dispatch_mode to fallback
 */
inline bool dispatch_fallback(StorageTypeVector* stypes, DispatchMode* dispatch) {
  for (auto& stype : *stypes) {
    type_assign(&stype, kDefaultStorage);
  }
  DISPATCH_MODE_ASSIGN_CHECK(dispatch, 0, DispatchMode::kFComputeFallback);
  return true;
}

inline std::vector<nnvm::NodeEntry>CreateNodeEntries(
  nnvm::ObjectPtr pNode,
  const std::vector<nnvm::NodeEntry>* pOgrads = nullptr,
  const std::vector<nnvm::NodeEntry>* pInputs = nullptr) {
  if (pOgrads)
    pNode->inputs.insert(pNode->inputs.end(), pOgrads->begin(), pOgrads->end());

  if (pInputs)
    pNode->inputs.insert(pNode->inputs.end(), pInputs->begin(), pInputs->end());

  if (!pNode->is_variable()) {
    CHECK_EQ(pNode->num_inputs(), pNode->inputs.size())
      << "Number of inputs to operator " << pNode->op()->name << " (" << pNode->num_inputs()
      << ") does not match the actual number of inputs provided to operator "
      << pNode->attrs.name << " (" << pNode->inputs.size() << ").";
  }

  std::vector<nnvm::NodeEntry> ret;
  for (uint32_t i = 0; i < pNode->num_outputs(); ++i)
    ret.emplace_back(nnvm::NodeEntry{pNode, i, 0});

  return ret;
}

// make a new node with operator op_name. Inputs are not filled.
inline nnvm::ObjectPtr MakeNode(
    const char* op_name, const std::string& name,
    std::vector<nnvm::NodeEntry> const * inputs = nullptr,
    std::unordered_map<std::string, std::string> const * dict = nullptr,
    nnvm::ObjectPtr const * fwd_node = nullptr) {
  auto p = nnvm::Node::Create();
  p->attrs.op = nnvm::Op::Get(op_name);
  p->attrs.name = name;
  if (dict != nullptr) p->attrs.dict = *dict;
  if (inputs != nullptr) p->inputs = *inputs;
  if (fwd_node != nullptr) {
    p->control_deps.emplace_back(*fwd_node);
  }
  if (p->op()->attr_parser != nullptr) {
    p->op()->attr_parser(&(p->attrs));
  }
  if (inputs != nullptr) {
    CHECK_EQ(p->num_inputs(), p->inputs.size())
      << "Number of inputs to operator " << op_name << " (" << p->num_inputs()
      << ") does not match the actual number of inputs provided to operator "
      << name << " (" << p->inputs.size() << ").";
  }
  return p;
}

inline nnvm::ObjectPtr MakeNode(
    const char* op_name, const std::string& name,
    const std::vector<nnvm::NodeEntry>& inputs,
    std::unordered_map<std::string, std::string> const * dict,
    nnvm::ObjectPtr const * fwd_node) {
  return MakeNode(op_name, name, &inputs, dict, fwd_node);
}


// quick helper to make node
inline std::vector<nnvm::NodeEntry> MakeGradNode(
    const char* op_name, const nnvm::ObjectPtr& n,
    const std::vector<nnvm::NodeEntry>& inputs,
    const std::unordered_map<std::string, std::string>& dict) {
  auto p = MakeNode(op_name, n->attrs.name + "_backward",
                    &inputs, &dict, &n);

  return CreateNodeEntries(p);
}

// quick helper to make gradient nodes that simply pass back zero. could be used in output ops.
inline std::vector<nnvm::NodeEntry> MakeZeroGradNodes(
    const nnvm::ObjectPtr& n,
    const std::vector<nnvm::NodeEntry>& ograds) {
  std::vector<nnvm::NodeEntry> ret;
  for (uint32_t i = 0; i < n->num_inputs(); ++i) {
    std::ostringstream os;
    if (1 == n->num_inputs()) {
      os << n->attrs.name << "_backward";
    } else {
      os << n->attrs.name << "_in" << i << "_backward";
    }
    ret.emplace_back(MakeNode("zeros_like", os.str(), {n->inputs[i]}, nullptr, &n));
  }
  return ret;
}


// check whether all output grads are zero.
inline bool CheckGradAllZero(const std::vector<nnvm::NodeEntry>& ograds) {
  static const auto zero_op = nnvm::Op::Get("_zeros");
  static const auto zero_like_op = nnvm::Op::Get("zeros_like");
  if (ograds.empty())
    return false;
  for (const auto& grad : ograds) {
    if (!grad.node)
      return false;
    if (grad.node->op() != zero_op && grad.node->op() != zero_like_op )
      return false;
  }
  return true;
}

// make gradient node that doesn't add to objective.
// i.e. igrads are always zero when ograds are zero.
inline std::vector<nnvm::NodeEntry> MakeNonlossGradNode(
    const char* op_name, const nnvm::ObjectPtr& n,
    const std::vector<nnvm::NodeEntry>& ograds,
    const std::vector<nnvm::NodeEntry>& inputs,
    const std::unordered_map<std::string, std::string>& dict) {
  if (CheckGradAllZero(ograds))
    return MakeZeroGradNodes(n, ograds);
  auto p = MakeNode(op_name, n->attrs.name + "_backward",
                    nullptr, &dict, &n);

  return CreateNodeEntries(p, &ograds, &inputs);
}

/*! \brief Parse keyword arguments as PType arguments and save to parsed */
template<typename PType>
inline void ParamParser(nnvm::NodeAttrs* attrs) {
  PType param;
  try {
    param.Init(attrs->dict);
  } catch (const dmlc::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto& k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
  attrs->parsed = std::move(param);
}

inline void CheckAllRowsPresent(const NDArray& arr, const std::string& func,
                                const std::string& param) {
  if (arr.storage_type() == kRowSparseStorage) {
    CHECK(arr.storage_shape()[0] == arr.shape()[0]) << func
          << " for RowSparse " << param << " is only implemented for "
          << "RowSparse " << param << " with all rows containing non-zeros. "
          << "Expects " << param << ".data.shape[0] (" << arr.storage_shape()[0]
          << ") == " << param << ".shape[0] (" << arr.shape()[0] << ").";
  } else {
    CHECK(arr.storage_type() == kDefaultStorage);
  }
}

inline void LogUnimplementedOp(const nnvm::NodeAttrs& attrs,
                               const OpContext &ctx,
                               const std::vector<NDArray> &inputs,
                               const std::vector<OpReqType> &req,
                               const std::vector<NDArray> &outputs) {
    using common::operator_string;
    LOG(FATAL) << "Not implemented: " << operator_string(attrs, ctx, inputs, req, outputs);
}

class OpSignature {
  std::vector<int64_t> eles;
  uint64_t hash;

 public:
  OpSignature() {
    hash = 0;
  }

  explicit OpSignature(uint64_t hash) {
    this->hash = hash;
  }

  /*
   * This is to reserve space for the vector.
   */
  void Reserve(size_t num) {
    eles.reserve(num);
  }

  /*
   * We provide different methods to add signature to an op.
   * For operations, such as convolutin and fully connected, which determines
   * the optimal data layout for the op, we only need to use the shape and data
   * type to sign the op. For other operations, such as activation, which uses
   * whatever layout in the input array, we have to use the shape, the data type
   * and the layout to sign the op.
   */

#if MXNET_USE_ONEDNN == 1
  void AddSign(const mkldnn::memory &mem) {
    auto desc = mem.get_desc();
    hash = hash * 2 + desc.data.format_kind;
    eles.push_back(desc.data.format_kind);
    hash = hash * 2 + desc.data.data_type;
    eles.push_back(desc.data.data_type);
    for (int i = 0; i < desc.data.ndims; i++) {
      hash = hash * 2 + desc.data.dims[i];
      eles.push_back(desc.data.dims[i]);
    }
    switch (desc.data.format_kind) {
      case mkldnn_blocked:
        hash = hash * 2 + desc.data.ndims;
        eles.push_back(desc.data.ndims);
        for (int i = 0; i < desc.data.ndims; i++) {
          hash = hash * 2 + desc.data.format_desc.blocking.strides[i];
          eles.push_back(desc.data.format_desc.blocking.strides[i]);
        }
        hash = hash * 2 + desc.data.format_desc.blocking.inner_nblks;
        eles.push_back(desc.data.format_desc.blocking.inner_nblks);
        for (int i = 0; i < desc.data.format_desc.blocking.inner_nblks; i++) {
          hash = hash * 2 + desc.data.format_desc.blocking.inner_blks[i];
          hash = hash * 2 + desc.data.format_desc.blocking.inner_idxs[i];
          eles.push_back(desc.data.format_desc.blocking.inner_blks[i]);
          eles.push_back(desc.data.format_desc.blocking.inner_idxs[i]);
        }
        break;
      case mkldnn_format_kind_wino:
        hash = hash * 2 + desc.data.format_desc.wino_desc.wino_format;
        eles.push_back(desc.data.format_desc.wino_desc.wino_format);
        break;
      case mkldnn_format_kind_rnn_packed:
        hash = hash * 2 + desc.data.format_desc.rnn_packed_desc.format;
        eles.push_back(desc.data.format_desc.rnn_packed_desc.format);
        hash = hash * 2 + desc.data.format_desc.rnn_packed_desc.n_parts;
        eles.push_back(desc.data.format_desc.rnn_packed_desc.n_parts);
        for (int i = 0; i < desc.data.format_desc.rnn_packed_desc.n_parts; ++i) {
          hash = hash * 2 + desc.data.format_desc.rnn_packed_desc.parts[i];
          hash = hash * 2 + desc.data.format_desc.rnn_packed_desc.part_pack_size[i];
          hash = hash * 2 + desc.data.format_desc.rnn_packed_desc.pack_part[i];
          eles.push_back(desc.data.format_desc.rnn_packed_desc.parts[i]);
          eles.push_back(desc.data.format_desc.rnn_packed_desc.part_pack_size[i]);
          eles.push_back(desc.data.format_desc.rnn_packed_desc.pack_part[i]);
        }
        hash = hash * 2 + desc.data.format_desc.rnn_packed_desc.n;
        hash = hash * 2 + desc.data.format_desc.rnn_packed_desc.ldb;
        hash = hash * 2 + desc.data.format_desc.rnn_packed_desc.offset_compensation;
        hash = hash * 2 + desc.data.format_desc.rnn_packed_desc.size;
        eles.push_back(desc.data.format_desc.rnn_packed_desc.n);
        eles.push_back(desc.data.format_desc.rnn_packed_desc.ldb);
        eles.push_back(desc.data.format_desc.rnn_packed_desc.offset_compensation);
        eles.push_back(desc.data.format_desc.rnn_packed_desc.size);
        break;
      default:
      // nothing need to add
        break;
    }
  }
#endif

  void AddSign(const std::vector<NDArray> &arrs) {
    for (auto &arr : arrs) {
      AddSign(arr);
    }
  }

  void AddSign(const NDArray &arr) {
#if MXNET_USE_ONEDNN == 1
    if (arr.IsMKLDNNData()) {
      AddSign(*(arr.GetMKLDNNData()));
    } else {
#endif
      hash = hash * 2 + arr.dtype();
      eles.push_back(arr.dtype());
      AddSign(arr.shape());
#if MXNET_USE_ONEDNN == 1
    }
#endif
  }

  void AddSign(const mxnet::ShapeVector &shapes) {
    for (auto &shape : shapes) {
      AddSign(shape);
    }
  }

  void AddSign(const mxnet::TShape &shape) {
    for (int i = 0; i < shape.ndim(); i++) {
      hash = hash * 2 + shape[i];
      eles.push_back(shape[i]);
    }
  }

  void AddSign(int val) {
    hash = hash * 2 + val;
    eles.push_back(val);
  }

  void AddSign(float val) {
    hash = dmlc::HashCombine(hash, val);
    eles.push_back(val);
  }

  bool operator==(const OpSignature &sign) const {
    if (hash != sign.hash)
      return false;
    if (eles.size() != sign.eles.size())
      return false;
    for (size_t i = 0; i < eles.size(); i++)
      if (eles[i] != sign.eles[i])
        return false;
    return true;
  }

  uint64_t GetHash() const {
    return hash;
  }
};

struct OpHash {
  size_t operator()(const OpSignature &sign) const {
    return sign.GetHash();
  }
};

template<typename ParamType>
class ParamOpSign: public OpSignature {
  const ParamType param;

  static size_t hash(const ParamType &param) {
    std::hash<ParamType> fn;
    return fn(param);
  }

 public:
  explicit ParamOpSign(const ParamType &_param): OpSignature(
      hash(_param)), param(_param) {
  }

  bool operator==(const ParamOpSign<ParamType> &sign) const {
    const OpSignature &this_upper = *this;
    const OpSignature &other_upper = sign;
    return this_upper == other_upper && param == sign.param;
  }
};

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_OPERATOR_COMMON_H_
