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
*  Copyright (c) 2016 by Contributors
* \file symbol.h
* \brief definition of symbol
* \author Chuntao Hong, Zhang Chen
*/

#ifndef MXNET_CPP_SYMBOL_H_
#define MXNET_CPP_SYMBOL_H_

#include <map>
#include <string>
#include <vector>
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/ndarray.h"
#include "mxnet-cpp/op_map.h"

namespace mxnet {
namespace cpp {

class Executor;

/*!
* \brief struct to store SymbolHandle
*/
struct SymBlob {
 public:
  /*!
  * \brief default constructor
  */
  SymBlob() : handle_(nullptr) {}
  /*!
  * \brief construct with SymbolHandle to store
  */
  explicit SymBlob(SymbolHandle handle) : handle_(handle) {}
  /*!
  * \brief destructor, free the SymbolHandle
  */
  ~SymBlob() { MXSymbolFree(handle_); }
  /*!
  * \brief the SymbolHandle to store
  */
  SymbolHandle handle_;

 private:
  SymBlob(const SymBlob &);
  SymBlob &operator=(const SymBlob &);
};

/*!
* \brief Symbol interface
*/
class Symbol {
 public:
  Symbol() {}
  /*!
  * \brief construct a Symbol with SymbolHandle
  * \param handle the given SymbolHandle
  */
  explicit Symbol(SymbolHandle handle);
  /*!
  * \brief construct a variable Symbol
  * \param name the name of the variable
  */
  explicit Symbol(const char *name);
  /*!
  * \brief construct a variable Symbol
  * \param name the name of the variable
  */
  explicit Symbol(const std::string &name);
  Symbol operator+(const Symbol &rhs) const;
  Symbol operator-(const Symbol &rhs) const;
  Symbol operator*(const Symbol &rhs) const;
  Symbol operator/(const Symbol &rhs) const;
  Symbol operator%(const Symbol &rhs) const;

  Symbol operator+(mx_float scalar) const;
  Symbol operator-(mx_float scalar) const;
  Symbol operator*(mx_float scalar) const;
  Symbol operator/(mx_float scalar) const;
  Symbol operator%(mx_float scalar) const;
  Symbol Copy() const;
  /*!
  * \brief construct a variable Symbol
  * \param name the name of the variable
  */
  static Symbol Variable(const std::string &name = "");
  Symbol operator[](int index);
  Symbol operator[](const std::string &index);
  /*!
  * \brief Create a symbol that groups symbols together
  * \param symbols List of symbols to be groupe
  */
  static Symbol Group(const std::vector<Symbol> &symbols);
  /*!
  * \brief load Symbol from a JSON file
  * \param file_name the name of the file
  */
  static Symbol Load(const std::string &file_name);
  /*!
  * \brief load Symbol from a JSON string
  * \param json_str the JSON string
  */
  static Symbol LoadJSON(const std::string &json_str);
  /*!
  * \brief save Symbol to a file
  * \param file_name the name of the file
  */
  void Save(const std::string &file_name) const;
  /*!
  * \brief save Symbol into a JSON string
  */
  std::string ToJSON() const;
  /*!
  * \brief save Symbol into a JSON string
  * \retutrn the symbol whose outputs are all the internals.
  */
  Symbol GetInternals() const;
  /*!
  * \return the SymbolHandle
  */
  SymbolHandle GetHandle() const { return (blob_ptr_) ? blob_ptr_->handle_: nullptr; }
  /*!
  * \brief construct an operator Symbol, with given input Symbol and config
  * \param name the name of the Symbol
  * \param input_keys the vector of keys of the input
  * \param input_values the vector of the intput Symbols
  * \param config_keys the vector of keys of the config
  * \param config_values the vecotr of values of the config
  */
  Symbol(const std::string &operator_name, const std::string &name,
         std::vector<const char *> input_keys,
         std::vector<SymbolHandle> input_values,
         std::vector<const char *> config_keys,
         std::vector<const char *> config_values);
  /*!
  * \brief infer the shapes by providing shapes of known argument shapes.
  * \param arg_shapes map of argument name to shape of arguments with known
  * shapes.
  * \param in_shapes used to store infered shapes of input arguments.
  * \param out_shapes used to store infered shapes of outputs.
  * \param aux_shapes use to store the infered shapes of auxiliary states
  */
  void InferShape(
      const std::map<std::string, std::vector<mx_uint> > &arg_shapes,
      std::vector<std::vector<mx_uint> > *in_shape,
      std::vector<std::vector<mx_uint> > *aux_shape,
      std::vector<std::vector<mx_uint> > *out_shape) const;
  /*!
  * \brief List the arguments names.
  *
  * The position of the returned list also corresponds to calling position in
  *operator()
  * \return the arguments list of this symbol, they can be either named or
  *unnamed (empty string).
  */
  std::vector<std::string> ListArguments() const;
  /*! \return lists all argument names and aux states of the symbol */
  std::vector<std::string> ListInputs() const;
  /*! \return get the descriptions of outputs for this symbol */
  std::vector<std::string> ListOutputs() const;
  /*! \return get the descriptions of auxiliary data for this symbol */
  std::vector<std::string> ListAuxiliaryStates() const;
  /*! \return get all attributes for this symbol */
  std::map<std::string, std::string> ListAttributes() const;
  /*!
   * \brief set key-value attribute to the symbol
   * @param key string represent the key for the attribute
   * @param value string represent the value for the attribute
   */
  void SetAttribute(const std::string& key, const std::string& value);
  /*!
   * \brief set a series of key-value attribute to the symbol
   * @param attrs string:string map represent the key value attributes
   */
  void SetAttributes(const std::map<std::string, std::string>& attrs);
  /*! \return get number of outputs for this symbol */
  mx_uint GetNumOutputs() const;
  /*! \return get the new symbol through subgraph API for this symbol */
  mxnet::cpp::Symbol GetBackendSymbol(const std::string& backendName) const;
  /*! \return get the name of the symbol */
  std::string GetName() const;
  /*!
  * \brief infer and construct all the arrays to bind to executor by providing
  * some known arrays.
  * \param context the context of all the infered arrays
  * \param arg_arrays infered input arguments arrays.
  * \param arad_arrays infered arrays to store the gradient output of the input
  * arguments.
  * \param aux_arrays infered arrays that is used as internal state in op.
  * \param args_map map of some given arguments arrays.
  * \param args_grad_store map of some gradient given store arrays.
  * \param args_req_type map of some given type of gradient saving. Can only be
  * in {kNullOp, kAddTo, kWriteTo}.
  * \param aux_map NDArray that stores the internal state in op
  */
  void InferExecutorArrays(
      const Context &context, std::vector<NDArray> *arg_arrays,
      std::vector<NDArray> *grad_arrays, std::vector<OpReqType> *grad_reqs,
      std::vector<NDArray> *aux_arrays,
      const std::map<std::string, NDArray> &args_map,
      const std::map<std::string, NDArray> &arg_grad_store =
          std::map<std::string, NDArray>(),
      const std::map<std::string, OpReqType> &grad_req_type =
          std::map<std::string, OpReqType>(),
      const std::map<std::string, NDArray> &aux_map =
          std::map<std::string, NDArray>()) const;
  /*!
  * \brief infer and construct all the input arguments arrays to bind to
  * executor by providing some known arguments arrays.
  * \param context the context of all the infered arrays.
  * \param args_map map of all the infered input arguments arrays.
  * \param known_args map of some given arguments arrays.
  */
  void InferArgsMap(const Context &context,
                    std::map<std::string, NDArray> *args_map,
                    const std::map<std::string, NDArray> &known_args) const;
  /*!
  * \brief Create an executor by bind symbol with context and arguments.
  *  If user do not want to compute the gradients of i-th argument,
  *grad_req_type[i] can be kNullOp.
  *  The input arrays in the given maps should have the same name with the input
  *symbol.
  *  Only need some of the necessary arrays, and the other arrays can be infered
  *automatically.
  *
  * \param context the context of binding.
  * \param args_map the NDArray that stores the input arguments to the symbol.
  * \param arg_grad_store NDArray that is used to store the gradient output of
  *the input arguments.
  * \param grad_req_type requirment type of gradient saving. Can only be in
  *{kNullOp, kAddTo, kWriteTo}.
  * \param aux_map NDArray that stores the internal state in op
  * \return a new executor, which need to be free manually.
  */
  Executor *SimpleBind(const Context &context,
                       const std::map<std::string, NDArray> &args_map,
                       const std::map<std::string, NDArray> &arg_grad_store =
                           std::map<std::string, NDArray>(),
                       const std::map<std::string, OpReqType> &grad_req_type =
                           std::map<std::string, OpReqType>(),
                       const std::map<std::string, NDArray> &aux_map =
                           std::map<std::string, NDArray>());
  /*!
  * \brief Create an executor by bind symbol with context and arguments.
  *  If user do not want to compute the gradients of i-th argument,
  *grad_req_type[i] can be kNullOp.
  *
  * \param context the context of binding.
  * \param arg_arrays the NDArray that stores the input arguments to the symbol.
  * \param grad_arrays NDArray that is used to store the gradient output of the
  *input arguments.
  * \param grad_reqs requirment type of gradient saving. Can only be in
  *{kNullOp, kAddTo, kWriteTo}.
  * \param aux_arrays NDArray that is used as internal state in op
  * \param group_to_ctx dict of string to mx.Context
  * \param shared_exec Executor to share memory with. This is intended for
  *runtime reshaping, variable length sequencesn etc.  The returned executor
  *shares state with shared_exec, and should not be used in parallel with it.
  * \return a new executor, which need to be free manually.
  */
  Executor *Bind(const Context &context, const std::vector<NDArray> &arg_arrays,
                 const std::vector<NDArray> &grad_arrays,
                 const std::vector<OpReqType> &grad_reqs,
                 const std::vector<NDArray> &aux_arrays,
                 const std::map<std::string, Context> &group_to_ctx =
                     std::map<std::string, Context>(),
                 Executor *shared_exec = nullptr);

 private:
  std::shared_ptr<SymBlob> blob_ptr_;
  static OpMap*& op_map();
};
Symbol operator+(mx_float lhs, const Symbol &rhs);
Symbol operator-(mx_float lhs, const Symbol &rhs);
Symbol operator*(mx_float lhs, const Symbol &rhs);
Symbol operator/(mx_float lhs, const Symbol &rhs);
Symbol operator%(mx_float lhs, const Symbol &rhs);
}  // namespace cpp
}  // namespace mxnet
#endif  // MXNET_CPP_SYMBOL_H_
