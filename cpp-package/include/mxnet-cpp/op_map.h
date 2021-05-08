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
* \file op_map.h
* \brief definition of OpMap
* \author Chuntao Hong
*/

#ifndef MXNET_CPP_OP_MAP_H_
#define MXNET_CPP_OP_MAP_H_

#include <map>
#include <string>
#include "mxnet-cpp/base.h"
#include "dmlc/logging.h"

namespace mxnet {
namespace cpp {

/*!
* \brief OpMap instance holds a map of all the symbol creators so we can
*  get symbol creators by name.
*  This is used internally by Symbol and Operator.
*/
class OpMap {
 public:
  /*!
  * \brief Create an Mxnet instance
  */
  inline OpMap() {
    mx_uint num_symbol_creators = 0;
    AtomicSymbolCreator *symbol_creators = nullptr;
    int r =
      MXSymbolListAtomicSymbolCreators(&num_symbol_creators, &symbol_creators);
    CHECK_EQ(r, 0);
    for (mx_uint i = 0; i < num_symbol_creators; i++) {
      const char *name;
      const char *description;
      mx_uint num_args;
      const char **arg_names;
      const char **arg_type_infos;
      const char **arg_descriptions;
      const char *key_var_num_args;
      r = MXSymbolGetAtomicSymbolInfo(symbol_creators[i], &name, &description,
        &num_args, &arg_names, &arg_type_infos,
        &arg_descriptions, &key_var_num_args);
      CHECK_EQ(r, 0);
      symbol_creators_[name] = symbol_creators[i];
    }

    nn_uint num_ops;
    const char **op_names;
    r = NNListAllOpNames(&num_ops, &op_names);
    CHECK_EQ(r, 0);
    for (nn_uint i = 0; i < num_ops; i++) {
      OpHandle handle;
      r = NNGetOpHandle(op_names[i], &handle);
      CHECK_EQ(r, 0);
      op_handles_[op_names[i]] = handle;
    }
  }

  /*!
  * \brief Get a symbol creator with its name.
  *
  * \param name name of the symbol creator
  * \return handle to the symbol creator
  */
  inline AtomicSymbolCreator GetSymbolCreator(const std::string &name) {
    if (symbol_creators_.count(name) == 0)
      return GetOpHandle(name);
    return symbol_creators_[name];
  }

  /*!
  * \brief Get an op handle with its name.
  *
  * \param name name of the op
  * \return handle to the op
  */
  inline OpHandle GetOpHandle(const std::string &name) {
    return op_handles_[name];
  }

 private:
  std::map<std::string, AtomicSymbolCreator> symbol_creators_;
  std::map<std::string, OpHandle> op_handles_;
};

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_OP_MAP_H_
