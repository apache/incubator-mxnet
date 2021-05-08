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
*  Copyright (c) 2019 by Contributors
* \file contrib.h
* \brief utility function to enable some contrib features
* \author Haohuan Wang
*/
#ifndef MXNET_CPP_CONTRIB_H_
#define MXNET_CPP_CONTRIB_H_

#include <iostream>
#include <string>
#include <map>
#include <vector>
#include "mxnet-cpp/symbol.h"

namespace mxnet {
namespace cpp {
namespace details {

  /*!
   * split a string with the given delimiter
   * @param str string to be parsed
   * @param delimiter delimiter
   * @return delimited list of string
   */
  inline std::vector<std::string> split(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> splitted;
    size_t last = 0;
    size_t next = 0;
    while ((next = str.find(delimiter, last)) != std::string::npos) {
      splitted.push_back(str.substr(last, next - last));
      last = next + 1;
    }
    splitted.push_back(str.substr(last));
    return splitted;
  }

}  // namespace details

namespace contrib {

  // needs to be same with
  //   https://github.com/apache/incubator-mxnet/blob/1c874cfc807cee755c38f6486e8e0f4d94416cd8/src/operator/subgraph/tensorrt/tensorrt-inl.h#L190
  static const std::string TENSORRT_SUBGRAPH_PARAM_IDENTIFIER = "subgraph_params_names";
  // needs to be same with
  //   https://github.com/apache/incubator-mxnet/blob/master/src/operator/subgraph/tensorrt/tensorrt.cc#L244
  static const std::string TENSORRT_SUBGRAPH_PARAM_PREFIX = "subgraph_param_";
  /*!
   * this is a mimic to https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/contrib/tensorrt.py#L37
   * @param symbol symbol that already called subgraph api
   * @param argParams original arg params, params needed by tensorrt will be removed after calling this function
   * @param auxParams original aux params, params needed by tensorrt will be removed after calling this function
   */
  inline void InitTensorRTParams(const mxnet::cpp::Symbol& symbol,
      std::map<std::string, mxnet::cpp::NDArray> *argParams,
      std::map<std::string, mxnet::cpp::NDArray> *auxParams) {
    mxnet::cpp::Symbol internals = symbol.GetInternals();
    mx_uint numSymbol = internals.GetNumOutputs();
    for (mx_uint i = 0; i < numSymbol; ++i) {
        std::map<std::string, std::string> attrs = internals[i].ListAttributes();
        if (attrs.find(TENSORRT_SUBGRAPH_PARAM_IDENTIFIER) != attrs.end()) {
            std::string new_params_names;
            std::map<std::string, mxnet::cpp::NDArray> tensorrtParams;
            std::vector<std::string> keys = details::split(
                attrs[TENSORRT_SUBGRAPH_PARAM_IDENTIFIER], ";");
            for (const auto& key : keys) {
                if (argParams->find(key) != argParams->end()) {
                    new_params_names += key + ";";
                    tensorrtParams[TENSORRT_SUBGRAPH_PARAM_PREFIX + key] = (*argParams)[key];
                    argParams->erase(key);
                } else if (auxParams->find(key) != auxParams->end()) {
                    new_params_names += key + ";";
                    tensorrtParams[TENSORRT_SUBGRAPH_PARAM_PREFIX + key] = (*auxParams)[key];
                    auxParams->erase(key);
                }
            }
            std::map<std::string, std::string> new_attrs = {};
            for (const auto& kv : tensorrtParams) {
                // passing the ndarray address into TRT node attributes to get the weight
                uint64_t address = reinterpret_cast<uint64_t>(kv.second.GetHandle());
                new_attrs[kv.first] = std::to_string(address);
            }
            if (!new_attrs.empty()) {
                internals[i].SetAttributes(new_attrs);
                internals[i].SetAttribute(TENSORRT_SUBGRAPH_PARAM_IDENTIFIER,
                    new_params_names.substr(0, new_params_names.length() - 1));
            }
        }
    }
}

}  // namespace contrib
}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_CONTRIB_H_
