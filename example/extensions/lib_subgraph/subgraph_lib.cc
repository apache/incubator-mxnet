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
 * \file subgraph_lib.cc
 * \brief subgraph operator implementation library file
 */

#include <iostream>
#include <algorithm>
#include "lib_api.h"

MXReturnValue parseAttrs(std::map<std::string, std::string> attrs,
                         int* num_in, int* num_out) {
  *num_in = 1;
  *num_out = 1;
  if (attrs.count(SUBGRAPH_SYM_JSON)) {
    // example of subgraph json parsing
    JsonParser jp;
    JsonVal val = jp.parse_to_json(attrs[SUBGRAPH_SYM_JSON]);
    int input = 0;
    for (auto &item : val.map[JsonVal("nodes")].list) {
      if (item.map[JsonVal("op")].str == "null")
        input++;
    }
    int output = val.map[JsonVal("heads")].list.size();
    *num_in = input;
    *num_out = output;
  }
  return MX_SUCCESS;
}

MXReturnValue inferType(std::map<std::string, std::string> attrs,
                        std::vector<int> &intypes,
                        std::vector<int> &outtypes) {
  outtypes[0] = intypes[0];
  return MX_SUCCESS;
}

MXReturnValue inferShape(std::map<std::string, std::string> attrs,
                         std::vector<std::vector<unsigned int>> &inshapes,
                         std::vector<std::vector<unsigned int>> &outshapes) {
  outshapes[0] = inshapes[0];
  return MX_SUCCESS;
}

/* function to execute log operator on floats */
void myLog(MXTensor &in, MXTensor &out) {
  float* inp = in.data<float>();
  float* outp = out.data<float>();
  for (int64_t i = 0; i < in.size(); i++) {
    outp[i] = logf(inp[i]);
  }
}
/* function to execute exp operator on floats */
void myExp(MXTensor &in, MXTensor &out) {
  float* inp = in.data<float>();
  float* outp =out.data<float>();
  for (int64_t i = 0; i < in.size(); i++) {
    outp[i] = expf(inp[i]);
  }
}

/* function to execute ops in subgraph
 * In MXNet, subgraphs are sorted in topological order
 * so all we need to do is go through the ops in order
 * and execute each op. 
 */
MXReturnValue myExecutor(std::vector<MXTensor> inputs,
                         std::vector<MXTensor> outputs,
                         std::string subgraph_sym) {
  std::cout << "Info: subgraph symbol is: " << std::endl;
  std::cout << subgraph_sym << std::endl;

  // convert json string to json object
  JsonParser parser;
  JsonVal json_val = parser.parse_to_json(subgraph_sym);
  // get nodes list
  JsonVal nodes = json_val.map[JsonVal("nodes")];
  //counter for inputs
  int input_cnt = 0;
  // temporary tensor storage
  std::vector<MXTensor> data;
  // track memory allocations to free later
  std::vector<void*> to_free;

  // loop over nodes
  for(int i=0; i<nodes.list.size(); i++) {
    JsonVal node = nodes.list[i];
    // get the op name
    std::string op = node.map[JsonVal("op")].str;
    // get node ID inputs to op
    JsonVal node_inputs = node.map[JsonVal("inputs")];
    
    // handle each op type
    if (op.compare("null") == 0) {
      // null is an input data to the subgraph, add to data storage
      data.push_back(inputs[input_cnt++]);
    } else if (op.compare("log") == 0) {
      // get input tensor based on node ID inputs from data storage
      MXTensor &input = data[node_inputs.list[0].list[0].num];
      // create temporary storage
      MXTensor tmp(malloc(input.size()*4), input.shape, input.dtype);
      // save allocated ptr to free later
      to_free.push_back(tmp.data_ptr);
      // execute log operator
      myLog(input,tmp);
      // add output tensor to data storage
      data.push_back(tmp);
    } else if (op.compare("exp") == 0) {
      // get input tensor based on node ID inputs from data storage
      MXTensor &input = data[node_inputs.list[0].list[0].num];
      // create temporary storage
      MXTensor tmp(malloc(input.size()*4), input.shape, input.dtype);
      // save allocated ptr to free later
      to_free.push_back(tmp.data_ptr);
      // execute exp operator 
      myExp(input,tmp);
      // add output tensor to data storage
      data.push_back(tmp);
    } else {
      std::cout << "Error! Unsupported op '" << op << "' found in myExecutor";
      return MX_FAIL;
    }
  }
  
  // get list of outputs from subgraph
  JsonVal heads = json_val.map[JsonVal("heads")];
  // copy all operator results to outputs of subgraph
  for (int j = 0; j < heads.list.size(); j++) {
    // get computed result
    MXTensor &result = data[heads.list[0].list[0].num];
    // get output tensor to pass to MX
    MXTensor &out = outputs[j];
    float *out_data = out.data<float>();
    float *res_data = result.data<float>();
    // loop and copy data
    for (int64_t i = 0; i < result.size(); i++) {
      out_data[i] = res_data[i];
    }
  }

  // free allocated temporary storage
  for (void* ptr : to_free) {
    free(ptr);
  }
  
  return MX_SUCCESS;
}

class MyStatefulOp : public CustomStatefulOp {
 public:
  explicit MyStatefulOp(std::string sym) : subgraph_sym(sym) {}

  MXReturnValue Forward(std::vector<MXTensor> inputs,
                        std::vector<MXTensor> outputs,
                        OpResource op_res) {
    return myExecutor(inputs, outputs, subgraph_sym);
  }

 private:
  std::string subgraph_sym;
};

MXReturnValue createOpState(std::map<std::string, std::string> attrs,
                            CustomStatefulOp** op_inst) {
  std::string serialized_subgraph = "[empty]";
  // MXNet subgraph is stored as Symbol in operator node attrs subgraphs field
  // custom subgraph is stored as json string in custom operator attrs map entry
  if (attrs.count(SUBGRAPH_SYM_JSON)) {
    // user can now parse json and run other custom ops inside subgraph
    serialized_subgraph = attrs[SUBGRAPH_SYM_JSON];
  }
  *op_inst = new MyStatefulOp(serialized_subgraph);
  std::cout << "Info: stateful operator created" << std::endl;
  return MX_SUCCESS;
}

REGISTER_OP(_custom_subgraph_op)
.setParseAttrs(parseAttrs)
.setInferType(inferType)
.setInferShape(inferShape)
.setCreateOpState(createOpState);

const std::vector<std::string> op_names({"exp","log"});

MXReturnValue mySupportedOps(std::string json,
                             const int num_ids,
                             int *ids) {
  //convert json string to json object
  JsonParser parser;
  JsonVal json_val = parser.parse_to_json(json);
  //get nodes list
  JsonVal nodes = json_val.map[JsonVal("nodes")];

  //loop over nodes
  for(int i=0; i<nodes.list.size(); i++) {
    JsonVal node = nodes.list[i];
    JsonVal op = node.map[JsonVal("op")];

    //get shape/type if available
    std::string shape;
    int dtype = -1;
    if(node.map.find(JsonVal("attrs")) != node.map.end()) {
      JsonVal attrs = node.map[JsonVal("attrs")];
      if(attrs.map.find(JsonVal("shape")) != attrs.map.end()) 
        shape = attrs.map[JsonVal("shape")].str;
      if(attrs.map.find(JsonVal("dtype")) != attrs.map.end())
        dtype = std::stoi(attrs.map[JsonVal("dtype")].str);
    }
    //check if op dtype is float
    std::cout << "dtype: " << dtype << "   " << ids[i] <<std::endl;
    if(dtype == kFloat32) {
      //check if op is in whitelist
      if(std::find(op_names.begin(),op_names.end(),op.str.c_str()) != op_names.end()) {
        std::cout << op.str << " : " << shape << " " << dtype << std::endl;
        // found op in whitelist, set value to 1 to include op in subgraph
        ids[i]=1;
      }
    }
  }
  return MX_SUCCESS;
}

REGISTER_PARTITIONER(myProp)
.addStrategy("strategy1", mySupportedOps, "_custom_subgraph_op");

MXReturnValue initialize(int version) {
  if (version >= 10400) {
    std::cout << "MXNet version " << version << " supported" << std::endl;
    return MX_SUCCESS;
  } else {
    std::cout << "MXNet version " << version << " not supported" << std::endl;
    return MX_FAIL;
  }
}
