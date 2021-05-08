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

#include <sys/stat.h>
#include <nnvm/pass_functions.h>
#include <algorithm>
#include <mutex>
#include "./fused_op.h"
#include "./fused_op-inl.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../../imperative/exec_pass.h"
#include "../../common/cuda/utils.h"
#include "../../common/cuda/rtc.h"

namespace mxnet {

namespace {

inline std::string mshadowTypeToString(int type) {
  switch (type) {
    case mshadow::kFloat32:
      return "float";
    case mshadow::kFloat64:
      return "double";
    case mshadow::kFloat16:
      return "half";
    case mshadow::kUint8:
      return "unsigned char";
    case mshadow::kInt8:
      return "char";
    case mshadow::kUint16:
      return "unsigned short";
    case mshadow::kInt16:
      return "short";
    case mshadow::kInt32:
      return "int";
    case mshadow::kInt64:
      return "long long";
    case mshadow::kBool:
      return "bool";
    default:
      LOG(FATAL) << "Unknown type enum " << type;
  }
  return "";
}

inline int mshadowTypeToVectorLength(int type) {
  switch (type) {
    case mshadow::kFloat32:
      return 1;
    case mshadow::kFloat64:
      return 1;
    case mshadow::kFloat16:
      return 2;
    case mshadow::kUint8:
      return 4;
    case mshadow::kInt8:
      return 4;
    case mshadow::kInt32:
      return 1;
    case mshadow::kInt64:
      return 1;
    case mshadow::kBool:
      return 4 / sizeof(bool);
    default:
      LOG(FATAL) << "Unknown type enum " << type;
  }
  return 0;
}

inline void replaceString(std::string *input, const std::string old, const std::string repl) {
    size_t pos = 0;
    while ((pos = input->find(old, pos)) != std::string::npos) {
        input->replace(pos, old.size(), repl);
        pos += repl.size();
    }
}

inline std::vector<int> splitStringToVector(const std::string& input, const std::string def) {
    size_t pos_start = 0, pos_end;
    const std::string& s = input.substr(1, input.length()-2);
    std::vector<int> res;

    auto convert_token = [def](std::string token){
        if (token == def) {
            return 0;
        }
        return std::stoi(token);
    };

    while ((pos_end = s.find(",", pos_start)) != std::string::npos) {
        std::string token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + 1;
        if (token.length() > 0) {
            res.push_back(convert_token(token));
        }
    }

    if (pos_start < s.length()) {
        res.push_back(convert_token(s.substr(pos_start)));
    }
    return res;
}

std::string ParseOpDescription(const std::vector<std::string>& op_desc,
                               const std::map<std::pair<int, int>, std::string>& variables,
                               const nnvm::IndexedGraph::Node& node) {
  const auto* source = node.source;
  std::string fmt = op_desc[0];
  for (size_t j = 1; j < op_desc.size(); ++j) {
    const std::string& desc = op_desc[j];
    std::string sub;
    if (desc[0] == '_') {
      // Argument
      const int arg_id = std::stoi(desc.substr(1));
      sub = variables.at({node.inputs[arg_id].node_id, node.inputs[arg_id].index});
    } else {
      sub = source->attrs.dict.at(desc);
    }
    size_t pos = fmt.find("%");
    CHECK_NE(pos, std::string::npos);
    fmt.replace(pos, 1, sub);
  }
  return fmt;
}

void AddShape(const mxnet::TShape& shape,
              std::vector<std::vector<int>>* shapes) {
  // We need alignment to 8 bytes for size_t in the Shape struct
  // so if ndim is odd, there will be 4B of padding
  int ndim = shape.ndim();
  const int offset = ndim % 2 == 0 ? 2 : 3;
  shapes->push_back(std::vector<int>(ndim + offset));
  std::vector<int>& tensor_shapes = shapes->back();
  size_t total_size = 1;
  for (int i = ndim-1; i >= 0; i--) {
    tensor_shapes[i] = shape[i];
    total_size *= shape[i];
  }
  size_t * shape_size_ptr = reinterpret_cast<size_t*>(&tensor_shapes[ndim + offset - 2]);
  *shape_size_ptr = total_size;
}

void AddPointerAndShape(const TBlob& data,
                        std::vector<void*> *ptrs,
                        std::vector<std::vector<int>>* shapes,
                        mshadow::Stream<gpu> * s) {
  using namespace mshadow;
  MSHADOW_TYPE_SWITCH_WITH_BOOL(data.type_flag_, DType, {
    Tensor<gpu, 1, DType> tensor = data.FlatTo1D<gpu, DType>(s);
    ptrs->push_back(tensor.dptr_);
    AddShape(data.shape_, shapes);
  });
}

}  // namespace

std::string FusedOp::GenerateCode(const std::vector<OpReqType> &req,
                           const std::vector<int> &in_dtypes,
                           const std::vector<int> &out_dtypes,
                           const std::vector<int> &in_ndims,
                           const std::vector<int> &out_ndims,
                           const mxnet::ShapeVector &node_shapes,
                           const std::vector<int> &node_dtypes,
                           const int nvec,
                           const std::string &kernel_name,
                           std::vector<uint32_t>* check_shapes) {
  const auto& g = subgraph_.indexed_graph();
  std::string code = "";
  int temp_name_counter = 0;
  using NodeEntry = nnvm::IndexedGraph::NodeEntry;
  std::map<std::pair<int, int>, std::string> variables;
  std::map<int, int> load_index;
  bool check_shapes_compile = true;

  std::vector<uint32_t> outputs(g.num_nodes());

  for (size_t i = 0; i < g.num_nodes(); ++i) {
    const auto& node = g[i];
    if (node.source != nullptr) {
      outputs[i] = node.source->num_outputs();
    } else {
      outputs[i] = 0;
    }
  }

  for (size_t i = 0; i < g.num_nodes(); ++i) {
    const auto& node = g[i];
    const auto* source = node.source;
    if (source != nullptr) {
        if (source->is_variable()) {
            load_index[i] = 1;
        } else {
            std::string op_name = source->op()->name;
            if (fusion::slice_ops.find(op_name) != fusion::slice_ops.end()) {
                load_index[node.inputs[0].node_id] = 0;
            }
        }
    }
  }
  for (size_t i = 0; i < g.num_nodes(); ++i) {
    const auto& node = g[i];
    const auto* source = node.source;
    if (source != nullptr) {
      if (source->is_variable()) {
        if (load_index[i]) {
          const auto& var_name = source->attrs.name;
          code += "const auto vec_" + var_name + " = op::load_index<nvec>(" +
                   var_name + ", offset, " + var_name + "_shape);\n";
          variables[{i, 0}] = var_name;
        }
        CHECK_EQ(outputs[i], 1);
      } else {
        std::string op_name = source->op()->name;
        if (fusion::slice_ops.find(op_name) != fusion::slice_ops.end()) {
          int node_id = node.inputs[0].node_id;
          const uint32_t input_entry_id = g.entry_id(node.inputs[0]);
          const auto& shape = node_shapes[input_entry_id];
          const int ndim = shape.ndim();
          const auto& var_name = g[node_id].source->attrs.name;
          const auto vec_name = "vec_" + var_name + "_" + std::to_string(i);
          load_index[node_id] = 0;
          auto parse_tuple = [ndim](const std::string& input, const std::string& def) {
            std::string out = input;
            replaceString(&out, " ", "");
            if (out[0] == '(') {
              replaceString(&out, "(", "{");
              replaceString(&out, ")", "}");
              // First check if out is ()
              int n_entries = out.size() != 2;
              for (size_t i = 1; i < out.size() - 1; ++i) {
                if (out[i] == ',') {
                  ++n_entries;
                }
              }
              if (n_entries != ndim) {
                out.pop_back();
                for (int i = n_entries; i < ndim; ++i) {
                  out += "," + def;
                }
                out += "}";
              }
            } else {
              out = "{" + std::move(out);
              for (int i = 1; i < ndim; ++i) {
                out += "," + def;
              }
              out += "}";
            }
            replaceString(&out, "None", def);
            return out;
          };
          auto parse_int = [](const std::string& input, const std::string& def) {
            std::string out = input;
            replaceString(&out, " ", "");
            replaceString(&out, "None", def);
            return out;
          };
          auto build_tuple = [ndim](int axis, const std::string str, const std::string def) {
            if (axis < 0 &&
                axis >= -ndim) {
              axis += ndim;
            }
            if (axis < 0 || axis >= ndim) {
              LOG(FATAL) << "Axis " << axis << " is out of bounds for array of dimension " << ndim;
            }
            std::string tuple = "{";
            for (int i = 0; i < axis; i++) {
                tuple += def + ",";
            }
            tuple += str;
            for (int i = axis + 1; i < ndim; i++) {
                tuple += "," + def;
            }
            tuple += "}";
            return tuple;
          };
          auto check_tuple = [ndim, nvec](const std::string str) {
            std::vector<int> tuple = splitStringToVector(str, "INT_MAX");
            if (tuple[ndim-1] % nvec == 0) {
              return true;
            }
            return false;
          };
          auto build_string_end = [i, ndim, var_name](std::string* code) {
            std::string end_var_name = var_name + "_" + std::to_string(i) + "_end";
            *code += "op::Shape<" + std::to_string(ndim) + "> "+ end_var_name + ";\n";
            *code += end_var_name + ".set(INT_MAX);\n";
            return end_var_name;
          };
          std::string begin;
          std::string end;
          if (op_name == "broadcast_like" || op_name == "slice_like") {
            uint32_t like_id = g.entry_id(i, 0);
            begin = build_tuple(0, "0", "0");
            std::string extra_var_name = "extra_" + std::to_string(like_id) + "_shape";
            if (std::find(extra_shape_args_.begin(), extra_shape_args_.end(), like_id) ==
                extra_shape_args_.end()) {
                extra_shape_args_.push_back(like_id);
            }
            if (check_shapes) {
              check_shapes->push_back(like_id);
              check_shapes->push_back(input_entry_id);
            }
            end = extra_var_name;
          } else {
            if (op_name == "slice_axis") {
              begin = parse_int(source->attrs.dict.at("begin"), "0");
              end = parse_int(source->attrs.dict.at("end"), "INT_MAX");
              int axis = std::stoi(source->attrs.dict.at("axis"));
              begin = build_tuple(axis, begin, "0");
              end = build_tuple(axis, end, "INT_MAX");
            } else {
              begin = parse_tuple(source->attrs.dict.at("begin"), "0");
              end = parse_tuple(source->attrs.dict.at("end"), "INT_MAX");
            }
            if (check_shapes) {
              if (check_tuple(begin) && check_tuple(end)) {
                check_shapes->push_back(input_entry_id);
              } else {
                check_shapes_compile = false;
              }
            }
          }
          std::string slice_func = "load_slice";
          if (!check_shapes) {
            slice_func = "fast_" + slice_func;
          }
          code += "const auto " + vec_name + " = op::" + slice_func + "<nvec>(" +
                  var_name + ", " + var_name + "_shape," + begin +
                  "," + end + ", offset);\n";
          CHECK_EQ(outputs[i], 1);
          variables[{i, 0}] = vec_name;
          continue;
        }
      }
    }
  }

  if (!check_shapes_compile) {
      check_shapes->clear();
  }

  size_t counter = 0;
  for (const auto& entry : g.outputs()) {
    std::string var_name = "output" + std::to_string(counter);
    code += "vector::VectorizedStorage<DType_" + var_name + \
            ", nvec> vec_" + var_name + ";\n";
    ++counter;
  }

  code += "for (int j = 0; j < nvec; j++ ) {\n";


  for (size_t i = 0; i < g.num_nodes(); ++i) {
    const auto& node = g[i];
    const auto* source = node.source;
    if (source != nullptr) {
      std::string var_name = "temp" + std::to_string(temp_name_counter++);
      if (source->is_variable()) {
        if (load_index[i]) {
            code += "const auto " + var_name + " = op::load(vec_" +
                    variables[{i, 0}] + ".scratch_.separate[j]);\n";
            CHECK_EQ(outputs[i], 1);
            variables[{i, 0}] = var_name;
        }
      } else {
        std::string op_name = source->op()->name;
        if (fusion::ops_desc.find(op_name) != fusion::ops_desc.end()) {
          const std::vector<std::vector<std::string>>& op_descs =
            fusion::ops_desc.at(op_name);
          CHECK_EQ(outputs[i], op_descs.size());
          size_t count = 0;
          for (const auto& op_desc : op_descs) {
            var_name = "temp" + std::to_string(temp_name_counter++);
            const std::string& fmt = ParseOpDescription(op_desc, variables, node);
            code += "const auto " + var_name + " = " + fmt + ";\n";
            variables[{i, count}] = var_name;
            ++count;
          }
          continue;
        }

        if (fusion::slice_ops.find(op_name) != fusion::slice_ops.end()) {
          code += "const auto " + var_name +
                  " = op::load(" + variables[{i, 0}] +
                  ".scratch_.separate[j]);\n";
          variables[{i, 0}] = var_name;
          continue;
        }


        // Special cases with variable number
        // of inputs/outputs, listed in
        // fusion::variable_io_ops
        if (op_name == "add_n") {
          CHECK_EQ(outputs[i], 1);
          const auto& arg = variables[{node.inputs[0].node_id, node.inputs[0].index}];
          code += "auto " + var_name + " = " + arg + ";\n";
          for (size_t inp = 1; inp < node.inputs.size(); ++inp) {
            const auto& temp_arg = variables[{node.inputs[inp].node_id, node.inputs[inp].index}];
            code += var_name + " = op::add(" + var_name + ", " + temp_arg + ");\n";
          }
          variables[{i, 0}] = var_name;
          continue;
        }

        if (op_name == "_backward_Activation") {
          CHECK_EQ(outputs[i], 1);
          std::string act_type = node.source->attrs.dict.at("act_type");
          std::string ograd, input;
          ograd = variables[{node.inputs[0].node_id, node.inputs[0].index}];
          if (act_type == "relu" ||
              act_type == "sigmoid" ||
              act_type == "tanh") {
            input = variables[{node.inputs[1].node_id, node.inputs[1].index}];
          } else {
            input = variables[{node.inputs[2].node_id, node.inputs[2].index}];
          }
          code += "const auto " + var_name + " = op::backward_" + act_type +
                  "(" + ograd + ", " + input + ");\n";

          variables[{i, 0}] = var_name;
          continue;
        }

        if (op_name == "amp_multicast" || op_name == "_backward_amp_multicast") {
          CHECK_EQ(outputs[i], node.inputs.size());
          for (size_t counter = 0; counter < outputs[i]; ++counter) {
            const auto& input = node.inputs[counter];
            var_name = "temp" + std::to_string(temp_name_counter++);
            const auto& arg = variables[{input.node_id, input.index}];
            code += "const auto " + var_name + " = " + arg + ";\n";
            variables[{i, counter}] = var_name;
          }
          continue;
        }

        if (op_name == "_backward_cast") {
          CHECK_EQ(outputs[i], 1);
          const int output_type = node_dtypes[g.entry_id(i, 0)];
          const auto& arg = variables[{node.inputs[0].node_id, node.inputs[0].index}];
          code += "const auto " + var_name + " = op::cast<" + mshadowTypeToString(output_type) +
                  ">(" + arg + ");\n";
          variables[{i, 0}] = var_name;
          continue;
        }

        // LeakyReLU, look for act_type
        if (op_name == "LeakyReLU") {
            std::string act_type = node.source->attrs.dict.at("act_type");
            const std::vector<std::vector<std::string>>& op_descs =
                fusion::LeakyReLU_ops.at(act_type);
            if (fusion::LeakyReLU_ops.find(act_type) != fusion::LeakyReLU_ops.end()) {
              CHECK_EQ(outputs[i], op_descs.size());
              size_t count = 0;
              for (const auto& op_desc : op_descs) {
                var_name = "temp" + std::to_string(temp_name_counter++);
                const std::string& fmt = ParseOpDescription(op_desc, variables, node);
                code += "const auto " + var_name + " = " + fmt + ";\n";
                variables[{i, count}] = var_name;
                ++count;
              }
              continue;
            }
        }
        if (op_name == "_backward_LeakyReLU") {
            std::string act_type = node.source->attrs.dict.at("act_type");
            const std::vector<std::vector<std::string>>& op_descs =
                fusion::LeakyReLU_bwd_ops.at(act_type);
            if (fusion::LeakyReLU_ops.find(act_type) != fusion::LeakyReLU_bwd_ops.end()) {
              CHECK_EQ(outputs[i], op_descs.size());
              size_t count = 0;
              for (const auto& op_desc : op_descs) {
                var_name = "temp" + std::to_string(temp_name_counter++);
                const std::string& fmt = ParseOpDescription(op_desc, variables, node);
                code += "const auto " + var_name + " = " + fmt + ";\n";
                variables[{i, count}] = var_name;
                ++count;
              }
              continue;
            }
        }

        LOG(FATAL) << "Unrecognized op " + op_name;
      }
    } else {
      LOG(FATAL) << "Encountered node with NULL source.";
    }
  }

  counter = 0;
  for (const auto& entry : g.outputs()) {
    const std::string& var = variables[{entry.node_id, entry.index}];
    const auto var_name = "output" + std::to_string(counter);
    code += "vec_" + var_name + ".scratch_.separate[j] = op::store("+ var +", " + var_name + ");\n";
    ++counter;
  }

  code += "}\n";

  counter = 0;

  for (const auto& entry : g.outputs()) {
    const std::string& var = variables[{entry.node_id, entry.index}];
    if (req[counter] == kWriteTo || req[counter] == kWriteInplace) {
      const auto var_name = "output" + std::to_string(counter);
      code += "op::store_index(vec_" + var_name + ", i, " + var_name + ", " +
              var_name + "_shape);\n";
    } else if (req[counter] == kAddTo) {
      const auto var_name = "output" + std::to_string(counter);
      code += "op::store_add_index(vec_" + var_name + ", i, " + var_name + ", " +
              var_name + "_shape);\n";
    } else if (req[counter] == kNullOp) {
      // nullptr req, do not do anything
    } else {
      LOG(FATAL) << "Encountered unexpected req.";
    }
    ++counter;
  }

  // Add boilerplate and type information
  std::string kernel_params = "";
  std::string tensor_params = "";
  nnvm::Symbol sym;
  sym.outputs = subgraph_.outputs;
  const std::vector<std::string> input_names = sym.ListInputNames(nnvm::Symbol::kAll);
  size_t num_params = in_dtypes.size() + out_dtypes.size();
  size_t i = 0;
  std::string aux_code = "static const int nvec = " + std::to_string(nvec) + ";\n";

  for (const auto &shape_id : extra_shape_args_) {
      std::string shape_name = "extra_" + std::to_string(shape_id) + "_shape";
      int ndim = node_shapes[shape_id].ndim();
      kernel_params += " const op::Shape<" + std::to_string(ndim) + "> " + shape_name;
      kernel_params += ", ";
  }
  for (const auto &type : in_dtypes) {
    std::string type_name = mshadowTypeToString(type);
    std::string dtype_var = "DType_" + input_names[i];
    std::string dim_var = "ndim_" + input_names[i];
    std::string dim_val = std::to_string(in_ndims[i]);
    aux_code = "using " + dtype_var + " = " + type_name + ";\n" + aux_code;
    aux_code = "static const int " + dim_var + " = " + dim_val + ";\n" + aux_code;
    tensor_params += dtype_var + "* " +input_names[i];
    kernel_params += " const op::Shape<" + dim_val + "> " + input_names[i]+"_shape";
    ++i;
    if (i < num_params) {
      tensor_params += ", ";
    }
    kernel_params += ", ";
  }
  for (const auto &type : out_dtypes) {
    std::string type_name = mshadowTypeToString(type);
    std::string out_name = "output" + std::to_string(i - in_dtypes.size());
    std::string dtype_var = "DType_" + out_name;
    std::string dim_var = "ndim_" + out_name;
    std::string dim_val = std::to_string(out_ndims[i - in_dtypes.size()]);
    aux_code = "static const int " + dim_var + " = " + dim_val + ";\n" + aux_code;
    aux_code = "using " + dtype_var + " = " + type_name + ";\n" + aux_code;
    tensor_params += dtype_var + "* " + out_name;
    kernel_params += " const op::Shape<" + dim_val + "> " + out_name+"_shape";
    ++i;
    if (i < num_params) {
      tensor_params += ", ";
    }
    kernel_params += ", ";
  }
  kernel_params += tensor_params;

  // Create kernel source (minus the common header)
  return aux_code + "\n" +
         "__launch_bounds__(" + std::to_string(FusedOp::NTHREADS) + ")\n" +
         "__global__ void FusedKernel_" + kernel_name +
         "(size_t N, " + kernel_params + ") {\n" +
         fusion::kernel_begin + "\n" +
         code + "\n" +
         fusion::kernel_end;
}

CUfunction FusedOp::CompileCode(const std::string &code,
                                const std::string &kernel_name,
                                int dev_id) {
  return common::cuda::rtc::get_function(code, "FusedKernel_" + kernel_name, "", dev_id);
}


void FusedOp::CheckShapesAndTypes(const std::vector<TBlob> &inputs,
                                  const std::vector<TBlob> &outputs,
                                  std::vector<int> *in_dtypes,
                                  std::vector<int> *in_ndims,
                                  std::vector<int> *out_dtypes,
                                  std::vector<int> *out_ndims,
                                  int *nvec) {
  std::vector<mxnet::TShape> in_shapes;
  std::vector<mxnet::TShape> out_shapes;
  CHECK_EQ(inputs.size(), inputs_.size());
  CHECK_EQ(outputs.size(), outputs_.size());

  for (size_t counter = 0; counter < inputs.size(); ++counter) {
    const auto& blob = inputs[counter];
    in_dtypes->push_back(blob.type_flag_);
    in_ndims->push_back(blob.ndim());
    in_shapes.push_back(blob.shape_);
    initialized_ = initialized_ && blob.type_flag_ == inputs_[counter].dtype;
    initialized_ = initialized_ && blob.ndim() == inputs_[counter].ndim;
    inputs_[counter].dtype = blob.type_flag_;
    inputs_[counter].ndim = blob.ndim();
    *nvec = max(*nvec, mshadowTypeToVectorLength(blob.type_flag_));
  }

  for (size_t counter = 0; counter < outputs.size(); ++counter) {
    const auto& blob = outputs[counter];
    out_dtypes->push_back(blob.type_flag_);
    out_ndims->push_back(blob.ndim());
    out_shapes.push_back(blob.shape_);
    initialized_ = initialized_ && blob.type_flag_ == outputs_[counter].dtype;
    initialized_ = initialized_ && blob.ndim() == outputs_[counter].ndim;
    outputs_[counter].dtype = blob.type_flag_;
    outputs_[counter].ndim = blob.ndim();
    *nvec = max(*nvec, mshadowTypeToVectorLength(blob.type_flag_));
  }

  for (auto it = intermediate_shapes_.begin();
       it != intermediate_shapes_.end();
       ++it) {
    if (it->input_attr == in_shapes && it->output_attr == out_shapes) {
      intermediate_shapes_.erase(intermediate_shapes_.begin(), it);
      break;
    }
  }
  for (auto it = intermediate_dtypes_.begin();
       it != intermediate_dtypes_.end();
       ++it) {
    if (it->input_attr == *in_dtypes && it->output_attr == *out_dtypes) {
      intermediate_dtypes_.erase(intermediate_dtypes_.begin(), it);
      break;
    }
  }
}

template <>
void FusedOp::Forward<gpu>(const nnvm::NodeAttrs& attrs,
                           const OpContext &ctx,
                           const std::vector<TBlob> &inputs,
                           const std::vector<OpReqType> &req,
                           const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  std::lock_guard<std::mutex> lock(my_mutex_);
  CHECK_GE(outputs.size(), 1) << "There needs to be at least 1 output.";

  std::vector<int> in_dtypes;
  std::vector<int> in_ndims;
  std::vector<int> out_dtypes;
  std::vector<int> out_ndims;
  int nvec = 1;

  CheckShapesAndTypes(inputs, outputs, &in_dtypes, &in_ndims,
                      &out_dtypes, &out_ndims, &nvec);

  const auto& node_shapes = intermediate_shapes_[0].internal_attr;
  const auto& node_dtypes = intermediate_dtypes_[0].internal_attr;

  int dev_id = ctx.run_ctx.ctx.dev_id;

  // A change between training and inference modes may require different kernel functions
  initialized_ = initialized_ && (req == saved_reqs_);
  saved_reqs_ = req;

  if (!initialized_) {
    const auto& code = GenerateCode(req, in_dtypes, out_dtypes, in_ndims, out_ndims,
                       node_shapes, node_dtypes, nvec, attrs.name, &check_shape_args_);
    kernel_functions_[fusion::kGeneral] = CompileCode(code, attrs.name, dev_id);
    if (check_shape_args_.size() > 0) {
      const auto& code = GenerateCode(req, in_dtypes, out_dtypes, in_ndims, out_ndims,
                           node_shapes, node_dtypes, nvec, attrs.name, nullptr);
      kernel_functions_[fusion::kShapeOptimized] = CompileCode(code, attrs.name, dev_id);
    }
    initialized_ = true;
    kernel_function_dev_id_ = dev_id;
  }

  // A change in device would force recompiling, but this is unexpected so signal as an error
  if (dev_id != kernel_function_dev_id_)
    LOG(FATAL) << "Fused op compiled for device " << kernel_function_dev_id_
               <<  ", not expecting switch to device " << dev_id;

  Stream<gpu>* s = ctx.get_stream<gpu>();
  std::vector<const void*> args;
  size_t N = 0;
  for (const auto& output : outputs) {
    N = std::max(N, output.shape_.Size());
  }
  N = (N + nvec - 1)/nvec;
  args.push_back(&N);

  unsigned int num_blocks = (N + FusedOp::NTHREADS - 1) / FusedOp::NTHREADS;

  std::vector<void*> ptrs;
  std::vector<std::vector<int>> shapes;

  for (const auto &shape_id : extra_shape_args_) {
    AddShape(node_shapes[shape_id], &shapes);
  }
  for (const auto &data : inputs) {
    AddPointerAndShape(data, &ptrs, &shapes, s);
  }
  for (const auto &data : outputs) {
    AddPointerAndShape(data, &ptrs, &shapes, s);
  }

  for (auto &tensor_shapes : shapes) {
    args.push_back(tensor_shapes.data());
  }
  for (auto &ptr : ptrs) {
    args.push_back(reinterpret_cast<void *>(&ptr));
  }
  int kernel_variant = fusion::kGeneral;
  if (check_shape_args_.size() > 0) {
    kernel_variant = fusion::kShapeOptimized;
      for (const auto &shape_id : check_shape_args_) {
          const auto& shape = node_shapes[shape_id];
          if (shape[shape.ndim()-1] % nvec != 0) {
            kernel_variant = fusion::kGeneral;
          }
      }
  }
  common::cuda::rtc::launch(kernel_functions_[kernel_variant],
                            {num_blocks, 1, 1},
                            {static_cast<unsigned int>(FusedOp::NTHREADS), 1, 1},
                            0, s, &args);
}

void FusedOpForwardGPU(const nnvm::NodeAttrs& attrs,
                    const OpContext &ctx,
                    const std::vector<TBlob> &inputs,
                    const std::vector<OpReqType> &req,
                    const std::vector<TBlob> &outputs) {
  const FusedOpPtr& op = nnvm::get<FusedOpPtr>(attrs.parsed);
  op->Forward<gpu>(attrs, ctx, inputs, req, outputs);
}

NNVM_REGISTER_OP(_FusedOp)
.set_attr<FCompute>("FCompute<gpu>", FusedOpForwardGPU);

}  // namespace mxnet
