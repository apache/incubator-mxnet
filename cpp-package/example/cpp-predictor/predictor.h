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

#pragma once

#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

enum MXDType {
    MX_NONE = -1,
    MX_FLOAT32 = 0,
    MX_FLOAT64 = 1,
    MX_FLOAT16 = 2,
    MX_UINT8 = 3,
    MX_INT32 = 4,
    MX_INT8 = 5,
    MX_INT64 = 6
};

static inline NDArray CopyToCtx(const NDArray &src, const Context &ctx) {
    NDArrayHandle destHandle;

    int dtype = src.GetDType();
    auto shape = src.GetShape();

    if (MXNDArrayCreateEx(shape.data(), shape.size(),
        ctx.GetDeviceType(), ctx.GetDeviceId(),
        false, dtype, &destHandle)) {
        throw std::runtime_error(MXGetLastError());
    }

    NDArray dest(destHandle);
    src.CopyTo(&dest);

    return dest;
}

static inline size_t ElemSize(MXDType dtype) {
    switch (dtype) {
    case MX_UINT8:
    case MX_INT8:
        return 1;

    case MX_FLOAT16:
        return 2;

    case MX_FLOAT32:
    case MX_INT32:
        return 4;

    case MX_FLOAT64:
    case MX_INT64:
        return 8;

    default:
        return 0;
    }
}

static inline bool FileExists(std::string filename) {
    std::ifstream f(filename);
    return f.good();
}

class Predictor {
 public:
    Predictor(const std::string& model_json_file,
        const std::string& model_params_file,
        const Shape& input_shape, const Context &ctx = Context::cpu(),
        MXDType input_dtype = MX_FLOAT32) {
        global_ctx = ctx;

        if (!FileExists(model_json_file)) {
            throw std::runtime_error("Model file does not exist");
        }
        net = Symbol::Load(model_json_file);

        if (!FileExists(model_params_file)) {
            throw std::runtime_error("Model parameters does not exist");
        }

        std::map<std::string, NDArray> parameters;
        NDArray::Load(model_params_file, 0, &parameters);

        std::map<std::string, OpReqType> grad_reqs;
        for (const auto &k : parameters) {
            if (k.first.substr(0, 4) == "aux:") {
                auto name = k.first.substr(4, k.first.size() - 4);
                aux_map[name] = CopyToCtx(k.second, global_ctx);
            }
            if (k.first.substr(0, 4) == "arg:") {
                auto name = k.first.substr(4, k.first.size() - 4);
                args_map[name] = CopyToCtx(k.second, global_ctx);
                grad_reqs[name] = OpReqType::kNullOp;
            }
        }

        NDArrayHandle data_handle;
        if (MXNDArrayCreateEx(input_shape.data(), input_shape.ndim(),
            ctx.GetDeviceType(), ctx.GetDeviceId(),
            false, input_dtype, &data_handle)) {
            throw std::runtime_error(MXGetLastError());
        }

        args_map["data"] = NDArray(data_handle);

        /*WaitAll is need when we copy data between GPU and the main memory*/
        NDArray::WaitAll();

        executor = net.SimpleBind(ctx, args_map,
            std::map<std::string, NDArray>(), grad_reqs, aux_map);
    }

    bool SetInput(NDArray data) {
        if (Shape(data.GetShape()) != input_shape) {
            return false;
        }

        if (args_map["data"].GetDType() != data.GetDType()) {
            return false;
        }

        data.CopyTo(&executor->arg_dict()["data"]);
        return true;
    }

    bool SetInput(void *data, size_t numElem) {
        auto input = executor->arg_dict()["data"];
        if (!data || input.Size() != numElem) {
            return false;
        }

        // we dont know that data type, but it only allow float pointer
        input.SyncCopyFromCPU(reinterpret_cast<mx_float *>(data), numElem);
        return true;
    }

    void Forward(bool waitall = true) {
        if (waitall) {
            NDArray::WaitAll();
        }

        executor->Forward(false);

        if (waitall) {
            NDArray::WaitAll();
        }
    }

    std::vector<Shape> GetOutputShapes() {
        std::vector<Shape> output_shapes;
        for (const auto &output : executor->outputs) {
            output_shapes.push_back(Shape(output.GetShape()));
        }
    }

    Shape GetOutputShape(int index) {
        return Shape(executor->outputs[index].GetShape());
    }

    std::vector<NDArray> GetOutputs() {
        std::vector<NDArray> cpu_outputs;

        for (const auto &output : executor->outputs) {
            cpu_outputs.push_back(CopyToCtx(output, Context::cpu()));
        }

        return cpu_outputs;
    }

    NDArray GetOutput(int index) {
        return CopyToCtx(executor->outputs[index], Context::cpu());
    }

    bool GetOutput(int index, void *data, size_t max_size) {
        size_t elem_size = ElemSize((MXDType)executor->outputs[index].GetDType());
        size_t output_size = executor->outputs[index].Size() * elem_size;

        if (output_size > max_size) {
            return false;
        }

        executor->outputs[index].SyncCopyToCPU(reinterpret_cast<mx_float *>(data));
        return true;
    }

    ~Predictor() {
        if (executor) {
            delete executor;
        }

        MXNotifyShutdown();
    }

 private:
    std::map<std::string, NDArray> args_map;
    std::map<std::string, NDArray> aux_map;

    Symbol net;
    Executor *executor;

    Shape input_shape;

    Context global_ctx = Context::cpu();
};
