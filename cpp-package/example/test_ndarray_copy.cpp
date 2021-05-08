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
 * 
 */
#include <vector>
#include "mxnet/c_api.h"
#include "dmlc/logging.h"
#include "mxnet-cpp/MxNetCpp.h"
using namespace mxnet::cpp;

enum TypeFlag {
  kFloat32 = 0,
  kFloat64 = 1,
  kFloat16 = 2,
  kUint8 = 3,
  kInt32 = 4,
  kInt8  = 5,
  kInt64 = 6,
};

/*
 * The file is used for testing if there exist type inconsistency
 * when using Copy API to create a new NDArray.
 * By running: build/test_ndarray.
 */
int main(int argc, char** argv) {
    std::vector<mx_uint> shape1{128, 2, 32};
    Shape shape2(32, 8, 64);

    int gpu_count = 0;
    if (MXGetGPUCount(&gpu_count) != 0) {
      LOG(ERROR) << "MXGetGPUCount failed";
      return -1;
    }

    Context context = (gpu_count > 0) ? Context::gpu() : Context::cpu();

    NDArray src1(shape1, context, true, kFloat16);
    NDArray src2(shape2, context, false, kInt8);
    NDArray dst1, dst2;
    dst1 = src1.Copy(context);
    dst2 = src2.Copy(context);
    NDArray::WaitAll();
    CHECK_EQ(src1.GetDType(), dst1.GetDType());
    CHECK_EQ(src2.GetDType(), dst2.GetDType());
    return 0;
}
