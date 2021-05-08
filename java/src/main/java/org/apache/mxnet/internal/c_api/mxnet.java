/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mxnet.internal.c_api.presets;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(
    value = {
        @Platform(
            value = {"linux", "macosx", "windows"},
            compiler = "cpp11",
            define = {"DMLC_USE_CXX11 1", "MSHADOW_USE_CBLAS 1", "MSHADOW_IN_CXX11 1", "MSHADOW_USE_CUDA 0", "MSHADOW_USE_F16C 0", "MXNET_USE_TVM_OP 0"},
            include = {"dlpack/dlpack.h", "mxnet/c_api.h", "mxnet/runtime/c_runtime_api.h", "nnvm/c_api.h"},
            link = "mxnet",
            linkpath = {"/usr/lib64/", "/usr/lib/", "/usr/local/lib/"},
            preload = {"gfortran@.5", "gfortran@.4", "gfortran@.3"},
            preloadpath = {"/usr/local/lib/gcc/10/", "/usr/local/lib/gcc/9/", "/usr/local/lib/gcc/8/",
                           "/usr/local/lib/gcc/7/", "/usr/local/lib/gcc/6/", "/usr/local/lib/gcc/5/"}
        ),
        @Platform(
            value = "windows",
            link = "libmxnet",
            preload = {"libwinpthread-1", "libgcc_s_seh-1", "libgfortran-5", "libgfortran-4", "libgfortran-3", "libopenblas"},
            preloadpath = "C:/msys64/mingw64/bin/"
        )
    },
    target = "org.apache.mxnet.internal.c_api",
    global = "org.apache.mxnet.internal.c_api.global.mxnet"
)
public class mxnet implements InfoMapper {

    public void map(InfoMap infoMap) {
        infoMap.put(new Info("MSHADOW_USE_F16C", "MXNET_USE_TVM_OP").define(false))
               .put(new Info("DLPACK_EXTERN_C", "DLPACK_DLL", "MXNET_EXTERN_C", "MXNET_DLL", "NNVM_DLL", "MXNDArrayCreateEx").cppTypes().annotations())
               .put(new Info("MXNDArrayCreateSparseEx64", "MXNDArrayGetAuxNDArray64", "MXNDArrayGetAuxType64",
                             "MXNDArrayGetShape64", "MXSymbolInferShape64", "MXSymbolInferShapePartial64").skip())
               .put(new Info("NDArrayHandle", "FunctionHandle", "AtomicSymbolCreator", "CachedOpHandle", "SymbolHandle", "AtomicSymbolHandle", "ExecutorHandle",
                             "DataIterCreator", "DataIterHandle", "DatasetCreator", "DatasetHandle", "BatchifyFunctionCreator", "BatchifyFunctionHandle","KVStoreHandle",
                             "RecordIOHandle", "RtcHandle", "CudaModuleHandle", "CudaKernelHandle", "ProfileHandle", "DLManagedTensorHandle", "ContextHandle",
                             "EngineFnPropertyHandle", "EngineVarHandle", "MXNetFunctionHandle", "MXNetObjectHandle", "OpHandle", "SymbolHandle", "GraphHandle",
                             "OptimizerCreator", "OptimizerHandle", "PredictorHandle").cast().javaNames("Pointer").valueTypes("Pointer").pointerTypes("PointerPointer"));
    }
}
