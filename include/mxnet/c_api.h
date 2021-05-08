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
 *  Copyright (c) 2015 by Contributors
 * \file c_api.h
 * \brief C API of mxnet
 */
#ifndef MXNET_C_API_H_
#define MXNET_C_API_H_

/*! \brief Inhibit C++ name-mangling for MXNet functions. */
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/*! \brief Keep the default value in C++ */
#ifdef __cplusplus
#define DEFAULT(x) = x
#else
#define DEFAULT(x)
#endif  // __cplusplus

#include <stdint.h>

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/*! \brief MXNET_DLL prefix for windows */
#ifdef _WIN32
#ifdef MXNET_EXPORTS
#define MXNET_DLL __declspec(dllexport)
#else
#define MXNET_DLL __declspec(dllimport)
#endif
#else
#define MXNET_DLL
#endif

/*! \brief manually define unsigned int */
typedef uint32_t mx_uint;
/*! \brief manually define float */
typedef float mx_float;
/*! \brief data type to store dim size */
typedef int64_t dim_t;
// all the handles are simply void *
// will be casted internally to specific pointers types
// these typedefs are mainly used for readablity reasons
/*! \brief handle to NDArray */
typedef void *NDArrayHandle;
/*! \brief handle to a mxnet narray function that changes NDArray */
typedef const void *FunctionHandle;
/*! \brief handle to a function that takes param and creates symbol */
typedef void *AtomicSymbolCreator;
/*! \brief handle to cached operator */
typedef void *CachedOpHandle;
/*! \brief handle to a symbol that can be bind as operator */
typedef void *SymbolHandle;
/*! \brief handle to a AtomicSymbol */
typedef void *AtomicSymbolHandle;
/*! \brief handle to an Executor */
typedef void *ExecutorHandle;
/*! \brief handle a dataiter creator */
typedef void *DataIterCreator;
/*! \brief handle to a DataIterator */
typedef void *DataIterHandle;
/*! \brief handle a dataset creator */
typedef void *DatasetCreator;
/*! \brief handle to a Dataset */
typedef void *DatasetHandle;
/*! \brief handle to a BatchifyFunction creator*/
typedef void *BatchifyFunctionCreator;
/*! \brief handle to a BatchifyFunction */
typedef void *BatchifyFunctionHandle;
/*! \brief handle to KVStore */
typedef void *KVStoreHandle;
/*! \brief handle to RecordIO */
typedef void *RecordIOHandle;
/*! \brief handle to MXRtc*/
typedef void *RtcHandle;
/*! \brief handle to rtc cuda module*/
typedef void *CudaModuleHandle;
/*! \brief handle to rtc cuda kernel*/
typedef void *CudaKernelHandle;
/*! \brief handle to a Profile object (domain, duration, counter, etc.) */
typedef void *ProfileHandle;
/*! \brief handle to DLManagedTensor*/
typedef void *DLManagedTensorHandle;
/*! \brief handle to Context */
typedef const void *ContextHandle;
/*! \brief handle to Engine FnProperty */
typedef const void *EngineFnPropertyHandle;
/*! \brief handle to Engine VarHandle */
typedef void *EngineVarHandle;

/*! \brief Engine asynchronous operation */
typedef void (*EngineAsyncFunc)(void*, void*, void*);
/*! \brief Engine synchronous operation */
typedef void (*EngineSyncFunc)(void*, void*);
/*! \brief Callback to free the param for EngineAsyncFunc/EngineSyncFunc */
typedef void (*EngineFuncParamDeleter)(void*);
/*! \brief Monitor callback called at operator level for cached op */
typedef void (*CachedOpMonitorCallback)(const char*,
                                        const char*,
                                        NDArrayHandle);


struct NativeOpInfo {
  void (*forward)(int, float**, int*, unsigned**, int*, void*);
  void (*backward)(int, float**, int*, unsigned**, int*, void*);
  void (*infer_shape)(int, int*, unsigned**, void*);
  void (*list_outputs)(char***, void*);
  void (*list_arguments)(char***, void*);
  // all functions also pass a payload void* pointer
  void* p_forward;
  void* p_backward;
  void* p_infer_shape;
  void* p_list_outputs;
  void* p_list_arguments;
};

struct NDArrayOpInfo {
  bool (*forward)(int, void**, int*, void*);
  bool (*backward)(int, void**, int*, void*);
  bool (*infer_shape)(int, int*, unsigned**, void*);
  bool (*list_outputs)(char***, void*);
  bool (*list_arguments)(char***, void*);
  bool (*declare_backward_dependency)(const int*, const int*, const int*,
                                      int*, int**, void*);
  // all functions also pass a payload void* pointer
  void* p_forward;
  void* p_backward;
  void* p_infer_shape;
  void* p_list_outputs;
  void* p_list_arguments;
  void* p_declare_backward_dependency;
};

typedef int (*MXGenericCallback)(void);

struct MXCallbackList {
  int num_callbacks;
  int (**callbacks)(void);
  void **contexts;
};

struct LibFeature {
  const char* name;
  bool enabled;
};

enum CustomOpCallbacks {
  kCustomOpDelete,
  kCustomOpForward,
  kCustomOpBackward
};

enum CustomOpPropCallbacks {
  kCustomOpPropDelete,
  kCustomOpPropListArguments,
  kCustomOpPropListOutputs,
  kCustomOpPropListAuxiliaryStates,
  kCustomOpPropInferShape,
  kCustomOpPropDeclareBackwardDependency,
  kCustomOpPropCreateOperator,
  kCustomOpPropInferType,
  kCustomOpPropInferStorageType,
  kCustomOpPropBackwardInferStorageType
};


typedef int (*CustomOpFBFunc)(int /*size*/, void** /*ptrs*/, int* /*tags*/,
                              const int* /*reqs*/, const int /*is_train*/,
                              void* /*state*/);
typedef int (*CustomOpDelFunc)(void* /*state*/);
typedef int (*CustomOpListFunc)(char*** /*args*/, void* /*state*/);
typedef int (*CustomOpInferShapeFunc)(int /*num_input*/, int* /*ndims*/,
                                      int** /*shapes*/, void* /*state*/);
typedef int (*CustomOpInferStorageTypeFunc)(int /*num_input*/, int* /*stypes*/, void* /*state*/);
typedef int (*CustomOpBackwardInferStorageTypeFunc)(int /*num_input*/,
                                                    int * /*stypes*/,
                                                    int * /*tags*/,
                                                    void * /*state*/);
typedef int (*CustomOpInferTypeFunc)(int /*num_input*/, int* /*types*/, void* /*state*/);
typedef int (*CustomOpBwdDepFunc)(const int* /*out_grad*/, const int* /*in_data*/,
                                  const int* /*out_data*/, int* /*num_deps*/,
                                  int** /*rdeps*/, void* /*state*/);
typedef int (*CustomOpCreateFunc)(const char* /*ctx*/, int /*num_inputs*/,
                                  unsigned** /*shapes*/, const int* /*ndims*/,
                                  const int* /*dtypes*/, struct MXCallbackList* /*ret*/,
                                  void* /*state*/);
typedef int (*CustomOpPropCreator)(const char* /*op_type*/, const int /*num_kwargs*/,
                                   const char** /*keys*/, const char** /*values*/,
                                   struct MXCallbackList* /*ret*/);


enum CustomFunctionCallbacks {
  kCustomFunctionBackward,
  kCustomFunctionDelete
};

typedef int (*CustomFunctionBwdFunc)(int /*num_ograds*/, int /*num_igrads*/, void** /*ptrs*/,
                                     const int* /*reqs*/, const int /*is_train*/,
                                     void* /*state*/);
typedef int (*CustomFunctionDelFunc)(void* /*state*/);

/*!
 * \brief return str message of the last error
 *  all function in this file will return 0 when success
 *  and -1 when an error occured,
 *  MXGetLastError can be called to retrieve the error
 *
 *  this function is threadsafe and can be called by different thread
 *  \return error info
 */
MXNET_DLL const char *MXGetLastError();

//-------------------------------------
// Part 0: Global State setups
//-------------------------------------

/*!
 * \brief Load library dynamically
 * \param path to the library .so file
 * \param 0 for quiet, 1 for verbose
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXLoadLib(const char *path, unsigned verbose, void** lib);

/*!
 * \brief Get list of features supported on the runtime
 * \param libFeature pointer to array of LibFeature
 * \param size of the array
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXLibInfoFeatures(const struct LibFeature **libFeature, size_t *size);

/*!
 * \brief return whether the mxnet library is compiled with cxx11 abi
 * \return whether mxnet is built with cxx11 abi
 */
MXNET_DLL int MXLibInfoCompiledWithCXX11ABI(int* result);

/*!
 * \brief Seed all global random number generators in mxnet.
 * \param seed the random number seed.
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXRandomSeed(int seed);

/*!
 * \brief Seed the global random number generator of the given device.
 * \param seed the random number seed.
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXRandomSeedContext(int seed, int dev_type, int dev_id);

/*!
 * \brief Notify the engine about a shutdown,
 *  This can help engine to print less messages into display.
 *
 *  User do not have to call this function.
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXNotifyShutdown();

/*!
 * \brief Set up configuration of profiler for the process passed as profile_process in keys
 * \param num_params Number of parameters
 * \param keys array of parameter keys
 * \param vals array of parameter values
 * \param kvstoreHandle handle to kvstore
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXSetProcessProfilerConfig(int num_params, const char* const* keys,
                                         const char* const* vals,
                                         KVStoreHandle kvstoreHandle);

/*!
 * \brief Set up configuration of profiler for worker/current process
 * \param num_params Number of parameters
 * \param keys array of parameter keys
 * \param vals array of parameter values
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXSetProfilerConfig(int num_params, const char* const* keys, const char* const* vals);

/*!
 * \brief Set up state of profiler for either worker or server process
 * \param state indicate the working state of profiler,
 *  profiler not running when state == 0,
 *  profiler running when state == 1
 * \param profile_process an int,
 * when 0 command is for worker/current process,
 * when 1 command is for server process
 * \param kvstoreHandle handle to kvstore, needed for server process profiling
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXSetProcessProfilerState(int state, int profile_process,
                                        KVStoreHandle kvStoreHandle);

/*!
 * \brief Set up state of profiler for current process
 * \param state indicate the working state of profiler,
 *  profiler not running when state == 0,
 *  profiler running when state == 1
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXSetProfilerState(int state);

/*!
 * \brief Set the scope of profiler for current process
 * \param scope indicate the working scope of profiler
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXSetProfilerScope(const char* scope);

/*!
 * \brief Save profile and stop profiler
 * \param finished true if stat output should stop after this point
 * \param profile_process an int,
 * when 0 command is for worker/current process,
 * when 1 command is for server process
 * \param kvstoreHandle handle to kvstore
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXDumpProcessProfile(int finished, int profile_process, KVStoreHandle kvStoreHandle);


/*!
 * \brief Save profile and stop profiler for worker/current process
 * \param finished true if stat output should stop after this point
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXDumpProfile(int finished);

/*!
 * \brief Print sorted aggregate stats to the a string
 *        How aggregate stats are stored will not change
 * \param out_str will receive a pointer to the output string
 * \param reset clear the aggregate stats after printing
 * \param format whether to return in tabular or json format
 * \param sort_by sort by total, avg, min, max, or count
 * \param ascending whether to sort ascendingly
 * \return 0 when success, -1 when failure happens.
 * \note
 */
MXNET_DLL int MXAggregateProfileStatsPrint(const char **out_str, int reset, int format,
                                           int sort_by, int ascending);

/*!
 * \brief Pause profiler tuning collection
 * \param paused If nonzero, profiling pauses. Otherwise, profiling resumes/continues
 * \param profile_process integer which denotes whether to process worker or server process
 * \param kvstoreHandle handle to kvstore
 * \return 0 when success, -1 when failure happens.
 * \note pausing and resuming is global and not recursive
 */
MXNET_DLL int MXProcessProfilePause(int paused, int profile_process, KVStoreHandle kvStoreHandle);

/*!
 * \brief Pause profiler tuning collection for worker/current process
 * \param paused If nonzero, profiling pauses. Otherwise, profiling resumes/continues
 * \return 0 when success, -1 when failure happens.
 * \note pausing and resuming is global and not recursive
 */
MXNET_DLL int MXProfilePause(int paused);

/*!
 * \brief Create profiling domain
 * \param domain String representing the domain name to create
 * \param out Return domain object
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileCreateDomain(const char *domain, ProfileHandle *out);

/*!
 * \brief Create profile task
 * \param name Name of the task
 * \param domain Domain of the task
 * \param out Output handle
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileCreateTask(ProfileHandle domain,
                                  const char *task_name,
                                  ProfileHandle *out);

/*!
 * \brief Create profile frame
 * \param name Name of the frame
 * \param domain Domain of the frame
 * \param out Output handle
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileCreateFrame(ProfileHandle domain,
                                   const char *frame_name,
                                   ProfileHandle *out);

/*!
 * \brief Create profile event
 * \param name Name of the event
 * \param out Output handle
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileCreateEvent(const char *event_name, ProfileHandle *out);

/*!
 * \brief Create profile counter
 * \param name Name of the counter
 * \param domain Domain of the counter
 * \param out Output handle
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileCreateCounter(ProfileHandle domain,
                                     const char *counter_name,
                                     ProfileHandle *out);

/*!
 * \brief Destroy a frame
 * \param frame_handle Handle to frame to destroy
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileDestroyHandle(ProfileHandle frame_handle);

/*!
 * \brief Start timing the duration of a profile duration object such as an event, task or frame
 * \param duration_handle handle to the duration object
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileDurationStart(ProfileHandle duration_handle);

/*!
 * \brief Stop timing the duration of a profile duration object such as an event, task or frame
 * \param duration_handle handle to the duration object
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileDurationStop(ProfileHandle duration_handle);

/*!
 * \brief Set a counter, given its handle
 * \param counter_handle Handle to counter to set
 * \param value Value to set the counter to (64-bit unsigned integer)
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileSetCounter(ProfileHandle counter_handle, uint64_t value);

/*!
 * \brief Adjust a counter by the given amount, given its handle
 * \param counter_handle Handle to counter to adjust
 * \param value Value to adjust the counter by (64-bit signed integer)
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileAdjustCounter(ProfileHandle counter_handle, int64_t value);

/*!
 * \brief Mark a single instant in time
 * \param domain Domain of the marker
 * \param instant_marker_name Name of the marker
 * \param scope Scope of marker ('global', 'process', 'thread', 'task', 'marker')
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXProfileSetMarker(ProfileHandle domain,
                                 const char *instant_marker_name,
                                 const char *scope);

/*!
 * \brief Set the number of OMP threads to use
 * \param thread_num Number of OMP threads desired
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXSetNumOMPThreads(int thread_num);

/*!
 * \brief set bulk execution limit
 * \param bulk_size new bulk_size
 * \param prev_bulk_size previous bulk_size
 */
MXNET_DLL int MXEngineSetBulkSize(int bulk_size, int* prev_bulk_size);

/*!
 * \brief Get the number of GPUs.
 * \param pointer to int that will hold the number of GPUs available.
 * \return 0 when success, -1 when failure happens.
 */
MXNET_DLL int MXGetGPUCount(int* out);

/*!
 * \brief get the free and total available memory on a GPU
 *  Note: Deprecated, use MXGetGPUMemoryInformation64 instead.
 * \param dev the GPU number to query
 * \param free_mem pointer to the integer holding free GPU memory
 * \param total_mem pointer to the integer holding total GPU memory
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXGetGPUMemoryInformation(int dev, int *free_mem, int *total_mem);

/*!
 * \brief get the free and total available memory on a GPU
 * \param dev the GPU number to query
 * \param free_mem pointer to the uint64_t holding free GPU memory
 * \param total_mem pointer to the uint64_t holding total GPU memory
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXGetGPUMemoryInformation64(int dev, uint64_t *free_mem, uint64_t *total_mem);

/*!
 * \brief get the MXNet library version as an integer
 * \param pointer to the integer holding the version number
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXGetVersion(int *out);

/*!
 * \brief Load TVM operator from the binary library
 * \param libpath TVM operators lib file
 * \return 0 when success, -1 when failure happens
 */
#if MXNET_USE_TVM_OP
MXNET_DLL int MXLoadTVMOp(const char *libpath);

struct OtherOptionEntity {
  int val;
};

struct OtherOptionSpace {
  OtherOptionEntity* entities;
  int entities_size;
};

struct ConfigSpace {
  int entity_map_size;
  char** entity_map_key;
  OtherOptionEntity* entity_map_val;
  int space_map_size;
  char** space_map_key;
  OtherOptionSpace* space_map_val;
};

typedef struct ConfigSpaces {
  int spaces_size;
  char** spaces_key;
  ConfigSpace* spaces_val;
} ConfigSpaces;

MXNET_DLL int MXLoadTVMConfig(ConfigSpaces config);
#endif  // MXNET_USE_TVM_OP


//-------------------------------------
// Part 1: NDArray creation and deletion
//-------------------------------------
/*!
 * \brief create a NDArray handle that is not initialized
 *  can be used to pass in as mutate variables
 *  to hold the result of NDArray
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayCreateNone(NDArrayHandle *out);

/*!
 * \brief create a NDArray with specified shape and data type
 *  This api is available when MXNet is built with flag
 *  USE_INT64_TENSOR_SIZE=0 (by default)
 * \param shape the pointer to the shape
 * \param ndim the dimension of the shape
 * \param dev_type device type, specify device we want to take
 * \param dev_id the device id of the specific device
 * \param delay_alloc whether to delay allocation until
 *    the narray is first mutated
 * \param dtype data type of created array
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayCreate(const uint32_t *shape,
                              uint32_t ndim,
                              int dev_type,
                              int dev_id,
                              int delay_alloc,
                              int dtype,
                              NDArrayHandle *out);
#define MXNDArrayCreateEx MXNDArrayCreate  // backward compatibility for external deps

/*!
 * \brief create a NDArray with specified shape and data type
 *  This api is available when MXNet is built with flag
 *  USE_INT64_TENSOR_SIZE=1 (not default) i.e. Large Tensor Support
 * \param shape the pointer to int64_t shape
 * \param ndim the dimension of the shape
 * \param dev_type device type, specify device we want to take
 * \param dev_id the device id of the specific device
 * \param delay_alloc whether to delay allocation until
 *    the narray is first mutated
 * \param dtype data type of created array
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayCreate64(const int64_t *shape,
                                int ndim,
                                int dev_type,
                                int dev_id,
                                int delay_alloc,
                                int dtype,
                                NDArrayHandle *out);

/*!
 * \brief create an empty sparse NDArray with specified shape and data type
 *  This api is available when MXNet is built with flag
 *  USE_INT64_TENSOR_SIZE=0 (by default)
 * \param storage_type the storage type of the ndarray
 * \param shape the pointer to the shape
 * \param ndim the dimension of the shape
 * \param dev_type device type, specify device we want to take
 * \param dev_id the device id of the specific device
 * \param delay_alloc whether to delay allocation until
 *        the narray is first mutated
 * \param dtype data type of created array
 * \param num_aux the number of aux data to support this ndarray
 * \param aux_type data type of the aux data for the created array
 * \param aux_ndims the dimension of the shapes of aux data
 * \param aux_shape the shapes of aux data
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayCreateSparseEx(int storage_type,
                                      const uint32_t *shape,
                                      uint32_t ndim,
                                      int dev_type,
                                      int dev_id,
                                      int delay_alloc,
                                      int dtype,
                                      uint32_t num_aux,
                                      int *aux_type,
                                      uint32_t *aux_ndims,
                                      const uint32_t *aux_shape,
                                      NDArrayHandle *out);

/*!
 * \brief create an empty sparse NDArray with specified shape and data type
 *  This api is available when MXNet is built with flag
 *  USE_INT64_TENSOR_SIZE=1 (not default) i.e. Large Tensor Support
 * \param storage_type the storage type of the ndarray
 * \param shape the pointer to the shape
 * \param ndim the dimension of the shape
 * \param dev_type device type, specify device we want to take
 * \param dev_id the device id of the specific device
 * \param delay_alloc whether to delay allocation until
 *        the narray is first mutated
 * \param dtype data type of created array
 * \param num_aux the number of aux data to support this ndarray
 * \param aux_type data type of the aux data for the created array
 * \param aux_ndims the dimension of the shapes of aux data
 * \param aux_shape the shapes of aux data
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayCreateSparseEx64(int storage_type,
                                        const int64_t *shape,
                                        int ndim,
                                        int dev_type,
                                        int dev_id,
                                        int delay_alloc,
                                        int dtype,
                                        uint32_t num_aux,
                                        int *aux_type,
                                        int *aux_ndims,
                                        const int64_t *aux_shape,
                                        NDArrayHandle *out);

/*!
 * \brief create a NDArray handle that is loaded from raw bytes.
 * \param buf the head of the raw bytes
 * \param size size of the raw bytes
 * \param out the returning handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayLoadFromRawBytes(const void *buf,
                                        size_t size,
                                        NDArrayHandle *out);
/*!
 * \brief save the NDArray into raw bytes.
 * \param handle the NDArray handle
 * \param out_size size of the raw bytes
 * \param out_buf the head of returning memory bytes.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArraySaveRawBytes(NDArrayHandle handle,
                                    size_t *out_size,
                                    const char **out_buf);
/*!
 * \brief Save list of narray into the file.
 * \param fname name of the file.
 * \param num_args number of arguments to save.
 * \param args the array of NDArrayHandles to be saved.
 * \param keys the name of the NDArray, optional, can be NULL
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayLegacySave(const char* fname,
                                  uint32_t num_args,
                                  NDArrayHandle* args,
                                  const char** keys);
/*!
 * \brief Save list of narray into the file.
 * \param fname name of the file.
 * \param num_args number of arguments to save.
 * \param args the array of NDArrayHandles to be saved.
 * \param keys the name of the NDArray, optional, can be NULL
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArraySave(const char* fname,
                            uint32_t num_args,
                            NDArrayHandle* args,
                            const char** keys);
/*!
 * \brief Load list of narray from the file.
 * \param fname name of the file.
 * \param out_size number of narray loaded.
 * \param out_arr head of the returning narray handles.
 * \param out_name_size size of output name arrray.
 * \param out_names the names of returning NDArrays, can be NULL
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayLoad(const char* fname,
                            uint32_t *out_size,
                            NDArrayHandle** out_arr,
                            uint32_t *out_name_size,
                            const char*** out_names);

/*!
 * \brief Load list / dictionary of narrays from file content loaded into memory.
 * This will load a list of ndarrays in a similar
 * manner to MXNDArrayLoad, however, it loads from
 * buffer containing the contents of a file, rather than
 * from a specified file.
 * \param ndarray_buffer pointer to the start of the ndarray file content
 * \param size size of the file
 * \param out_size number of narray loaded.
 * \param out_arr head of the returning narray handles.
 * \param out_name_size size of output name arrray.
 * \param out_names the names of returning NDArrays, can be NULL
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayLoadFromBuffer(const void *ndarray_buffer,
                                      size_t size,
                                      uint32_t *out_size,
                                      NDArrayHandle** out_arr,
                                      uint32_t *out_name_size,
                                      const char*** out_names);

/*!
 * \brief Perform a synchronize copy from a contiguous CPU memory region.
 *
 *  This function will call WaitToWrite before the copy is performed.
 *  This is useful to copy data from existing memory region that are
 *  not wrapped by NDArray(thus dependency not being tracked).
 *
 * \param handle the NDArray handle
 * \param data the data source to copy from.
 * \param size the memory size we want to copy from.
 */
MXNET_DLL int MXNDArraySyncCopyFromCPU(NDArrayHandle handle,
                                       const void *data,
                                       size_t size);
/*!
 * \brief Perform a synchronize copyto a contiguous CPU memory region.
 *
 *  This function will call WaitToRead before the copy is performed.
 *  This is useful to copy data from existing memory region that are
 *  not wrapped by NDArray(thus dependency not being tracked).
 *
 * \param handle the NDArray handle
 * \param data the data source to copy into.
 * \param size the memory size we want to copy into.
 */
MXNET_DLL int MXNDArraySyncCopyToCPU(NDArrayHandle handle,
                                     void *data,
                                     size_t size);

/*!
 * \brief Copy src.data() to dst.data() if i = -1, else dst.aux_data(i) if i >= 0
 * This function blocks. Do not use it in performance critical code.
 * \param handle_dst handle of a dst ndarray whose data/aux_data has been allocated
 * \param handle_src handle of a src ndarray which has default storage type
 * \param i dst data blob indicator
 */
MXNET_DLL int MXNDArraySyncCopyFromNDArray(NDArrayHandle handle_dst,
                                           const NDArrayHandle handle_src,
                                           const int i);

/*!
 * \brief check whether the NDArray format is valid
 * \param full_check if `True`, rigorous check, O(N) operations
 *    Otherwise basic check, O(1) operations
 */
MXNET_DLL int MXNDArraySyncCheckFormat(NDArrayHandle handle, const bool full_check);

/*!
 * \brief Wait until all the pending writes with respect NDArray are finished.
 *  Always call this before read data out synchronizely.
 * \param handle the NDArray handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayWaitToRead(NDArrayHandle handle);

/*!
 * \brief Wait until all the pending read/write with respect NDArray are finished.
 *  Always call this before write data into NDArray synchronizely.
 * \param handle the NDArray handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayWaitToWrite(NDArrayHandle handle);

/*!
 * \brief wait until all delayed operations in
 *   the system is completed
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayWaitAll();

/*!
 * \brief free the narray handle
 * \param handle the handle to be freed
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayFree(NDArrayHandle handle);

/*!
 * \brief Slice the NDArray along axis 0.
 *  This api is available when MXNet is built with flag
 *  USE_INT64_TENSOR_SIZE=0 (by default)
 * \param handle the handle to the NDArray
 * \param slice_begin The beginning index of slice
 * \param slice_end The ending index of slice
 * \param out The NDArrayHandle of sliced NDArray
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArraySlice(NDArrayHandle handle,
                             uint32_t slice_begin,
                             uint32_t slice_end,
                             NDArrayHandle *out);

/*!
 * \brief Slice the NDArray along axis 0.
 *  This api is available when MXNet is built with flag
 *  USE_INT64_TENSOR_SIZE=1 (not default) i.e. Large Tensor Support
 * \param handle the handle to the NDArray
 * \param slice_begin The beginning index of slice
 * \param slice_end The ending index of slice
 * \param out The NDArrayHandle of sliced NDArray
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArraySlice64(NDArrayHandle handle,
                               int64_t slice_begin,
                               int64_t slice_end,
                               NDArrayHandle *out);

/*!
 * \brief Index the NDArray along axis 0.
 *  This api is available when MXNet is built with flag
 *  USE_INT64_TENSOR_SIZE=0 (by default)
 * \param handle the handle to the NDArray
 * \param idx the index
 * \param out The NDArrayHandle of output NDArray
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayAt(NDArrayHandle handle,
                          uint32_t idx,
                          NDArrayHandle *out);

/*!
 * \brief Index the NDArray along axis 0.
 *  This api is available when MXNet is built with flag
 *  USE_INT64_TENSOR_SIZE=1 (not default) i.e. Large Tensor Support
 * \param handle the handle to the NDArray
 * \param idx the index
 * \param out The NDArrayHandle of output NDArray
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayAt64(NDArrayHandle handle,
                            int64_t idx,
                            NDArrayHandle *out);

/*!
 * \brief get the storage type of the array
 */
MXNET_DLL int MXNDArrayGetStorageType(NDArrayHandle handle,
                                      int *out_storage_type);

/*!
 * \brief Reshape the NDArray.
 * \param handle the handle to the narray
 * \param ndim number of dimensions of new shape
 * \param dims new shape
 * \param out the NDArrayHandle of reshaped NDArray
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayReshape(NDArrayHandle handle,
                               int ndim,
                               int *dims,
                               NDArrayHandle *out);

/*!
 * \brief Reshape the NDArray.
 * \param handle the handle to the narray
 * \param ndim number of dimensions of new shape
 * \param dims new shape
 * \param out the NDArrayHandle of reshaped NDArray
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayReshape64(NDArrayHandle handle,
                                 int ndim,
                                 dim_t *dims,
                                 bool reverse,
                                 NDArrayHandle *out);

/*!
 * \brief get the shape of the array
 *  This api is available when MXNet is built with flag
 *  USE_INT64_TENSOR_SIZE=0 (by default)
 * \param handle the handle to the narray
 * \param out_dim the output dimension
 * \param out_pdata pointer holder to get data pointer of the shape
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayGetShape(NDArrayHandle handle,
                                int *out_dim,
                                const int **out_pdata);

/*!
 * \brief get the shape of the array
 *  This api is available when MXNet is built with flag
 *  USE_INT64_TENSOR_SIZE=1 (not default) i.e. Large Tensor Support
 * \param handle the handle to the narray
 * \param out_dim the output dimension
 * \param out_pdata pointer holder to get data pointer of the shape
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayGetShape64(NDArrayHandle handle,
                                  int *out_dim,
                                  const int64_t **out_pdata);

/*!
 * \brief get the content of the data in NDArray
 * \param handle the handle to the ndarray
 * \param out_pdata pointer holder to get pointer of data
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayGetData(NDArrayHandle handle,
                               void **out_pdata);
/*!
* \brief Create a reference view of NDArray that
*  represents as DLManagedTensor
*  Notice: MXNet uses asynchronous execution. Please call MXNDArrayWaitToRead or
*          MXNDArrayWaitToWrite before calling MXNDArrayToDLPack.
* \param handle the handle to the ndarray
* \param out_dlpack pointer holder to get pointer of DLManagedTensor
* \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXNDArrayToDLPack(NDArrayHandle handle,
                                       DLManagedTensorHandle *out_dlpack);

/*!
* \brief Create a NDArray backed by a dlpack tensor.
*
* This allows us to create a NDArray using the memory
* allocated by an external deep learning framework
* that is DLPack compatible.
*
* The memory is retained until the NDArray went out of scope.
*
* \param dlpack the pointer of the input DLManagedTensor
* \param transient_handle whether the handle will be destructed before calling the deleter
* \param out_handle pointer holder to get pointer of NDArray
* \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXNDArrayFromDLPack(DLManagedTensorHandle dlpack,
                                  const bool transient_handle,
                                  NDArrayHandle *out_handle);

/*!
 * \brief Delete a dlpack tensor
 * \param dlpack the pointer of the input DLManagedTensor
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayCallDLPackDeleter(DLManagedTensorHandle dlpack);

/*!
 * \brief get the type of the data in NDArray
 * \param handle the handle to the narray
 * \param out_dtype pointer holder to get type of data
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayGetDType(NDArrayHandle handle,
                               int *out_dtype);

/*!
 * \brief get the type of the ith aux data in NDArray
 *  This api is available when MXNet is built with flag
 *  USE_INT64_TENSOR_SIZE=0 (by default)
 * \param handle the handle to the narray
 * \param i the index of the aux data
 * \param out_type pointer holder to get type of aux data
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayGetAuxType(NDArrayHandle handle,
                                  uint32_t i,
                                  int *out_type);

/*!
 * \brief get the type of the ith aux data in NDArray
 *  This api is available when MXNet is built with flag
 *  USE_INT64_TENSOR_SIZE=1 (not default) i.e. Large Tensor Support
 * \param handle the handle to the narray
 * \param i the index of the aux data
 * \param out_type pointer holder to get type of aux data
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayGetAuxType64(NDArrayHandle handle,
                                    int64_t i,
                                    int *out_type);

/*!
 * \brief Get a deep copy of the ith aux data blob
 *  This api is available when MXNet is built with flag
 *  USE_INT64_TENSOR_SIZE=0 (by default)
 * in the form of an NDArray of default storage type.
 * This function blocks. Do not use it in performance critical code.
 */
MXNET_DLL int MXNDArrayGetAuxNDArray(NDArrayHandle handle,
                                     uint32_t i,
                                     NDArrayHandle *out);

/*!
 * \brief Get a deep copy of the ith aux data blob
 *  This api is available when MXNet is built with flag
 *  USE_INT64_TENSOR_SIZE=1 (not default) i.e. Large Tensor Support
 * in the form of an NDArray of default storage type.
 * This function blocks. Do not use it in performance critical code.
 */
MXNET_DLL int MXNDArrayGetAuxNDArray64(NDArrayHandle handle,
                                       int64_t i,
                                       NDArrayHandle *out);

/*!
 * \brief Get a deep copy of the data blob
 * in the form of an NDArray of default storage type.
 * This function blocks. Do not use it in performance critical code.
 */
MXNET_DLL int MXNDArrayGetDataNDArray(NDArrayHandle handle,
                                      NDArrayHandle *out);
/*!
 * \brief get the context of the NDArray
 * \param handle the handle to the narray
 * \param out_dev_type the output device type
 * \param out_dev_id the output device id
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayGetContext(NDArrayHandle handle,
                                  int *out_dev_type,
                                  int *out_dev_id);
/*!
 * \brief return gradient buffer attached to this NDArray
 * \param handle NDArray handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayGetGrad(NDArrayHandle handle, NDArrayHandle *out);
/*!
 * \brief detach and ndarray from computation graph by clearing entry_
 * \param handle NDArray handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayDetach(NDArrayHandle handle, NDArrayHandle *out);
/*!
 * \brief set the flag for gradient array state.
 * \param handle NDArray handle
 * \param state the new state.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArraySetGradState(NDArrayHandle handle, int state);
/*!
 * \brief set the flag for gradient array state.
 * \param handle NDArray handle
 * \param state the new state.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayGetGradState(NDArrayHandle handle, int *out);
//--------------------------------
// Part 2: functions on NDArray
//--------------------------------
/*!
 * \brief list all the available functions handles
 *   most user can use it to list all the needed functions
 * \param out_size the size of returned array
 * \param out_array the output function array
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXListFunctions(uint32_t *out_size,
                              FunctionHandle **out_array);

/*!
 * \brief get the function handle by name
 * \param name the name of the function
 * \param out the corresponding function handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXGetFunction(const char *name,
                            FunctionHandle *out);
/*!
 * \brief Get the information of the function handle.
 * \param fun The function handle.
 * \param name The returned name of the function.
 * \param description The returned description of the function.
 * \param num_args Number of arguments.
 * \param arg_names Name of the arguments.
 * \param arg_type_infos Type information about the arguments.
 * \param arg_descriptions Description information about the arguments.
 * \param return_type Return type of the function.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXFuncGetInfo(FunctionHandle fun,
                            const char **name,
                            const char **description,
                            uint32_t *num_args,
                            const char ***arg_names,
                            const char ***arg_type_infos,
                            const char ***arg_descriptions,
                            const char **return_type DEFAULT(NULL));
/*!
 * \brief get the argument requirements of the function
 * \param fun input function handle
 * \param num_use_vars how many NDArrays to be passed in as used_vars
 * \param num_scalars scalar variable is needed
 * \param num_mutate_vars how many NDArrays to be passed in as mutate_vars
 * \param type_mask the type mask of this function
 * \return 0 when success, -1 when failure happens
 * \sa MXFuncInvoke
 */
MXNET_DLL int MXFuncDescribe(FunctionHandle fun,
                             uint32_t *num_use_vars,
                             uint32_t *num_scalars,
                             uint32_t *num_mutate_vars,
                             int *type_mask);
/*!
 * \brief invoke a function, the array size of passed in arguments
 *   must match the values in the
 * \param fun the function
 * \param use_vars the normal arguments passed to function
 * \param scalar_args the scalar qarguments
 * \param mutate_vars the mutate arguments
 * \param num_params number of keyword parameters
 * \param param_keys keys for keyword parameters
 * \param param_vals values for keyword parameters
 * \return 0 when success, -1 when failure happens
 * \sa MXFuncDescribeArgs
 */
MXNET_DLL int MXFuncInvoke(FunctionHandle fun,
                           NDArrayHandle *use_vars,
                           float *scalar_args,
                           NDArrayHandle *mutate_vars,
                           int num_params,
                           char **param_keys,
                           char **param_vals);
/*!
 * \brief invoke a nnvm op and imperative function
 * \param creator the op
 * \param num_inputs number of input NDArrays
 * \param inputs input NDArrays
 * \param num_outputs number of output NDArrays
 * \param outputs output NDArrays
 * \param num_params number of keyword parameters
 * \param param_keys keys for keyword parameters
 * \param param_vals values for keyword parameters
 * \param out_stypes output ndarrays' stypes
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXImperativeInvoke(AtomicSymbolCreator creator,
                                 int num_inputs,
                                 NDArrayHandle *inputs,
                                 int *num_outputs,
                                 NDArrayHandle **outputs,
                                 int num_params,
                                 const char **param_keys,
                                 const char **param_vals,
                                 const int **out_stypes);
/*!
 * \brief set whether to record operator for autograd
 * \param is_recording 1 when recording, 0 when not recording.
 * \param prev returns the previous status before this set.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXAutogradSetIsRecording(int is_recording, int* prev);
/*!
 * \brief set whether to record operator for autograd
 * \param is_training 1 when training, 0 when testing
 * \param prev returns the previous status before this set.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXAutogradSetIsTraining(int is_training, int* prev);
/*!
 * \brief get whether autograd recording is on
 * \param curr returns the current status.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXAutogradIsRecording(bool* curr);
/*!
 * \brief get whether training mode is on
 * \param curr returns the current status.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXAutogradIsTraining(bool* curr);
/*!
 * \brief get whether numpy compatibility is on
 * \param curr returns the current status
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXIsNumpyShape(int* curr);
/*!
 * \brief set numpy compatibility switch
 * \param is_np_shape 1 when numpy shape semantics is thread local on,
 *        2 when numpy shape semantics is global on and 0 when off
 * \param prev returns the previous status before this set
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSetIsNumpyShape(int is_np_shape, int* prev);
/*!
 * \brief get numpy default data type
 * \param curr returns the current status
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXIsNumpyDefaultDtype(bool* curr);
/*!
 * \brief set numpy default data type
 * \param dtype_flag false when default dtype is flaot32,
 *                   true when default dtype is flaot64.
 * \param prev returns the previous status before this set
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSetIsNumpyDefaultDtype(bool dtype_flag, bool* prev);
/*!
 * \brief mark NDArrays as variables to compute gradient for autograd
 * \param num_var number of variable NDArrays
 * \param var_handles variable NDArrays
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXAutogradMarkVariables(uint32_t num_var,
                                      NDArrayHandle *var_handles,
                                      uint32_t *reqs_array,
                                      NDArrayHandle *grad_handles);
/*!
 * \brief compute the gradient of outputs w.r.t variabels
 * \param num_output number of output NDArray
 * \param output_handles output NDArrays
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXAutogradComputeGradient(uint32_t num_output,
                                        NDArrayHandle* output_handles);
/*!
 * \brief compute the gradient of outputs w.r.t variabels
 * \param num_output number of output NDArray
 * \param output_handles output NDArrays
 * \param ograd_handles head gradient for NDArrays
 * \param retain_graph whether to keep the graph after backward
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXAutogradBackward(uint32_t num_output,
                                 NDArrayHandle* output_handles,
                                 NDArrayHandle* ograd_handles,
                                 int retain_graph);
/*!
 * \brief compute the gradient of outputs w.r.t variabels
 * \param num_output number of output NDArray
 * \param output_handles output NDArrays
 * \param ograd_handles head gradient for NDArrays
 * \param num_variables number of variables
 * \param
 * \param retain_graph whether to keep the graph after backward
 * \param is_train whether to do backward for training or inference
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXAutogradBackwardEx(uint32_t num_output,
                                   NDArrayHandle *output_handles,
                                   NDArrayHandle *ograd_handles,
                                   uint32_t num_variables,
                                   NDArrayHandle *var_handles,
                                   int retain_graph,
                                   int create_graph,
                                   int is_train,
                                   NDArrayHandle **grad_handles,
                                   int **grad_stypes);
/*
 * \brief get the graph constructed by autograd.
 * \param handle ndarray handle
 * \param out output symbol handle
 */
MXNET_DLL int MXAutogradGetSymbol(NDArrayHandle handle, SymbolHandle *out);

/*!
 * \brief create cached operator, allows to choose thread_safe version
 * of cachedop
 */
MXNET_DLL int MXCreateCachedOp(SymbolHandle handle,
                               int num_flags,
                               const char** keys,
                               const char** vals,
                               CachedOpHandle *out,
                               bool thread_safe DEFAULT(false));

/*!
 * \brief free cached operator
 */
MXNET_DLL int MXFreeCachedOp(CachedOpHandle handle);

/*!
 * \brief get optimized graph from the cached op
 */
MXNET_DLL int MXCachedOpGetOptimizedSymbol(CachedOpHandle handle,
                                           SymbolHandle *out);

/*!
 * \brief invoke a cached op
 * \param handle the handle to the cached op
 * \param num_inputs number of input NDArrays
 * \param inputs input NDArrays
 * \param num_outputs number of output NDArrays
 * \param default_dev_type the default context type
 * \param default_dev_id the default context device id
 * \param outputs output NDArrays
 * \param out_stypes output ndarrays' stypes
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXInvokeCachedOp(CachedOpHandle handle,
                               int num_inputs,
                               NDArrayHandle *inputs,
                               int default_dev_type,
                               int default_dev_id,
                               int *num_outputs,
                               NDArrayHandle **outputs,
                               const int** out_stypes);

/*!
 * \brief cached op set monitor callback
 */
MXNET_DLL int MXCachedOpRegisterOpHook(CachedOpHandle handle,
                                       CachedOpMonitorCallback callback,
                                       bool monitor_all);

/*!
 * \brief Get current status of deferred compute mode
 * \param curr returns the current status.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArrayIsDeferredCompute(int *curr);

/*!
 * \brief set whether to enable deferred compute mode
 * \param deferred_compute_enabled 1 to enable, 0 to disable.
 * \param prev returns the previous status before this set.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArraySetIsDeferredCompute(int deferred_compute_enabled, int *prev);

/*!
 * \brief Associate variables with deferred compute arrays
 * \param arrays ndarray handles to be matched with variables
 * \param variables symbol handles of variables to be matched with ndarrays
 * \param num number of arrays and variables respectively
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXNDArraySetDeferredComputeVariable(NDArrayHandle *arrays,
                                                  SymbolHandle *variables,
                                                  int num);

/*!
 * \brief Convert the graph constructed during deferred computation mode to a Symbol.
 * \param output_handles ndarray handles of outputs
 * \param out grouped output symbol handle
 *
 * Construct a Symbol for the deferred computation graph. output_handles
 * specifies the outputs of interest which the returned symbol will compute.
 */
MXNET_DLL int MXNDArrayGetDeferredComputeSymbol(NDArrayHandle *output_handles,
                                                int num_outputs,
                                                SymbolHandle *out);

//--------------------------------------------
// Part 3: symbolic configuration generation
//--------------------------------------------
/*!
 * \brief list all the available operator names, include entries.
 * \param out_size the size of returned array
 * \param out_array the output operator name array.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXListAllOpNames(uint32_t *out_size,
                               const char ***out_array);

/*!
 * \brief list all the available AtomicSymbolEntry
 * \param out_size the size of returned array
 * \param out_array the output AtomicSymbolCreator array
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolListAtomicSymbolCreators(uint32_t *out_size,
                                               AtomicSymbolCreator **out_array);

/*!
 * \brief Get the name of an atomic symbol.
 * \param creator the AtomicSymbolCreator.
 * \param name The returned name of the creator.
 */
MXNET_DLL int MXSymbolGetAtomicSymbolName(AtomicSymbolCreator creator,
                                          const char **name);

/*!
 * \brief Get the input symbols of the graph.
 * \param sym The graph.
 * \param inputs The input symbols of the graph.
 * \param input_size the number of input symbols returned.
 */
MXNET_DLL int MXSymbolGetInputSymbols(SymbolHandle sym, SymbolHandle **inputs,
                                      int *input_size);

/*!
 * \brief Cut a subgraph whose nodes are marked with a subgraph attribute.
 * The input graph will be modified. A variable node will be created for each
 * edge that connects to nodes outside the subgraph. The outside nodes that
 * connect to the subgraph will be returned.
 * \param sym The graph.
 * \param inputs The nodes that connect to the subgraph.
 * \param input_size The number of such nodes.
 */
MXNET_DLL int MXSymbolCutSubgraph(SymbolHandle sym, SymbolHandle **inputs,
                                  int *input_size);

/*!
 * \brief Get the detailed information about atomic symbol.
 * \param creator the AtomicSymbolCreator.
 * \param name The returned name of the creator.
 * \param description The returned description of the symbol.
 * \param num_args Number of arguments.
 * \param arg_names Name of the arguments.
 * \param arg_type_infos Type informations about the arguments.
 * \param arg_descriptions Description information about the arguments.
 * \param key_var_num_args The keyword argument for specifying variable number of arguments.
 *            When this parameter has non-zero length, the function allows variable number
 *            of positional arguments, and will need the caller to pass it in in
 *            MXSymbolCreateAtomicSymbol,
 *            With key = key_var_num_args, and value = number of positional arguments.
 * \param return_type Return type of the function, can be Symbol or Symbol[]
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGetAtomicSymbolInfo(AtomicSymbolCreator creator,
                                          const char **name,
                                          const char **description,
                                          uint32_t *num_args,
                                          const char ***arg_names,
                                          const char ***arg_type_infos,
                                          const char ***arg_descriptions,
                                          const char **key_var_num_args,
                                          const char **return_type DEFAULT(NULL));
/*!
 * \brief Create an AtomicSymbol.
 *
 * A Symbol is said to be atomic if it is not composed of other Symbols. Atomic
 * Symbols can be composed.
 *
 * \param creator the AtomicSymbolCreator
 * \param num_param the number of parameters
 * \param keys the keys to the params
 * \param vals the vals of the params
 * \param out pointer to the created symbol handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolCreateAtomicSymbol(AtomicSymbolCreator creator,
                                         uint32_t num_param,
                                         const char **keys,
                                         const char **vals,
                                         SymbolHandle *out);
/*!
 * \brief Create a Variable Symbol.
 * \param name name of the variable
 * \param out pointer to the created symbol handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolCreateVariable(const char *name, SymbolHandle *out);
/*!
 * \brief Create a Symbol by grouping list of symbols together
 * \param num_symbols number of symbols to be grouped
 * \param symbols array of symbol handles
 * \param out pointer to the created symbol handle
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolCreateGroup(uint32_t num_symbols,
                                  SymbolHandle *symbols,
                                  SymbolHandle *out);
/*!
 * \brief Load a symbol from a json file.
 * \param fname the file name.
 * \param out the output symbol.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolCreateFromFile(const char *fname, SymbolHandle *out);
/*!
 * \brief Load a symbol from a json string.
 * \param json the json string.
 * \param out the output symbol.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolCreateFromJSON(const char *json, SymbolHandle *out);
/*!
 * \brief Remove the operators amp_cast and amp_multicast
 * \param sym_handle the input symbol.
 * \param ret_sym_handle the output symbol.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolRemoveAmpCast(SymbolHandle sym_handle, SymbolHandle* ret_sym_handle);
/*!
 * \brief Save a symbol into a json file.
 * \param symbol the input symbol.
 * \param fname the file name.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolSaveToFile(SymbolHandle symbol, const char *fname);
/*!
 * \brief Save a symbol into a json string
 * \param symbol the input symbol.
 * \param out_json output json string.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolSaveToJSON(SymbolHandle symbol, const char **out_json);
/*!
 * \brief Free the symbol handle.
 * \param symbol the symbol
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolFree(SymbolHandle symbol);
/*!
 * \brief Copy the symbol to another handle
 * \param symbol the source symbol
 * \param out used to hold the result of copy
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolCopy(SymbolHandle symbol, SymbolHandle *out);
/*!
 * \brief Print the content of symbol, used for debug.
 * \param symbol the symbol
 * \param out_str pointer to hold the output string of the printing.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolPrint(SymbolHandle symbol, const char **out_str);
/*!
 * \brief Get string name from symbol
 * \param symbol the source symbol
 * \param out The result name.
 * \param success Whether the result is contained in out.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGetName(SymbolHandle symbol,
                              const char** out,
                              int *success);
/*!
 * \brief Get string attribute from symbol
 * \param symbol the source symbol
 * \param key The key of the symbol.
 * \param out The result attribute, can be NULL if the attribute do not exist.
 * \param success Whether the result is contained in out.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGetAttr(SymbolHandle symbol,
                              const char* key,
                              const char** out,
                              int *success);
/*!
 * \brief Set string attribute from symbol.
 *  NOTE: Setting attribute to a symbol can affect the semantics(mutable/immutable) of symbolic graph.
 *
 *  Safe recommendaton: use  immutable graph
 *  - Only allow set attributes during creation of new symbol as optional parameter
 *
 *  Mutable graph (be careful about the semantics):
 *  - Allow set attr at any point.
 *  - Mutating an attribute of some common node of two graphs can cause confusion from user.
 *
 * \param symbol the source symbol
 * \param key The key of the symbol.
 * \param value The value to be saved.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolSetAttr(SymbolHandle symbol,
                              const char* key,
                              const char* value);
/*!
 * \brief Get all attributes from symbol, including all descendents.
 * \param symbol the source symbol
 * \param out_size The number of output attributes
 * \param out 2*out_size strings representing key value pairs.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolListAttr(SymbolHandle symbol,
                               uint32_t *out_size,
                               const char*** out);
/*!
 * \brief Get all attributes from symbol, excluding descendents.
 * \param symbol the source symbol
 * \param out_size The number of output attributes
 * \param out 2*out_size strings representing key value pairs.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolListAttrShallow(SymbolHandle symbol,
                                      uint32_t *out_size,
                                      const char*** out);
/*!
 * \brief List arguments in the symbol.
 * \param symbol the symbol
 * \param out_size output size
 * \param out_str_array pointer to hold the output string array
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolListArguments(SymbolHandle symbol,
                                    uint32_t *out_size,
                                    const char ***out_str_array);

/*!
 * \brief List returns in the symbol.
 * \param symbol the symbol
 * \param out_size output size
 * \param out_str_array pointer to hold the output string array
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolListOutputs(SymbolHandle symbol,
                                  uint32_t *out_size,
                                  const char ***out_str_array);

/*!
 * \brief Get number of outputs of the symbol.
 * \param symbol The symbol
 * \param out_size number of outputs
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGetNumOutputs(SymbolHandle symbol,
                                    uint32_t *output_count);

/*!
 * \brief Get a symbol that contains all the internals.
 * \param symbol The symbol
 * \param out The output symbol whose outputs are all the internals.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGetInternals(SymbolHandle symbol,
                                   SymbolHandle *out);
/*!
 * \brief Get a symbol that contains all the inputs.
 * \param symbol The symbol
 * \param out The output symbol whose outputs are all the internals.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGetInputs(SymbolHandle symbol,
                                SymbolHandle *out);
/*!
 * \brief Get a symbol that contains only direct children.
 * \param symbol The symbol
 * \param out The output symbol whose outputs are the direct children.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGetChildren(SymbolHandle symbol,
                                  SymbolHandle *out);
/*!
 * \brief Get index-th outputs of the symbol.
 * \param symbol The symbol
 * \param index the Index of the output.
 * \param out The output symbol whose outputs are the index-th symbol.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGetOutput(SymbolHandle symbol,
                                uint32_t index,
                                SymbolHandle *out);

/*!
 * \brief List auxiliary states in the symbol.
 * \param symbol the symbol
 * \param out_size output size
 * \param out_str_array pointer to hold the output string array
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolListAuxiliaryStates(SymbolHandle symbol,
                                          uint32_t *out_size,
                                          const char ***out_str_array);

/*!
 * \brief Compose the symbol on other symbols.
 *
 *  This function will change the sym hanlde.
 *  To achieve function apply behavior, copy the symbol first
 *  before apply.
 *
 * \param sym the symbol to apply
 * \param name the name of symbol
 * \param num_args number of arguments
 * \param keys the key of keyword args (optional)
 * \param args arguments to sym
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolCompose(SymbolHandle sym,
                              const char *name,
                              uint32_t num_args,
                              const char** keys,
                              SymbolHandle* args);
/*!
 * \brief Get the gradient graph of the symbol
 *
 * \param sym the symbol to get gradient
 * \param num_wrt number of arguments to get gradient
 * \param wrt the name of the arguments to get gradient
 * \param out the returned symbol that has gradient
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolGrad(SymbolHandle sym,
                           uint32_t num_wrt,
                           const char** wrt,
                           SymbolHandle* out);

/*!
 * \brief infer shape of unknown input shapes given the known one.
 *  The shapes are packed into a CSR matrix represented by arg_ind_ptr and arg_shape_data
 *  The call will be treated as a kwargs call if key != NULL or num_args==0, otherwise it is positional.
 *  This api is available when MXNet is built with flag
 *  USE_INT64_TENSOR_SIZE=0 (by default)
 * \param sym symbol handle
 * \param num_args number of input arguments.
 * \param keys the key of keyword args (optional)
 * \param arg_ind_ptr the head pointer of the rows in CSR
 * \param arg_shape_data the content of the CSR
 * \param in_shape_size sizeof the returning array of in_shapes
 * \param in_shape_ndim returning array of shape dimensions of eachs input shape.
 * \param in_shape_data returning array of pointers to head of the input shape.
 * \param out_shape_size sizeof the returning array of out_shapes
 * \param out_shape_ndim returning array of shape dimensions of each output shape.
 * \param out_shape_data returning array of pointers to head of the output shape.
 * \param aux_shape_size sizeof the returning array of aux_shapes
 * \param aux_shape_ndim returning array of shape dimensions of each auxiliary shape.
 * \param aux_shape_data returning array of pointers to head of the auxiliary shape.
 * \param complete whether infer shape completes or more information is needed.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolInferShape(SymbolHandle sym,
                                 uint32_t num_args,
                                 const char** keys,
                                 const uint32_t *arg_ind_ptr,
                                 const int *arg_shape_data,
                                 uint32_t *in_shape_size,
                                 const int **in_shape_ndim,
                                 const int ***in_shape_data,
                                 uint32_t *out_shape_size,
                                 const int **out_shape_ndim,
                                 const int ***out_shape_data,
                                 uint32_t *aux_shape_size,
                                 const int **aux_shape_ndim,
                                 const int ***aux_shape_data,
                                 int *complete);

/*!
 * \brief infer shape of unknown input shapes given the known one.
 *  The shapes are packed into a CSR matrix represented by arg_ind_ptr and arg_shape_data
 *  The call will be treated as a kwargs call if key != NULL or num_args==0, otherwise it is positional.
 *  This api is available when MXNet is built with flag
 *  USE_INT64_TENSOR_SIZE=1 (not default) i.e. Large Tensor Support
 * \param sym symbol handle
 * \param num_args number of input arguments.
 * \param keys the key of keyword args (optional)
 * \param arg_ind_ptr the head pointer of the rows in CSR
 * \param arg_shape_data the content of the CSR
 * \param in_shape_size sizeof the returning array of in_shapes
 * \param in_shape_ndim returning array of shape dimensions of each input shape.
 * \param in_shape_data returning array of pointers to head of the input shape.
 * \param out_shape_size sizeof the returning array of out_shapes
 * \param out_shape_ndim returning array of shape dimensions of each output shape.
 * \param out_shape_data returning array of pointers to head of the output shape.
 * \param aux_shape_size sizeof the returning array of aux_shapes
 * \param aux_shape_ndim returning array of shape dimensions of each auxiliary shape.
 * \param aux_shape_data returning array of pointers to head of the auxiliary shape.
 * \param complete whether infer shape completes or more information is needed.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolInferShape64(SymbolHandle sym,
                                   uint32_t num_args,
                                   const char** keys,
                                   const int64_t *arg_ind_ptr,
                                   const int64_t *arg_shape_data,
                                   size_t *in_shape_size,
                                   const int **in_shape_ndim,
                                   const int64_t ***in_shape_data,
                                   size_t *out_shape_size,
                                   const int **out_shape_ndim,
                                   const int64_t ***out_shape_data,
                                   size_t *aux_shape_size,
                                   const int **aux_shape_ndim,
                                   const int64_t ***aux_shape_data,
                                   int *complete);

/*!
 * \brief partially infer shape of unknown input shapes given the known one.
 *
 *  Return partially inferred results if not all shapes could be inferred.
 *  The shapes are packed into a CSR matrix represented by arg_ind_ptr and arg_shape_data
 *  The call will be treated as a kwargs call if key != NULL or num_args==0, otherwise it is positional.
 *  This api is available when MXNet is built with flag
 *  USE_INT64_TENSOR_SIZE=0 (by default)
 *
 * \param sym symbol handle
 * \param num_args number of input arguments.
 * \param keys the key of keyword args (optional)
 * \param arg_ind_ptr the head pointer of the rows in CSR
 * \param arg_shape_data the content of the CSR
 * \param in_shape_size sizeof the returning array of in_shapes
 * \param in_shape_ndim returning array of shape dimensions of each input shape.
 * \param in_shape_data returning array of pointers to head of the input shape.
 * \param out_shape_size sizeof the returning array of out_shapes
 * \param out_shape_ndim returning array of shape dimensions of each output shape.
 * \param out_shape_data returning array of pointers to head of the output shape.
 * \param aux_shape_size sizeof the returning array of aux_shapes
 * \param aux_shape_ndim returning array of shape dimensions of each auxiliary shape.
 * \param aux_shape_data returning array of pointers to head of the auxiliary shape.
 * \param complete whether infer shape completes or more information is needed.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolInferShapePartial(SymbolHandle sym,
                                        uint32_t num_args,
                                        const char** keys,
                                        const uint32_t *arg_ind_ptr,
                                        const int *arg_shape_data,
                                        uint32_t *in_shape_size,
                                        const int **in_shape_ndim,
                                        const int ***in_shape_data,
                                        uint32_t *out_shape_size,
                                        const int **out_shape_ndim,
                                        const int ***out_shape_data,
                                        uint32_t *aux_shape_size,
                                        const int **aux_shape_ndim,
                                        const int ***aux_shape_data,
                                        int *complete);

/*!
 * \brief partially infer shape of unknown input shapes given the known one.
 *
 *  Return partially inferred results if not all shapes could be inferred.
 *  The shapes are packed into a CSR matrix represented by arg_ind_ptr and arg_shape_data
 *  The call will be treated as a kwargs call if key != NULL or num_args==0, otherwise it is positional.
 *  This api is available when MXNet is built with flag
 *  USE_INT64_TENSOR_SIZE=1 (not default) i.e. Large Tensor Support
 *
 * \param sym symbol handle
 * \param num_args number of input arguments.
 * \param keys the key of keyword args (optional)
 * \param arg_ind_ptr the head pointer of the rows in CSR
 * \param arg_shape_data the content of the CSR
 * \param in_shape_size sizeof the returning array of in_shapes
 * \param in_shape_ndim returning array of shape dimensions of each input shape.
 * \param in_shape_data returning array of pointers to head of the input shape.
 * \param out_shape_size sizeof the returning array of out_shapes
 * \param out_shape_ndim returning array of shape dimensions of each output shape.
 * \param out_shape_data returning array of pointers to head of the output shape.
 * \param aux_shape_size sizeof the returning array of aux_shapes
 * \param aux_shape_ndim returning array of shape dimensions of each auxiliary shape.
 * \param aux_shape_data returning array of pointers to head of the auxiliary shape.
 * \param complete whether infer shape completes or more information is needed.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolInferShapePartial64(SymbolHandle sym,
                                          uint32_t num_args,
                                          const char** keys,
                                          const int64_t *arg_ind_ptr,
                                          const int64_t *arg_shape_data,
                                          size_t *in_shape_size,
                                          const int **in_shape_ndim,
                                          const int64_t ***in_shape_data,
                                          size_t *out_shape_size,
                                          const int **out_shape_ndim,
                                          const int64_t ***out_shape_data,
                                          size_t *aux_shape_size,
                                          const int **aux_shape_ndim,
                                          const int64_t ***aux_shape_data,
                                          int *complete);

/*!
 * \brief infer type of unknown input types given the known one.
 *  The types are packed into a CSR matrix represented by arg_ind_ptr and arg_type_data
 *  The call will be treated as a kwargs call if key != NULL or num_args==0, otherwise it is positional.
 *
 * \param sym symbol handle
 * \param num_args numbe of input arguments.
 * \param keys the key of keyword args (optional)
 * \param arg_type_data the content of the CSR
 * \param in_type_size sizeof the returning array of in_types
 * \param in_type_data returning array of pointers to head of the input type.
 * \param out_type_size sizeof the returning array of out_types
 * \param out_type_data returning array of pointers to head of the output type.
 * \param aux_type_size sizeof the returning array of aux_types
 * \param aux_type_data returning array of pointers to head of the auxiliary type.
 * \param complete whether infer type completes or more information is needed.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolInferType(SymbolHandle sym,
                                uint32_t num_args,
                                const char** keys,
                                const int *arg_type_data,
                                uint32_t *in_type_size,
                                const int **in_type_data,
                                uint32_t *out_type_size,
                                const int **out_type_data,
                                uint32_t *aux_type_size,
                                const int **aux_type_data,
                                int *complete);

/*!
 * \brief partially infer type of unknown input types given the known one.
 *
 *  Return partially inferred results if not all types could be inferred.
 *  The types are packed into a CSR matrix represented by arg_ind_ptr and arg_type_data
 *  The call will be treated as a kwargs call if key != NULL or num_args==0, otherwise it is positional.
 *
 * \param sym symbol handle
 * \param num_args numbe of input arguments.
 * \param keys the key of keyword args (optional)
 * \param arg_type_data the content of the CSR
 * \param in_type_size sizeof the returning array of in_types
 * \param in_type_data returning array of pointers to head of the input type.
 * \param out_type_size sizeof the returning array of out_types
 * \param out_type_data returning array of pointers to head of the output type.
 * \param aux_type_size sizeof the returning array of aux_types
 * \param aux_type_data returning array of pointers to head of the auxiliary type.
 * \param complete whether infer type completes or more information is needed.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXSymbolInferTypePartial(SymbolHandle sym,
                                       uint32_t num_args,
                                       const char** keys,
                                       const int *arg_type_data,
                                       uint32_t *in_type_size,
                                       const int **in_type_data,
                                       uint32_t *out_type_size,
                                       const int **out_type_data,
                                       uint32_t *aux_type_size,
                                       const int **aux_type_data,
                                       int *complete);

/*!
 * \brief Convert a symbol into a quantized symbol where FP32 operators are replaced with INT8
 * \param sym_handle symbol to be converted
 * \param ret_sym_handle quantized symbol result
 * \param dev_type device type
 * \param num_excluded_sym_names number of layers excluded from being quantized in the input symbol
 * \param excluded_sym_names node names to be excluded from being quantized
 * \param num_excluded_op_names number of operators excluded from being quantized in the input symbol
 * \param excluded_op_names operator names to be excluded from being quantized
 * \param num_offline number of parameters that are quantized offline
 * \param offline_params array of c strings representing the names of params quantized offline
 * \param quantized_dtype the quantized destination type for input data
 * \param calib_quantize **Deprecated**. quantize op will always be calibrated if could
 * \param quantize_mode quantize mode to be used in quantize pass
 * \param quantize_granularity quantize granularity, tensor-wise or channel-wise
 * \param out_num_calib_names return the number of nodes to be calibrated
 * \param out_calib_names return the node names to be calibrated
 */
MXNET_DLL int MXQuantizeSymbol(SymbolHandle sym_handle,
                               SymbolHandle *ret_sym_handle,
                               const int* dev_type,
                               const uint32_t num_excluded_sym_names,
                               const char **excluded_sym_names,
                               const uint32_t num_excluded_op_names,
                               const char **excluded_op_names,
                               const uint32_t num_offline, const char **offline_params,
                               const char *quantized_dtype, const bool calib_quantize,
                               const char *quantize_mode, const char *quantize_granularity,
                               uint32_t* out_num_calib_names, const char ***out_calib_names);

/*!
 * \brief Convert a symbol into a mixed precision symbol with cast operators for target dtype casting
 * \param sym_handle symbol to be converted
 * \param ret_sym_handle mixed precision symbol result
 * \param num_args number of arguments for known dtypes
 * \param arg_type_data arg types of the arguments
 * \param target_dtype target_dtype for mixed precision symbol
 * \param cast_optional_params whether to cast optional params to target_dtype
 * \param num_target_dtype_op_names number of ops to be casted to target_dtype
 * \param num_fp32_op_names number of ops to be casted to FP32
 * \param num_widest_dtype_op_names number of ops to be casted to widest dtype
 * \param num_conditional_fp32_op_names number of ops to be casted to FP32 based on a condition
 * \param num_excluded_symbols number of symbols to be excluded from casting
 * \param num_model_params number of model parameters
 * \param num_widest_dtype_op_names number of ops to be casted to the widest dtype
 * \param num_conditional_fp32_op_names number of ops to be cast to fp32 based on precision
 * \param target_dtype_op_names op names to be casted to target_dtype
 * \param fp32_op_names op names to be casted to fp32
 * \param widest_dtype_op_names names to be casted to widest dtype
 * \param conditional_fp32_op_names names to be casted to FP32 conditionally
 * \param excluded_symbols symbol names to be excluded from casting
 * \param param_names param names for conditional FP32 casting
 * \param param_values param values for conditional FP32 casting
 * \param arg_names argument names for which type information is provided
 * \param model_param_names names for model parameters
 */
MXNET_DLL int MXReducePrecisionSymbol(SymbolHandle sym_handle,
                                      SymbolHandle *ret_sym_handle,
                                      uint32_t num_args,
                                      const int* arg_type_data,
                                      uint32_t num_ind_ptr,
                                      const int* ind_ptr,
                                      const int* target_dtype,
                                      const int cast_optional_params,
                                      const uint32_t num_target_dtype_op_names,
                                      const uint32_t num_fp32_op_names,
                                      const uint32_t num_widest_dtype_op_names,
                                      const uint32_t num_conditional_fp32_op_names,
                                      const uint32_t num_excluded_symbols,
                                      const uint32_t num_model_params,
                                      const char **target_dtype_op_names,
                                      const char **fp32_op_names,
                                      const char **widest_dtype_op_names,
                                      const char **conditional_fp32_op_names,
                                      const char **excluded_symbols,
                                      const char **conditional_param_names,
                                      const char **conditional_param_vals,
                                      const char **model_param_names,
                                      const char **arg_names);
/*!
 * \brief Set calibration table to node attributes in the sym
 * \param sym_handle symbol whose node attributes are to be set by calibration table
 * \param num_layers number of layers in the calibration table
 * \param layer names stored as keys in the calibration table
 * \param low_quantiles low quantiles of layers stored in the calibration table
 * \param high_quantiles high quantiles of layers stored in the calibration table
 * \param ret_sym_handle returned symbol
 */
MXNET_DLL int MXSetCalibTableToQuantizedSymbol(SymbolHandle qsym_handle,
                                               const uint32_t num_layers,
                                               const char** layer_names,
                                               const float* low_quantiles,
                                               const float* high_quantiles,
                                               SymbolHandle* ret_sym_handle);

/*!
 * \brief Run subgraph pass based on the backend provided
 * \param sym_handle symbol to be converted
 * \param backend backend names for subgraph pass
 * \param ret_sym_handle returned symbol
 */
MXNET_DLL int MXGenBackendSubgraph(SymbolHandle sym_handle, const char *backend,
                                   SymbolHandle *ret_sym_handle);

/*!
 * \brief Generate atomic symbol (able to be composed) from a source symbol
 * \param sym_handle source symbol
 * \param ret_sym_handle returned atomic symbol
 */
MXNET_DLL int MXGenAtomicSymbolFromSymbol(SymbolHandle sym_handle, SymbolHandle *ret_sym_handle);
/*!
 * \brief Partitions symbol for given backend, potentially creating subgraphs
 * \param sym_handle symbol to be partitioned
 * \param dev_type context device type
 * \param backend_name backend name
 * \param ret_sym_handle partitioned symbol returned
 * \param len number of args
 * \param in_args_handle args array
 * \param num_options number of key value pairs
 * \param keys keys for options
 * \param vals values corresponding to keys
 * \param num_input_shapes number of input shapes
 * \param input_shape_names names of the input shapes
 * \param input_shape_data pointer to the contiguous data shapes
 * \param input_shape_idx array of per shape starting idx, the shape length for the i-th input shape
 * is calculate as input_shape_idx[i+1] - input_shape_idx[i]
 * \param num_input_dtypes number of input data types
 * \param input_dtype_names array of names of the input data types
 * \param input_dtypes array of values of the input data types
 * \param num_input_stypesnumber of input storage types
 * \param input_stype_names array of names of the input storage types
 * \param input_stypes array of values of input storage types
 * \param skip_infer if the optimization should skip the attribute inferences
 * (to use if the backend does not require shape inference)
 * \param new_args_cnt pointer a number to store the number of new args
 * \param new_args_handle pointer on array to store the new args handles
 * \param new_arg_names_handle pointer on array to store the new args names
 * \param new_aux_cnt pointer a number to store the number of new aux
 * \param new_aux_handle pointer on array to store the new aux handles
 * \param new_aux_names_handle pointer on array to store the new aux names
 */
MXNET_DLL int MXOptimizeForBackend(SymbolHandle sym_handle,
                                   const char* backend_name,
                                   const int dev_type,
                                   SymbolHandle* ret_sym_handle,
                                   const mx_uint args_len,
                                   NDArrayHandle* in_args_handle,
                                   const mx_uint aux_len,
                                   NDArrayHandle* in_aux_handle,
                                   const mx_uint num_options,
                                   const char** keys,
                                   const char** vals,
                                   const uint32_t num_input_shapes,
                                   const char** input_shape_names,
                                   const int64_t* input_shape_data,
                                   const uint32_t* input_shape_idx,
                                   const uint32_t num_input_dtypes,
                                   const char** input_dtype_names,
                                   const int* input_dtypes,
                                   const uint32_t num_input_stypes,
                                   const char** input_stype_names,
                                   const int* input_stypes,
                                   bool skip_infer,
                                   int* new_args_cnt,
                                   NDArrayHandle** new_args_handle,
                                   char*** new_arg_names_handle,
                                   int* new_aux_cnt,
                                   NDArrayHandle** new_aux_handle,
                                   char*** new_aux_names_handle);


//--------------------------------------------
// Part 5: IO Interface
//--------------------------------------------
/*!
 * \brief List all the available iterator entries
 * \param out_size the size of returned iterators
 * \param out_array the output iteratos entries
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXListDataIters(uint32_t *out_size,
                              DataIterCreator **out_array);
/*!
 * \brief Init an iterator, init with parameters
 * the array size of passed in arguments
 * \param handle of the iterator creator
 * \param num_param number of parameter
 * \param keys parameter keys
 * \param vals parameter values
 * \param out resulting iterator
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterCreateIter(DataIterCreator handle,
                                   uint32_t num_param,
                                   const char **keys,
                                   const char **vals,
                                   DataIterHandle *out);
/*!
 * \brief Get the detailed information about data iterator.
 * \param creator the DataIterCreator.
 * \param name The returned name of the creator.
 * \param description The returned description of the symbol.
 * \param num_args Number of arguments.
 * \param arg_names Name of the arguments.
 * \param arg_type_infos Type informations about the arguments.
 * \param arg_descriptions Description information about the arguments.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterGetIterInfo(DataIterCreator creator,
                                    const char **name,
                                    const char **description,
                                    uint32_t *num_args,
                                    const char ***arg_names,
                                    const char ***arg_type_infos,
                                    const char ***arg_descriptions);
/*!
 * \brief Free the handle to the IO module
 * \param handle the handle pointer to the data iterator
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterFree(DataIterHandle handle);
/*!
 * \brief Move iterator to next position
 * \param handle the handle to iterator
 * \param out return value of next
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterNext(DataIterHandle handle,
                             int *out);
/*!
 * \brief Call iterator.Reset
 * \param handle the handle to iterator
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterBeforeFirst(DataIterHandle handle);

/*!
 * \brief Call iterator.GetLenHint. Note that some iterators don't provide length.
 * \param handle the handle to iterator
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterGetLenHint(DataIterHandle handle,
                                   int64_t *len);
/*!
 * \brief Get the handle to the NDArray of underlying data
 * \param handle the handle pointer to the data iterator
 * \param out handle to underlying data NDArray
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterGetData(DataIterHandle handle,
                                NDArrayHandle *out);
/*!
 * \brief Get the image index by array.
 * \param handle the handle pointer to the data iterator
 * \param out_index output index of the array.
 * \param out_size output size of the array.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterGetIndex(DataIterHandle handle,
                                 uint64_t **out_index,
                                 uint64_t *out_size);
/*!
 * \brief Get the padding number in current data batch
 * \param handle the handle pointer to the data iterator
 * \param pad pad number ptr
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterGetPadNum(DataIterHandle handle,
                                  int *pad);

/*!
 * \brief Get the handle to the NDArray of underlying label
 * \param handle the handle pointer to the data iterator
 * \param out the handle to underlying label NDArray
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterGetLabel(DataIterHandle handle,
                                 NDArrayHandle *out);
/*!
 * \brief Get the handles to specified underlying ndarrays of index
 * \param handle the handle pointer to the data iterator
 * \param num_outputs the length of outputs
 * \param out the handle to an array of NDArrays that stores pointers to handles
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDataIterGetItems(DataIterHandle handle,
                                int* num_outputs,
                                NDArrayHandle **outputs);

/*!
 * \brief List all the available dataset entries
 * \param out_size the size of returned datasets
 * \param out_array the output dataset entries
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXListDatasets(uint32_t *out_size,
                             DatasetCreator **out_array);
/*!
 * \brief Init an dataset, init with parameters
 * the array size of passed in arguments
 * \param handle of the dataset creator
 * \param num_param number of parameter
 * \param keys parameter keys
 * \param vals parameter values
 * \param out resulting dataset
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDatasetCreateDataset(DatasetCreator handle,
                                     uint32_t num_param,
                                     const char **keys,
                                     const char **vals,
                                     DatasetHandle *out);
/*!
 * \brief Get the detailed information about dataset.
 * \param creator the DatasetCreator.
 * \param name The returned name of the creator.
 * \param description The returned description of the symbol.
 * \param num_args Number of arguments.
 * \param arg_names Name of the arguments.
 * \param arg_type_infos Type informations about the arguments.
 * \param arg_descriptions Description information about the arguments.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDatasetGetDatasetInfo(DatasetCreator creator,
                                      const char **name,
                                      const char **description,
                                      uint32_t *num_args,
                                      const char ***arg_names,
                                      const char ***arg_type_infos,
                                      const char ***arg_descriptions);
/*!
 * \brief Free the handle to the IO module
 * \param handle the handle pointer to the dataset
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDatasetFree(DatasetHandle handle);
/*!
 * \brief Get dataset overal length(size)
 * \param handle the handle to dataset
 * \param out return value of GetLen
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDatasetGetLen(DatasetHandle handle,
                              uint64_t *out);
/*!
 * \brief Get Output NDArray given specified indices
 * \param handle the handle to dataset
 * \param index the index of the dataset item to be retrieved
 * \param num_outputs the number of output ndarrays
 * \param outputs the pointers to handles of ndarrays
 * \param is_scalar if not zeros then output should be casted to scalars
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXDatasetGetItems(DatasetHandle handle,
                                uint64_t index,
                                int* num_outputs,
                                NDArrayHandle **outputs);

/*!
 * \brief List all the available batchify function entries
 * \param out_size the size of returned batchify functions
 * \param out_array the output batchify function entries
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXListBatchifyFunctions(uint32_t *out_size,
                                      BatchifyFunctionCreator **out_array);
/*!
 * \brief Init an batchify function, init with parameters
 * the array size of passed in arguments
 * \param handle of the batchify function creator
 * \param num_param number of parameter
 * \param keys parameter keys
 * \param vals parameter values
 * \param out resulting batchify function
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXBatchifyFunctionCreateFunction(BatchifyFunctionCreator handle,
                                     uint32_t num_param,
                                     const char **keys,
                                     const char **vals,
                                     BatchifyFunctionHandle *out);
/*!
 * \brief Get the detailed information about batchify function.
 * \param creator the batchifyFunctionCreator.
 * \param name The returned name of the creator.
 * \param description The returned description of the symbol.
 * \param num_args Number of arguments.
 * \param arg_names Name of the arguments.
 * \param arg_type_infos Type informations about the arguments.
 * \param arg_descriptions Description information about the arguments.
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXBatchifyFunctionGetFunctionInfo(BatchifyFunctionCreator creator,
                                      const char **name,
                                      const char **description,
                                      uint32_t *num_args,
                                      const char ***arg_names,
                                      const char ***arg_type_infos,
                                      const char ***arg_descriptions);
/*!
 * \brief Invoke the Batchify Function
 * \param handle the handle pointer to the batchify function
 * \param batch_size the batch size
 * \param num_output the number of ndarrays for output
 * \param inputs the pointers to input ndarrays
 * \param ouptuts the pointers to output ndarrays
 * \return 0 when success, -1 when failure happens
 */                                      
MXNET_DLL int MXBatchifyFunctionInvoke(BatchifyFunctionHandle handle,
                                       int batch_size,
                                       int num_output,
                                       NDArrayHandle *inputs,
                                       NDArrayHandle **outputs);
/*!
 * \brief Free the handle to the IO module
 * \param handle the handle pointer to the batchify function
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXBatchifyFunctionFree(BatchifyFunctionHandle handle);
//--------------------------------------------
// Part 6: basic KVStore interface
//--------------------------------------------
/*!
 * \brief Initialized ps-lite environment variables
 * \param num_vars number of variables to initialize
 * \param keys environment keys
 * \param vals environment values
 */
MXNET_DLL int MXInitPSEnv(uint32_t num_vars,
                          const char **keys,
                          const char **vals);


/*!
 * \brief Create a kvstore
 * \param type the type of KVStore
 * \param out The output type of KVStore
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreCreate(const char *type,
                              KVStoreHandle *out);

/*!
 * \brief Set parameters to use low-bit compressed gradients
 * \param handle handle to the kvstore
 * \param keys keys for compression parameters
 * \param vals values for compression parameters
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreSetGradientCompression(KVStoreHandle handle,
                                              uint32_t num_params,
                                              const char** keys,
                                              const char** vals);

/*!
 * \brief Delete a KVStore handle.
 * \param handle handle to the kvstore
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreFree(KVStoreHandle handle);
/*!
 * \brief Init a list of (key,value) pairs in kvstore
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreInit(KVStoreHandle handle,
                            uint32_t num,
                            const int* keys,
                            NDArrayHandle* vals);

/*!
 * \brief Init a list of (key,value) pairs in kvstore, where each key is a string
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreInitEx(KVStoreHandle handle,
                              uint32_t num,
                              const char** keys,
                              NDArrayHandle* vals);

/*!
 * \brief Push a list of (key,value) pairs to kvstore
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \param priority the priority of the action
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStorePush(KVStoreHandle handle,
                            uint32_t num,
                            const int* keys,
                            NDArrayHandle* vals,
                            int priority);
/*!
 * \brief Push a list of (key,value) pairs to kvstore, where each key is a string
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \param priority the priority of the action
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStorePushEx(KVStoreHandle handle,
                              uint32_t num,
                              const char** keys,
                              NDArrayHandle* vals,
                              int priority);
/*!
 * \brief pull a list of (key, value) pairs from the kvstore
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \param priority the priority of the action
 * \param ignore_sparse whether to ignore sparse arrays in the request
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStorePullWithSparse(KVStoreHandle handle,
                                      uint32_t num,
                                      const int* keys,
                                      NDArrayHandle* vals,
                                      int priority,
                                      bool ignore_sparse);
/*!
 * \brief pull a list of (key, value) pairs from the kvstore, where each key is a string
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \param priority the priority of the action
 * \param ignore_sparse whether to ignore sparse arrays in the request
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStorePullWithSparseEx(KVStoreHandle handle,
                                        uint32_t num,
                                        const char** keys,
                                        NDArrayHandle* vals,
                                        int priority,
                                        bool ignore_sparse);
/*!
 * \brief pull a list of (key, value) pairs from the kvstore
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \param priority the priority of the action
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStorePull(KVStoreHandle handle,
                            uint32_t num,
                            const int* keys,
                            NDArrayHandle* vals,
                            int priority);
/*!
 * \brief pull a list of (key, value) pairs from the kvstore, where each key is a string
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \param priority the priority of the action
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStorePullEx(KVStoreHandle handle,
                              uint32_t num,
                              const char** keys,
                              NDArrayHandle* vals,
                              int priority);

/*!
 * \brief pull a list of (key, value) pairs from the kvstore, where each key is an integer.
 *        The NDArray pulled back will be in row_sparse storage with only the specified
 *        row_ids present based row_ids (others rows are zeros).
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \param row_ids the list of row_id NDArrays
 * \param priority the priority of the action
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStorePullRowSparse(KVStoreHandle handle,
                                     uint32_t num,
                                     const int* keys,
                                     NDArrayHandle* vals,
                                     const NDArrayHandle* row_ids,
                                     int priority);
/*!
 * \brief pull a list of (key, value) pairs from the kvstore, where each key is a string.
 *        The NDArray pulled back will be in row_sparse storage with only the specified
 *        row_ids present based row_ids (others rows are zeros).
 * \param handle handle to the kvstore
 * \param num the number of key-value pairs
 * \param keys the list of keys
 * \param vals the list of values
 * \param row_ids the list of row_id NDArrays
 * \param priority the priority of the action
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStorePullRowSparseEx(KVStoreHandle handle,
                                       uint32_t num,
                                       const char** keys,
                                       NDArrayHandle* vals,
                                       const NDArrayHandle* row_ids,
                                       int priority);

/*!
 * \brief broadcast a list of (key, value) pairs from the kvstore
 * \param handle handle to the kvstore
 * \param vnum the number of key-value pairs corresponding to vkeys
 * \param vkeys the list of keys for the values to be pushed
 * \param onum the number of key-value pairs corresponding to okeys
 * \param okeys the list of keys for the values to be pulled
 * \param vals the list of values
 * \param outs the list of outputs
 * \param priority the priority of the action
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreBroadcast(KVStoreHandle handle,
                                 mx_uint vnum,
                                 const int* vkeys,
                                 mx_uint onum,
                                 const int* okeys,
                                 NDArrayHandle* vals,
                                 NDArrayHandle* outs,
                                 int priority);
/*!
 * \brief broadcast a list of (key, value) pairs from the kvstore,
 * where each key is a string
 * \param handle handle to the kvstore
 * \param vnum the number of key-value pairs corresponding to vkeys
 * \param vkeys the list of keys for the values to be pushed
 * \param onum the number of key-value pairs corresponding to okeys
 * \param okeys the list of keys for the values to be pulled
 * \param vals the list of values
 * \param outs the list of outputs
 * \param priority the priority of the action
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreBroadcastEx(KVStoreHandle handle,
                                   mx_uint vnum,
                                   const char** vkeys,
                                   mx_uint onum,
                                   const char** okeys,
                                   NDArrayHandle* vals,
                                   NDArrayHandle* outs,
                                   int priority);

/*!
 * \brief push and pull a list of (key, value) pairs from the kvstore
 * \param handle handle to the kvstore
 * \param vnum the number of key-value pairs corresponding to vkeys
 * \param vkeys the list of keys for the values to be pushed
 * \param onum the number of key-value pairs corresponding to okeys
 * \param okeys the list of keys for the values to be pulled
 * \param vals the list of values
 * \param outs the list of outputs
 * \param priority the priority of the action
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStorePushPull(KVStoreHandle handle,
                                mx_uint vnum,
                                const int* vkeys,
                                mx_uint onum,
                                const int* okeys,
                                NDArrayHandle* vals,
                                NDArrayHandle* outs,
                                int priority);
/*!
 * \brief push and pull a list of (key, value) pairs from the kvstore,
 * where each key is a string
 * \param handle handle to the kvstore
 * \param vnum the number of key-value pairs corresponding to vkeys
 * \param vkeys the list of keys for the values to be pushed
 * \param onum the number of key-value pairs corresponding to okeys
 * \param okeys the list of keys for the values to be pulled
 * \param vals the list of values
 * \param outs the list of outputs
 * \param priority the priority of the action
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStorePushPullEx(KVStoreHandle handle,
                                  mx_uint vnum,
                                  const char** vkeys,
                                  mx_uint onum,
                                  const char** okeys,
                                  NDArrayHandle* vals,
                                  NDArrayHandle* outs,
                                  int priority);

/*!
 * \brief user-defined updater for the kvstore
 * It's this updater's responsibility to delete \a recv and \a local
 * \param the key
 * \param recv the pushed value on this key
 * \param local the value stored on local on this key
 * \param handle The additional handle to the updater
 */
typedef void (MXKVStoreUpdater)(int key,
                                NDArrayHandle recv,
                                NDArrayHandle local,
                                void *handle);
/*!
 * \brief user-defined updater for the kvstore with string keys
 * It's this updater's responsibility to delete \a recv and \a local
 * \param the key
 * \param recv the pushed value on this key
 * \param local the value stored on local on this key
 * \param handle The additional handle to the updater
 */
typedef void (MXKVStoreStrUpdater)(const char* key,
                                   NDArrayHandle recv,
                                   NDArrayHandle local,
                                   void *handle);
/*!
 * \brief register a push updater
 * \param handle handle to the KVStore
 * \param updater udpater function
 * \param updater_handle The additional handle used to invoke the updater
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreSetUpdater(KVStoreHandle handle,
                                  MXKVStoreUpdater updater,
                                  void *updater_handle);
/*!
 * \brief register a push updater with int keys and one with string keys
 * \param handle handle to the KVStore
 * \param updater updater function with int keys
 * \param str_updater updater function with string keys
 * \param updater_handle The additional handle used to invoke the updater
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreSetUpdaterEx(KVStoreHandle handle,
                                    MXKVStoreUpdater updater,
                                    MXKVStoreStrUpdater str_updater,
                                    void *updater_handle);
/*!
 * \brief get the type of the kvstore
 * \param handle handle to the KVStore
 * \param type a string type
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreGetType(KVStoreHandle handle,
                               const char** type);
//--------------------------------------------
// Part 6: advanced KVStore for multi-machines
//--------------------------------------------

/**
 * \brief return The rank of this node in its group, which is in [0, GroupSize).
 *
 * \param handle handle to the KVStore
 * \param ret the node rank
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreGetRank(KVStoreHandle handle,
                               int *ret);

/**
 * \brief return The number of nodes in this group, which is
 * - number of workers if if `IsWorkerNode() == true`,
 * - number of servers if if `IsServerNode() == true`,
 * - 1 if `IsSchedulerNode() == true`,
 * \param handle handle to the KVStore
 * \param ret the group size
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreGetGroupSize(KVStoreHandle handle,
                                    int *ret);

/**
 * \brief return whether or not this process is a worker node.
 * \param ret 1 for yes, 0 for no
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreIsWorkerNode(int *ret);


/**
 * \brief return whether or not this process is a server node.
 * \param ret 1 for yes, 0 for no
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreIsServerNode(int *ret);


/**
 * \brief return whether or not this process is a scheduler node.
 * \param ret 1 for yes, 0 for no
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreIsSchedulerNode(int *ret);

/**
 * \brief global barrier among all worker machines
 *
 * \param handle handle to the KVStore
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreBarrier(KVStoreHandle handle);

/**
 * \brief whether to do barrier when finalize
 *
 * \param handle handle to the KVStore
 * \param barrier_before_exit whether to do barrier when kvstore finalize
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreSetBarrierBeforeExit(KVStoreHandle handle,
                                            const int barrier_before_exit);

/**
 * \brief the prototype of a server controller
 * \param head the head of the command
 * \param body the body of the command
 * \param controller_handle helper handle for implementing controller
 */
typedef void (MXKVStoreServerController)(int head,
                                         const char *body,
                                         void *controller_handle);

/**
 * \brief Run as server (or scheduler)
 * \param handle handle to the KVStore
 * \param controller the user-defined server controller
 * \param controller_handle helper handle for implementing controller
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreRunServer(KVStoreHandle handle,
                                 MXKVStoreServerController controller,
                                 void *controller_handle);

/**
 * \brief Send a command to all server nodes
 * \param handle handle to the KVStore
 * \param cmd_id the head of the command
 * \param cmd_body the body of the command
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXKVStoreSendCommmandToServers(KVStoreHandle handle,
                                             int cmd_id,
                                             const char* cmd_body);

/**
 * \brief Get the number of ps dead node(s) specified by {node_id}
 *
 * \param handle handle to the KVStore
 * \param node_id Can be a node group or a single node.
 *                kScheduler = 1, kServerGroup = 2, kWorkerGroup = 4
 * \param number Ouptut number of dead nodes
 * \param timeout_sec A node fails to send heartbeart in {timeout_sec} seconds
 *                    will be presumed as 'dead'
 */
MXNET_DLL int MXKVStoreGetNumDeadNode(KVStoreHandle handle,
                                      const int node_id,
                                      int *number,
                                      const int timeout_sec DEFAULT(60));

/**
 * \brief Create a RecordIO writer object
 * \param uri path to file
 * \param out handle pointer to the created object
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOWriterCreate(const char *uri, RecordIOHandle *out);

/**
 * \brief Delete a RecordIO writer object
 * \param handle handle to RecordIO object
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOWriterFree(RecordIOHandle handle);

/**
 * \brief Write a record to a RecordIO object
 * \param handle handle to RecordIO object
 * \param buf buffer to write
 * \param size size of buffer
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOWriterWriteRecord(RecordIOHandle handle,
                                          const char *buf, size_t size);

/**
 * \brief Get the current writer pointer position
 * \param handle handle to RecordIO object
 * \param pos handle to output position
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOWriterTell(RecordIOHandle handle, size_t *pos);

/**
 * \brief Create a RecordIO reader object
 * \param uri path to file
 * \param out handle pointer to the created object
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOReaderCreate(const char *uri, RecordIOHandle *out);

/**
 * \brief Delete a RecordIO reader object
 * \param handle handle to RecordIO object
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOReaderFree(RecordIOHandle handle);

/**
 * \brief Write a record to a RecordIO object
 * \param handle handle to RecordIO object
 * \param buf pointer to return buffer
 * \param size point to size of buffer
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOReaderReadRecord(RecordIOHandle handle,
                                        char const **buf, size_t *size);

/**
 * \brief Set the current reader pointer position
 * \param handle handle to RecordIO object
 * \param pos target position
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOReaderSeek(RecordIOHandle handle, size_t pos);

/**
 * \brief Get the current writer pointer position
 * \param handle handle to RecordIO object
 * \param pos handle to output position
 * \return 0 when success, -1 when failure happens
*/
MXNET_DLL int MXRecordIOReaderTell(RecordIOHandle handle, size_t *pos);

/**
 * \brief Create a MXRtc object
*/
MXNET_DLL int MXRtcCreate(char* name, uint32_t num_input, uint32_t num_output,
                          char** input_names, char** output_names,
                          NDArrayHandle* inputs, NDArrayHandle* outputs,
                          char* kernel, RtcHandle *out);

/**
 * \brief Run cuda kernel
*/
MXNET_DLL int MXRtcPush(RtcHandle handle, uint32_t num_input, uint32_t num_output,
                        NDArrayHandle* inputs, NDArrayHandle* outputs,
                        uint32_t gridDimX,
                        uint32_t gridDimY,
                        uint32_t gridDimZ,
                        uint32_t blockDimX,
                        uint32_t blockDimY,
                        uint32_t blockDimZ);

/**
 * \brief Delete a MXRtc object
*/
MXNET_DLL int MXRtcFree(RtcHandle handle);
/*
 * \brief register custom operators from frontend.
 * \param op_type name of custom op
 * \param creator
 */
MXNET_DLL int MXCustomOpRegister(const char* op_type, CustomOpPropCreator creator);
/*
 * \brief record custom function for backward later.
 * \param num_inputs number of input NDArrays.
 * \param inputs handle to input NDArrays.
 * \param num_outputs number of output NDArrays.
 * \param outputs handle to output NDArrays.
 * \param callbacks callbacks for backward function.
 */
MXNET_DLL int MXCustomFunctionRecord(int num_inputs, NDArrayHandle *inputs,
                                     int num_outputs, NDArrayHandle *outputs,
                                     struct MXCallbackList *callbacks);
/*
 * \brief create cuda rtc module
 * \param source cuda source code
 * \param num_options number of compiler flags
 * \param options compiler flags
 * \param num_exports number of exported function names
 * \param exported function names
 * \param out handle to created module
 */
MXNET_DLL int MXRtcCudaModuleCreate(const char* source, int num_options,
                                    const char** options, int num_exports,
                                    const char** exports, CudaModuleHandle *out);
/*
 * \brief delete cuda rtc module
 * \param handle handle to cuda module
 */
MXNET_DLL int MXRtcCudaModuleFree(CudaModuleHandle handle);
/*
 * \brief get kernel from module
 * \param handle handle to cuda module
 * \param name name of kernel function
 * \param num_args number of arguments
 * \param is_ndarray whether argument is ndarray
 * \param is_const whether argument is constant
 * \param arg_types data type of arguments
 * \param out created kernel
 */
MXNET_DLL int MXRtcCudaKernelCreate(CudaModuleHandle handle, const char* name,
                                    int num_args, int* is_ndarray, int* is_const,
                                    int* arg_types, CudaKernelHandle *out);
/*
 * \brief delete kernel
 * \param handle handle to previously created kernel
 */
MXNET_DLL int MXRtcCudaKernelFree(CudaKernelHandle handle);
/*
 * \brief launch cuda kernel
 * \param handle handle to kernel
 * \param dev_id (GPU) device id
 * \param args pointer to arguments
 * \param grid_dim_x grid dimension x
 * \param grid_dim_y grid dimension y
 * \param grid_dim_z grid dimension z
 * \param block_dim_x block dimension x
 * \param block_dim_y block dimension y
 * \param block_dim_z block dimension z
 * \param shared_mem size of dynamically allocated shared memory
 */
MXNET_DLL int MXRtcCudaKernelCall(CudaKernelHandle handle, int dev_id, void** args,
                                  uint32_t grid_dim_x, uint32_t grid_dim_y,
                                  uint32_t grid_dim_z, uint32_t block_dim_x,
                                  uint32_t block_dim_y, uint32_t block_dim_z,
                                  uint32_t shared_mem);
/*!
 * \brief Get shared memory handle from NDArray
 * \param handle NDArray handle.
 * \param shared_pid output PID
 * \param shared_id output shared memory id.
 */
MXNET_DLL int MXNDArrayGetSharedMemHandle(NDArrayHandle handle, int* shared_pid,
                                          int* shared_id);

/*!
 * \brief Release all unreferenced memory from the devices storage managers memory pool
 * \param dev_type device type, specify device we want to take
 * \param dev_id the device id of the specific device
 */
MXNET_DLL int MXStorageEmptyCache(int dev_type, int dev_id);

/*!
 * \brief Reconstruct NDArray from shared memory handle
 * \param shared_pid shared PID
 * \param shared_id shared memory id
 * \param shape pointer to NDArray dimensions
 * \param ndim number of NDArray dimensions
 * \param dtype data type of NDArray
 * \param out constructed NDArray
 */
MXNET_DLL int MXNDArrayCreateFromSharedMem(int shared_pid, int shared_id, const int *shape,
                                           int ndim, int dtype, NDArrayHandle *out);

/*!
  * \brief Push an asynchronous operation to the engine.
  * \param async_func Execution function whici takes a parameter on_complete
  *                   that must be called when the execution ompletes.
  * \param func_param The parameter set on calling async_func, can be NULL.
  * \param deleter The callback to free func_param, can be NULL.
  * \param ctx_handle Execution context.
  * \param const_vars_handle The variables that current operation will use
  *                          but not mutate.
  * \param num_const_vars The number of const_vars_handle.
  * \param mutable_vars_handle The variables that current operation will mutate.
  * \param num_mutable_vars The number of mutable_vars_handle.
  * \param prop_handle Property of the function.
  * \param priority Priority of the action, as hint to the engine.
  * \param opr_name The operation name.
  * \param wait Whether this is a WaitForVar operation.
  */
MXNET_DLL int MXEnginePushAsync(EngineAsyncFunc async_func, void* func_param,
                                EngineFuncParamDeleter deleter, ContextHandle ctx_handle,
                                EngineVarHandle const_vars_handle, int num_const_vars,
                                EngineVarHandle mutable_vars_handle, int num_mutable_vars,
                                EngineFnPropertyHandle prop_handle DEFAULT(NULL),
                                int priority DEFAULT(0), const char* opr_name DEFAULT(NULL),
                                bool wait DEFAULT(false));

/*!
  * \brief Push a synchronous operation to the engine.
  * \param sync_func Execution function that executes the operation.
  * \param func_param The parameter set on calling sync_func, can be NULL.
  * \param deleter The callback to free func_param, can be NULL.
  * \param ctx_handle Execution context.
  * \param const_vars_handle The variables that current operation will use
  *                          but not mutate.
  * \param num_const_vars The number of const_vars_handle.
  * \param mutable_vars_handle The variables that current operation will mutate.
  * \param num_mutable_vars The number of mutable_vars_handle.
  * \param prop_handle Property of the function.
  * \param priority Priority of the action, as hint to the engine.
  * \param opr_name The operation name.
  */
MXNET_DLL int MXEnginePushSync(EngineSyncFunc sync_func, void* func_param,
                               EngineFuncParamDeleter deleter, ContextHandle ctx_handle,
                               EngineVarHandle const_vars_handle, int num_const_vars,
                               EngineVarHandle mutable_vars_handle, int num_mutable_vars,
                               EngineFnPropertyHandle prop_handle DEFAULT(NULL),
                               int priority DEFAULT(0), const char* opr_name DEFAULT(NULL));
/*!
 * \brief Create an NDArray from source sharing the same data chunk.
 * \param src source NDArray
 * \param out new NDArray sharing the same data chunck with src
 */
MXNET_DLL int MXShallowCopyNDArray(NDArrayHandle src, NDArrayHandle* out);
/*!
 * \brief Create an Symbol from source sharing the same graph structure.
 * \param src source Symbol
 * \param out new Symbol sharing the same graph structure with src
 */
MXNET_DLL int MXShallowCopySymbol(SymbolHandle src, SymbolHandle * out);

/*!
  * \brief Push an asynchronous operation to the engine.
  * \param async_func Execution function whici takes a parameter on_complete
  *                   that must be called when the execution ompletes.
  * \param func_param The parameter set on calling async_func, can be NULL.
  * \param deleter The callback to free func_param, can be NULL.
  * \param ctx_handle Execution context.
  * \param const_nds_handle The NDArrays that current operation will use
  *                          but not mutate.
  * \param num_const_nds The number of const_nds_handle.
  * \param mutable_nds_handle The NDArrays that current operation will mutate.
  * \param num_mutable_nds The number of mutable_nds_handle.
  * \param prop_handle Property of the function.
  * \param priority Priority of the action, as hint to the engine.
  * \param opr_name The operation name.
  * \param wait Whether this is a WaitForVar operation.
  */
MXNET_DLL int MXEnginePushAsyncND(EngineAsyncFunc async_func, void* func_param,
                                  EngineFuncParamDeleter deleter, ContextHandle ctx_handle,
                                  NDArrayHandle* const_nds_handle, int num_const_nds,
                                  NDArrayHandle* mutable_nds_handle, int num_mutable_nds,
                                  EngineFnPropertyHandle prop_handle DEFAULT(NULL),
                                  int priority DEFAULT(0), const char* opr_name DEFAULT(NULL),
                                  bool wait DEFAULT(false));

/*!
  * \brief Push a synchronous operation to the engine.
  * \param sync_func Execution function that executes the operation.
  * \param func_param The parameter set on calling sync_func, can be NULL.
  * \param deleter The callback to free func_param, can be NULL.
  * \param ctx_handle Execution context.
  * \param const_nds_handle The NDArrays that current operation will use
  *                          but not mutate.
  * \param num_const_nds The number of const_nds_handle.
  * \param mutable_nds_handle The NDArrays that current operation will mutate.
  * \param num_mutable_nds The number of mutable_nds_handle.
  * \param prop_handle Property of the function.
  * \param priority Priority of the action, as hint to the engine.
  * \param opr_name The operation name.
  */
MXNET_DLL int MXEnginePushSyncND(EngineSyncFunc sync_func, void* func_param,
                                 EngineFuncParamDeleter deleter, ContextHandle ctx_handle,
                                 NDArrayHandle* const_nds_handle, int num_const_nds,
                                 NDArrayHandle* mutable_nds_handle, int num_mutable_nds,
                                 EngineFnPropertyHandle prop_handle DEFAULT(NULL),
                                 int priority DEFAULT(0), const char* opr_name DEFAULT(NULL));

/*!
 * \brief This function checks if any dynamic shape op is present in the symbol.
 * \param sym_handle handler of the input symbol.
 * \param has_dynamic_shape Flag to indicate if the symbol contains dynamic shape op.
 */
MXNET_DLL int MXCheckDynamicShapeOp(SymbolHandle sym_handle,
                                    bool* has_dynamic_shape);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // MXNET_C_API_H_
