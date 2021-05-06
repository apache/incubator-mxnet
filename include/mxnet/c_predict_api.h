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
 * \file c_predict_api.h
 * \brief C predict API of mxnet, contains a minimum API to run prediction.
 *  This file is self-contained, and do not dependent on any other files.
 */
#ifndef MXNET_C_PREDICT_API_H_
#define MXNET_C_PREDICT_API_H_

/*! \brief Inhibit C++ name-mangling for MXNet functions. */
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

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
/*! \brief handle to Predictor */
typedef void *PredictorHandle;
/*! \brief handle to NDArray list */
typedef void *NDListHandle;
/*! \brief handle to NDArray */
typedef void *NDArrayHandle;
/*! \brief callback used for add monitoring to nodes in the graph */
typedef void (*PredMonitorCallback)(const char*,
                                    NDArrayHandle,
                                    void*);

/*! \brief enum of NDArray dtypes */
enum DType {
    kDTypeFloat32 = 0,
    kDTypeFloat64 = 1,
    kDTypeFloat16 = 2,
    kDTypeUInt8 = 3,
    kDTypeInt32 = 4,
    kDTypeInt8 = 5,
    kDTypeInt64 = 6
};
/*!
 * \brief Get the last error happeneed.
 * \return The last error happened at the predictor.
 */
MXNET_DLL const char* MXGetLastError();

/*!
 * \brief create a predictor
 * \param symbol_json_str The JSON string of the symbol.
 * \param param_bytes The in-memory raw bytes of parameter ndarray file.
 * \param param_size The size of parameter ndarray file.
 * \param dev_type The device type, 1: cpu, 2:gpu
 * \param dev_id The device id of the predictor.
 * \param num_input_nodes Number of input nodes to the net,
 *    For feedforward net, this is 1.
 * \param input_keys The name of input argument.
 *    For feedforward net, this is {"data"}
 * \param input_shape_indptr Index pointer of shapes of each input node.
 *    The length of this array = num_input_nodes + 1.
 *    For feedforward net that takes 4 dimensional input, this is {0, 4}.
 * \param input_shape_data A flattened data of shapes of each input node.
 *    For feedforward net that takes 4 dimensional input, this is the shape data.
 * \param out The created predictor handle.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredCreate(const char* symbol_json_str,
                           const void* param_bytes,
                           int param_size,
                           int dev_type, int dev_id,
                           uint32_t num_input_nodes,
                           const char** input_keys,
                           const uint32_t* input_shape_indptr,
                           const uint32_t* input_shape_data,
                           PredictorHandle* out);

/*!
 * \brief create a predictor
 * \param symbol_json_str The JSON string of the symbol.
 * \param param_bytes The in-memory raw bytes of parameter ndarray file.
 * \param param_size The size of parameter ndarray file.
 * \param dev_type The device type, 1: cpu, 2: gpu
 * \param dev_id The device id of the predictor.
 * \param num_input_nodes Number of input nodes to the net.
 *    For feedforward net, this is 1.
 * \param input_keys The name of the input argument.
 *    For feedforward net, this is {"data"}
 * \param input_shape_indptr Index pointer of shapes of each input node.
 *    The length of this array = num_input_nodes + 1.
 *    For feedforward net that takes 4 dimensional input, this is {0, 4}.
 * \param input_shape_data A flattened data of shapes of each input node.
 *    For feedforward net that takes 4 dimensional input, this is the shape data.
 * \param num_provided_arg_dtypes
 *    The length of provided_arg_dtypes.
 * \param provided_arg_dtype_names
 *    The provided_arg_dtype_names the names of args for which dtypes are provided.
 * \param provided_arg_dtypes
 *    The provided_arg_dtypes the dtype provided
 * \param out The created predictor handle.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredCreateEx(const char* symbol_json_str,
                             const void* param_bytes,
                             int param_size,
                             int dev_type, int dev_id,
                             const uint32_t num_input_nodes,
                             const char** input_keys,
                             const uint32_t* input_shape_indptr,
                             const uint32_t* input_shape_data,
                             const uint32_t num_provided_arg_dtypes,
                             const char** provided_arg_dtype_names,
                             const int* provided_arg_dtypes,
                             PredictorHandle* out);

/*!
 * \brief create a predictor wich customized outputs
 * \param symbol_json_str The JSON string of the symbol.
 * \param param_bytes The in-memory raw bytes of parameter ndarray file.
 * \param param_size The size of parameter ndarray file.
 * \param dev_type The device type, 1: cpu, 2:gpu
 * \param dev_id The device id of the predictor.
 * \param num_input_nodes Number of input nodes to the net,
 *    For feedforward net, this is 1.
 * \param input_keys The name of input argument.
 *    For feedforward net, this is {"data"}
 * \param input_shape_indptr Index pointer of shapes of each input node.
 *    The length of this array = num_input_nodes + 1.
 *    For feedforward net that takes 4 dimensional input, this is {0, 4}.
 * \param input_shape_data A flattened data of shapes of each input node.
 *    For feedforward net that takes 4 dimensional input, this is the shape data.
 * \param num_output_nodes Number of output nodes to the net,
 * \param output_keys The name of output argument.
 *    For example {"global_pool"}
 * \param out The created predictor handle.
 * \return 0 when success, -1 when failure.
 */

MXNET_DLL int MXPredCreatePartialOut(const char* symbol_json_str,
                                     const void* param_bytes,
                                     int param_size,
                                     int dev_type, int dev_id,
                                     uint32_t num_input_nodes,
                                     const char** input_keys,
                                     const uint32_t* input_shape_indptr,
                                     const uint32_t* input_shape_data,
                                     uint32_t num_output_nodes,
                                     const char** output_keys,
                                     PredictorHandle* out);

/*!
 * \brief create predictors for multiple threads. One predictor for a thread.
 * \param symbol_json_str The JSON string of the symbol.
 * \param param_bytes The in-memory raw bytes of parameter ndarray file.
 * \param param_size The size of parameter ndarray file.
 * \param dev_type The device type, 1: cpu, 2:gpu
 * \param dev_id The device id of the predictor.
 * \param num_input_nodes Number of input nodes to the net,
 *    For feedforward net, this is 1.
 * \param input_keys The name of input argument.
 *    For feedforward net, this is {"data"}
 * \param input_shape_indptr Index pointer of shapes of each input node.
 *    The length of this array = num_input_nodes + 1.
 *    For feedforward net that takes 4 dimensional input, this is {0, 4}.
 * \param input_shape_data A flattened data of shapes of each input node.
 *    For feedforward net that takes 4 dimensional input, this is the shape data.
 * \param num_threads The number of threads that we'll run the predictors.
 * \param out An array of created predictor handles. The array has to be large
 *   enough to keep `num_threads` predictors.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredCreateMultiThread(const char* symbol_json_str,
                                      const void* param_bytes,
                                      int param_size,
                                      int dev_type, int dev_id,
                                      uint32_t num_input_nodes,
                                      const char** input_keys,
                                      const uint32_t* input_shape_indptr,
                                      const uint32_t* input_shape_data,
                                      int num_threads,
                                      PredictorHandle* out);

/*!
 * \brief Creates a predictor with all input data and params.
 * The order of input_data, input_shape_indptr and input_shape_data
 * should follow the order of input_keys.
 * \param symbol_json_str The JSON string of the symbol
 * \param dev_type The device type, 1: cpu, 2:gpu
 * \param dev_id The device id of the predictor
 * \param input_keys The names of input data and params
 * \param input_data The array of input data and param pointers
 * \param input_dtypes The array of input data dtypes
 * \param input_keys The name of input argument.
 * \param input_shape_indptr Index pointer of shapes of each input node.
 *    The length of this array = num_input_nodes + 1.
 * \param input_shape_data A flatted data of shapes of each input node.
 * \param num_input_nodes Number of input nodes to the net
 * \param out The created predictor handle.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredCreateEx2(const char* symbol_json_str,
                              int dev_type, int dev_id,
                              const char** input_keys,
                              const void** input_data,
                              const int* input_dtypes,
                              const mx_uint* input_shape_indptr,
                              const mx_uint* input_shape_data,
                              const mx_uint num_input_nodes,
                              PredictorHandle* out);
/*!
 * \brief Creates a predictor with customized outputs.
 * The order of input_data, input_shape_indptr and input_shape_data
 * should follow the order of input_keys.
 * \param symbol_json_str The JSON string of the symbol
 * \param dev_type The device type, 1: cpu, 2:gpu
 * \param dev_id The device id of the predictor
 * \param input_keys The names of input data and params
 * \param input_data The array of input data and param pointers
 * \param input_keys The name of input argument.
 * \param input_shape_indptr Index pointer of shapes of each input node.
 *    The length of this array = num_input_nodes + 1.
 * \param input_shape_data A flatted data of shapes of each input node.
 * \param num_input_nodes Number of input nodes to the net
 * \param num_output_nodes Number of output nodes to the net,
 * \param output_keys The name of output argument.
 * \param out The created predictor handle.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredCreatePartialOutEx(const char* symbol_json_str,
                                       int dev_type, int dev_id,
                                       const char** input_keys,
                                       const mx_float** input_data,
                                       const mx_uint* input_shape_indptr,
                                       const mx_uint* input_shape_data,
                                       const mx_uint num_input_nodes,
                                       const mx_uint num_output_nodes,
                                       const char** output_keys,
                                       PredictorHandle* out);
/*!
 * \brief Creates a predictor with customized outputs.
 * The order of input_data, input_shape_indptr and input_shape_data
 * should follow the order of input_keys.
 * \param symbol_json_str The JSON string of the symbol
 * \param dev_type The device type, 1: cpu, 2:gpu
 * \param dev_id The device id of the predictor
 * \param input_keys The names of input data and params
 * \param input_data The array of input data and param pointers
 * \param input_dtypes The array of input data dtypes
 * \param input_shape_indptr Index pointer of shapes of each input node.
 *    The length of this array = num_input_nodes + 1.
 * \param input_shape_data A flatted data of shapes of each input node.
 * \param num_input_nodes Number of input nodes to the net
 * \param num_output_nodes Number of output nodes to the net,
 * \param output_keys The name of output argument.
 * \param out The created predictor handle.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredCreatePartialOutEx2(const char* symbol_json_str,
                                        int dev_type, int dev_id,
                                        const char** input_keys,
                                        const void** input_data,
                                        const int* input_dtypes,
                                        const mx_uint* input_shape_indptr,
                                        const mx_uint* input_shape_data,
                                        const mx_uint num_input_nodes,
                                        const mx_uint num_output_nodes,
                                        const char** output_keys,
                                        PredictorHandle* out);
/*!
 * \brief Change the input shape of an existing predictor.
 * \param num_input_nodes Number of input nodes to the net,
 *    For feedforward net, this is 1.
 * \param input_keys The name of input argument.
 *    For feedforward net, this is {"data"}
 * \param input_shape_indptr Index pointer of shapes of each input node.
 *    The length of this array = num_input_nodes + 1.
 *    For feedforward net that takes 4 dimensional input, this is {0, 4}.
 * \param input_shape_data A flattened data of shapes of each input node.
 *    For feedforward net that takes 4 dimensional input, this is the shape data.
 * \param handle The original predictor handle.
 * \param out The reshaped predictor handle.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredReshape(uint32_t num_input_nodes,
                  const char** input_keys,
                  const uint32_t* input_shape_indptr,
                  const uint32_t* input_shape_data,
                  PredictorHandle handle,
                  PredictorHandle* out);
/*!
 * \brief Get the list of output node names.
 * The returned out_names is valid until MXPredFree is called on the given PredictorHandle.
 * \param handle The handle of the predictor.
 * \param out_names_length Used to hold The length of the out_names array.
 * \param out_names Used to hold the array of output node names.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredGetOutputNames(PredictorHandle handle,
                                   mx_uint *out_names_length,
                                   const char ***out_names);
/*!
 * \brief Get the shape of output node.
 *  The returned shape_data and shape_ndim is only valid before next call to MXPred function.
 * \param handle The handle of the predictor.
 * \param index The index of output node, set to 0 if there is only one output.
 * \param shape_data Used to hold pointer to the shape data
 * \param shape_ndim Used to hold shape dimension.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredGetOutputShape(PredictorHandle handle,
                                   uint32_t index,
                                   uint32_t** shape_data,
                                   uint32_t* shape_ndim);

/*!
 * \brief Get the dtype of output node.
 * The returned data type is only valid before next call to MXPred function.
 * \param handle The handle of the predictor.
 * \param out_index The index of the output node, set to 0 if there is only one output.
 * \param out_dtype The dtype of the output node
 */
MXNET_DLL int MXPredGetOutputType(PredictorHandle handle,
                                  uint32_t out_index,
                                  int* out_dtype);
/*!
 * \brief Get the shape of input node.
 * \param handle The handle of the predictor.
 * \param key The name of input argument.
 * \param shape_data Used to hold pointer to the shape data.
 * \param shape_ndim Used to hold shape dimension.
 * \param key_found Used to return if input key is a valid argument
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredGetInputShape(PredictorHandle handle,
                                   const char* key,
                                   mx_uint** shape_data,
                                   mx_uint* shape_ndim,
                                   int* key_found);
/*!
 * \brief Get the dtype of input node.
 * \param handle The handle of the predictor.
 * \param key The name of input argument.
 * \param dtype Used to hold pointer to the dtype
 * \param key_found Used to return if input key is a valid argument.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredGetInputType(PredictorHandle handle,
                                  const char* key,
                                  int* dtype,
                                  int* key_found);
/*!
 * \brief Set the input data of predictor.
 * \param handle The predictor handle.
 * \param key The name of input node to set.
 *     For feedforward net, this is "data".
 * \param data The pointer to the data to be set, with the shape specified in MXPredCreate.
 * \param size The size of data array, used for safety check.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredSetInput(PredictorHandle handle,
                             const char* key,
                             const float* data,
                             uint32_t size);
/*!
 * \brief Run a forward pass to get the output.
 * \param handle The handle of the predictor.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredForward(PredictorHandle handle);
/*!
 * \brief Run a interactive forward pass to get the output.
 *  This is helpful for displaying progress of prediction which can be slow.
 *  User must call PartialForward from step=0, keep increasing it until step_left=0.
 * \code
 * int step_left = 1;
 * for (int step = 0; step_left != 0; ++step) {
 *    MXPredPartialForward(handle, step, &step_left);
 *    printf("Current progress [%d/%d]\n", step, step + step_left + 1);
 * }
 * \endcode
 * \param handle The handle of the predictor.
 * \param step The current step to run forward on.
 * \param step_left The number of steps left
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredPartialForward(PredictorHandle handle, int step, int* step_left);
/*!
 * \brief Get the output value of prediction.
 * \param handle The handle of the predictor.
 * \param index The index of output node, set to 0 if there is only one output.
 * \param data User allocated data to hold the output.
 * \param size The size of data array, used for safe checking.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredGetOutput(PredictorHandle handle,
                              uint32_t index,
                              float* data,
                              uint32_t size);
/*!
 * \brief Free a predictor handle.
 * \param handle The handle of the predictor.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXPredFree(PredictorHandle handle);
/*!
 * \brief Create a NDArray List by loading from ndarray file.
 *     This can be used to load mean image file.
 * \param nd_file_bytes The byte contents of nd file to be loaded.
 * \param nd_file_size The size of the nd file to be loaded.
 * \param out The out put NDListHandle
 * \param out_length Length of the list.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXNDListCreate(const char* nd_file_bytes,
                             int nd_file_size,
                             NDListHandle *out,
                             uint32_t* out_length);
/*!
 * \brief Get an element from list
 * \param handle The handle to the NDArray
 * \param index The index in the list
 * \param out_key The output key of the item
 * \param out_data The data region of the item
 * \param out_shape The shape of the item.
 * \param out_ndim The number of dimension in the shape.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXNDListGet(NDListHandle handle,
                          uint32_t index,
                          const char** out_key,
                          const float** out_data,
                          const uint32_t** out_shape,
                          uint32_t* out_ndim);

/*!
 * \brief Get an element from list
 * \param handle The handle to the NDArray
 * \param index The index in the list
 * \param out_key The output key of the item
 * \param out_data The data region of the item
 * \param out_dtype The dtype of the item.
 * \param out_shape The shape of the item.
 * \param out_ndim The number of dimension in the shape.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXNDListGetEx(NDListHandle handle,
                            mx_uint index,
                            const char** out_key,
                            const void** out_data,
                            int* out_dtype,
                            const mx_uint** out_shape,
                            mx_uint* out_ndim);

/*!
 * \brief set a call back to notify the completion of operation and allow for
 * additional monitoring
 */
MXNET_DLL int MXPredSetMonitorCallback(PredictorHandle handle,
                                       PredMonitorCallback callback,
                                       void* callback_handle,
                                       bool monitor_all);
/*!
 * \brief Free a MXAPINDList
 * \param handle The handle of the MXAPINDList.
 * \return 0 when success, -1 when failure.
 */
MXNET_DLL int MXNDListFree(NDListHandle handle);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // MXNET_C_PREDICT_API_H_
