---
layout: page_api
title: Multi Threaded Inference
action: Get Started
action_url: /get_started
permalink: /api/cpp/docs/tutorials/multi_threaded_inference.html
is_tutorial: true
tag: cpp
---
<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Multi Threaded Inference API

A long standing request from MXNet users has been to invoke parallel inference on a model from multiple threads while sharing the parameters.
With this use case in mind, the threadsafe version of CachedOp was added to provide a way for customers to do multi-threaded inference for MXNet users.
This doc attempts to do the following:
1. Discuss the current state of thread safety in MXNet
2. Explain how one can use C API and thread safe version of cached op, along with CPP package to achieve iultithreaded inference. This will be useful for end users as well as frontend developers of different language bindings
3. Discuss the limitations of the above approach
4. Future Work

## Current state of Thread Safety in MXNet

Examining the current state of thread safety in MXNet we can arrive to the following conclusion:

1. MXNet Dependency Engine is thread safe (except for WaitToRead invoked inside a spawned thread. Please see Limitations section)
2. Graph Executor which is Module/Symbolic/C Predict API backend is not thread safe
3. Cached Op (Gluon Backend) is not thread safe

The CachedOpThreadSafe and corresponding C APIs were added to address point 3 above and provide a way
for MXNet users to do multi-threaded inference.

```
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
```

## Multithreaded inference in MXNet with C API and CPP Package

### Prerequisites
To complete this tutorial you need to:
- Learn the basics about [MXNet C++ API](/api/cpp)
- Build MXNet from source with make/cmake
- Build the multi-threaded inference example

### Setup the MXNet C++ API
To use the C++ API in MXNet, you need to build MXNet from source with C++ package. Please follow the [built from source guide](/get_started/build_from_source.html), and [C++ Package documentation](/api/cpp.html)
The summary of those two documents is that you need to build MXNet from source with `USE_CPP_PACKAGE` flag set to 1.
This example requires a build with CUDA and CUDNN.

### Get the example
If you have built mxnet from source with cmake, then do the following:

```bash
$ cp build/cpp-package/example/multi_threaded_inference .
```

### Run multi threaded inference example
The example is tested with models such as `imagenet1k-inception-bn`, `imagenet1k-resnet-50`,
`imagenet1k-resnet-152`, `imagenet1k-resnet-18`

To run the multi threaded inference example:

First export `LD_LIBRARY_PATH`:

```bash
$ export LD_LIBRARY_PATH=<MXNET_LIB_DIR>:$LD_LIBRARY_PATH
```

```bash
$ ./multi_threaded_inference [model_name] [is_gpu] [file_names]
```
e.g.

```bash
./multi_threaded_inference imagenet1k-inception-bn 1 grace_hopper.jpg dog.jpg
```

The above script spawns 2 threads, shares the same cachedop and params among two threads, and runs inference on GPU. It returns the inference results in the order in which files are provided.

NOTE: This example is to demonstrate the multi-threaded-inference with cached op. The inference results work well only with specific models (e.g. imagenet1k-inception-bn). The results may not necessarily be very accurate because of different preprocessing step required etc.

### Code walkthrough multi-threaded inference with CachedOp

The multi threaded inference example (`multi_threaded_inference.cc`) involves the following steps:

1. Parse arguments and load input image into ndarray
2. Prepare input data and load parameters, copying data to a specific context
3. Preparing arguments to pass to the CachedOp and calling C API to **create cached op**
4. Prepare lambda function which will run in spawned threads. Call C API to **invoke cached op** within the lambda function.
5. Spawn multiple threads and wait for all threads to complete.
6. Post process data to obtain inference results and cleanup.

### Step 1: Parse arguments and load input image into ndarray

[https://github.com/apache/incubator-mxnet/example/multi_threaded_inference/multi_threaded_inference.cc#L299-L341](multi_threaded_inference.cc#L299-L341)

The above code parses arguments, loads the image file into a ndarray with a specific shape. There are a few things that are set by default and not configurable. For example, `static_alloc` and `static_shape` are by default set to true.


### Step 2: Prepare input data and load parameters, copying data to a specific context

[https://github.com/apache/incubator-mxnet/example/multi_threaded_inference/multi_threaded_inference.cc#L147-L205](multi_threaded_inference.cc#L147-L205)

The above code loads params and copies input data and params to specific context.

### Step 3: Preparing arguments to pass to the CachedOp and calling C API to create cached op

[https://github.com/apache/incubator-mxnet/example/multi_threaded_inference/multi_threaded_inference.cc#L207-L233](multi_threaded_inference.cc#L207-233)

The above code prepares `flag_key_cstrs` and `flag_val_cstrs` to be passed the Cached op.
The C API call is made with `MXCreateCachedOp`. This will lead to creation of thread safe cached
op since the `thread_safe` (which is the last parameter to `MXCreateCachedOp`) is set to
true. When this is set to false, it will invoke CachedOp instead of CachedOpThreadSafe.


### Step 4: Prepare lambda function which will run in spawned threads

[https://github.com/apache/incubator-mxnet/example/multi_threaded_inference/multi_threaded_inference.cc#L248-L262](multi_threaded_inference.cc#L248-262)

The above creates the lambda function taking the thread number as the argument.
If `random_sleep` is set it will sleep for a random number (secs) generated between 0 to 5 seconds.
Following this, it invokes `MXInvokeCachedOp`(from the hdl it determines whether to invoke cached op threadsafe version or not).
When this is set to false, it will invoke CachedOp instead of CachedOpThreadSafe.

### Step 5: Spawn multiple threads and wait for all threads to complete

[https://github.com/anirudh2290/apache/incubator-mxnet/example/multi_threaded_inference/multi_threaded_inference.cc#L264-L276](multi_threaded_inference.cc#L264-L276)

Spawns multiple threads, joins and waits to wait for all ops to complete.
The other alternative is to wait in the thread on the output ndarray and remove the WaitAll after join.

### Step 6: Post process data to obtain inference results and cleanup

[https://github.com/apache/incubator-/mxnet/example/multi_threaded_inference/multi_threaded_inference.cc#L286-L293](multi_threaded_inference.cc#L286-293)

The above code outputs results for different threads and cleans up the thread safe cached op.

## Current Limitations

1. Only operators tested with the existing model coverage are supported. Other operators and operator types (stateful operators, custom operators are not supported. Existing model coverage is as follows (this list will keep growing as we test more models with different model types):

|Models Tested|ONEDNN|CUDNN|NO-CUDNN|
| --- | --- | --- | --- |
| imagenet1k-resnet-18 | Yes | Yes | Yes |
| imagenet1k-resnet-152 | Yes | Yes | Yes |
| imagenet1k-resnet-50 | Yes | Yes | Yes |

2. Only dense storage types are supported currently.
3. Multi GPU Inference not supported currently.
4. Instantiating multiple instances of SymbolBlockThreadSafe is not supported. Can run parallel inference only on one model per process.
5. dynamic shapes not supported in thread safe cached op.
6. Bulking of ops is not supported.
7. This only supports inference use cases currently, training use cases are not supported.
8. Graph rewrites with subgraph API currently not supported.
9. There is currently no frontend API support to run multi threaded inference. Users can use CreateCachedOp and InvokeCachedOp in combination with
the CPP frontend to run multi-threaded inference as of today.
10. Multi threaded inference with threaded engine with Module/Symbolic API and C Predict API are not currently supported.
11. Exception thrown with `wait_to_read` in individual threads can cause issues. Calling invoke from each thread and calling WaitAll after thread joins should still work fine.
12. Tested only on environments supported by CI. This means that MacOS is not supported.

## Future Work

Future work includes Increasing model coverage and addressing most of the limitations mentioned under Current Limitations except the training use case.
For more updates, please subscribe to discussion activity on RFC: https://github.com/apache/incubator-mxnet/issues/16431.
