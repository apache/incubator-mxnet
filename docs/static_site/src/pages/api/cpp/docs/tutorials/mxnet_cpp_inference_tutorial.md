---
layout: page_api
title: C++ API inference tutorial
action: Get Started
action_url: /get_started
permalink: /api/cpp/docs/tutorials/cpp_inference.html
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

# C++ API inference tutorial

## Overview
MXNet provides various useful tools and interfaces for deploying your model for inference. For example, you can use [MXNet Model Server](https://github.com/awslabs/mxnet-model-server) to start a service and host your trained model easily.
Besides that, you can also use MXNet's different language APIs to integrate your model with your existing service. We provide [Python](/api/python/docs/api/), [Java](/api/java/docs/api/#package), [Scala](/api/scala/docs/api), and [C++](/api/cpp/docs/api/) APIs.
We will focus on the MXNet C++ API. We have slightly modified the code in [C++ Inference Example](https://github.com/apache/incubator-mxnet/tree/master/cpp-package/example/inference) for our use case.

## Prerequisites

To complete this tutorial, you need to:
- Complete the training part of [Gluon end to end tutorial](/api/python/docs/tutorials/getting-started/gluon_from_experiment_to_deployment.html).
- Learn the basics about [MXNet C++ API](/api/cpp).


## Setup the MXNet C++ API

To use the C++ API in MXNet, you need to build MXNet from source with C++ package. Please follow the [built from source guide](/get_started/ubuntu_setup.html), and [C++ Package documentation](/api/cpp).
The summary of those two documents is that you need to build MXNet from source with `USE_CPP_PACKAGE` flag set to 1. For example: `make -j USE_CPP_PACKAGE=1`.

## Load the model and run inference

After you complete [the previous tutorial](/api/python/docs/tutorials/getting-started/gluon_from_experiment_to_deployment.html), you will get the following output files:
1. Model Architecture stored in `flower-recognition-symbol.json`
2. Model parameter values stored in `flower-recognition-0040.params` (`0040` is for 40 epochs we ran)
3. Label names stored in `synset.txt`
4. Mean and standard deviation values stored in `mean_std_224` for image normalization.


Now we need to write the C++ code to load them and run prediction on a test image.
The full code is available in the [C++ Inference Example](https://github.com/apache/incubator-mxnet/tree/master/cpp-package/example/inference), we will walk you through it and point out the necessary changes to make for our use case.



### Write a predictor using the MXNet C++ API

In general, the C++ inference code should follow the 4 steps below. We can do that using a Predictor class.
1. Load the pre-trained model
2. Load the parameters of pre-trained model
3. Load the image to be classified in to NDArray and apply image transformation we did in training
4. Run the forward pass and predict the class of the input image

```c++
class Predictor {
 public:
    Predictor() {}
    Predictor(const std::string& model_json_file,
              const std::string& model_params_file,
              const Shape& input_shape,
              bool gpu_context_type = false,
              const std::string& synset_file = "",
              const std::string& mean_image_file = "");
    void PredictImage(const std::string& image_file);
    ~Predictor();

 private:
    void LoadModel(const std::string& model_json_file);
    void LoadParameters(const std::string& model_parameters_file);
    void LoadSynset(const std::string& synset_file);
    NDArray LoadInputImage(const std::string& image_file);
    void LoadMeanImageData();
    void LoadDefaultMeanImageData();
    void NormalizeInput(const std::string& mean_image_file);
    inline bool FileExists(const std::string& name) {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
    }
    NDArray mean_img;
    std::map<std::string, NDArray> args_map;
    std::map<std::string, NDArray> aux_map;
    std::vector<std::string> output_labels;
    Symbol net;
    Executor *executor;
    Shape input_shape;
    NDArray mean_image_data;
    NDArray std_dev_image_data;
    Context global_ctx = Context::cpu();
    std::string mean_image_file;
};
```

### Load the model, synset file, and normalization values

In the Predictor constructor, you need to provide paths to saved json and param files. After that, add the following methods `LoadModel` and `LoadParameters` to load the network and its parameters. This part is the same as [the example](https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/inference/imagenet_inference.cpp).

Next, we need to load synset file, and normalization values. We have made the following change since our synset file contains flower names and we used both mean and standard deviation for image normalization.

```c++
/*
 * The following function loads the synset file.
 * This information will be used later to report the label of input image.
 */
void Predictor::LoadSynset(const std::string& synset_file) {
  if (!FileExists(synset_file)) {
    LG << "Synset file " << synset_file << " does not exist";
    throw std::runtime_error("Synset file does not exist");
  }
  LG << "Loading the synset file.";
  std::ifstream fi(synset_file.c_str());
  if (!fi.is_open()) {
    std::cerr << "Error opening synset file " << synset_file << std::endl;
    throw std::runtime_error("Error in opening the synset file.");
  }
  std::string lemma;
  while (getline(fi, lemma)) {
    output_labels.push_back(lemma);
  }
  fi.close();
}

/*
 * The following function loads the mean and standard deviation values.
 * This data will be used for normalizing the image before running the forward
 * pass.
 * The output data has the same shape as that of the input image data.
 */
void Predictor::LoadMeanImageData() {
  LG << "Load the mean image data that will be used to normalize "
     << "the image before running forward pass.";
  mean_image_data = NDArray(input_shape, global_ctx, false);
  mean_image_data.SyncCopyFromCPU(
        NDArray::LoadToMap(mean_image_file)["mean_img"].GetData(),
        input_shape.Size());
  NDArray::WaitAll();
   std_dev_image_data = NDArray(input_shape, global_ctx, false);
   std_dev_image_data.SyncCopyFromCPU(
       NDArray::LoadToMap(mean_image_file)["std_img"].GetData(),
       input_shape.Size());
    NDArray::WaitAll();
}
```



### Load input image

Now let's add a method to load the input image we want to predict and converts it to NDArray for prediction.
```c++
NDArray Predictor::LoadInputImage(const std::string& image_file) {
  if (!FileExists(image_file)) {
    LG << "Image file " << image_file << " does not exist";
    throw std::runtime_error("Image file does not exist");
  }
  LG << "Loading the image " << image_file << std::endl;
  std::vector<float> array;
  cv::Mat mat = cv::imread(image_file);
  /*resize pictures to (224, 224) according to the pretrained model*/
  int height = input_shape[2];
  int width = input_shape[3];
  int channels = input_shape[1];
  cv::resize(mat, mat, cv::Size(height, width));
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        array.push_back(static_cast<float>(mat.data[(i * height + j) * 3 + c]));
      }
    }
  }
  NDArray image_data = NDArray(input_shape, global_ctx, false);
  image_data.SyncCopyFromCPU(array.data(), input_shape.Size());
  NDArray::WaitAll();
  return image_data;
}
```

### Predict the image

Finally, let's run the inference. It's basically using MXNet executor to do a forward pass. To run predictions on multiple images, you can load the images in a list of NDArrays and run prediction in batches. Note that the Predictor class may not be thread safe. Calling it in multi-threaded environments was not tested. To utilize multi-threaded prediction, you need to use the C predict API. Please follow the [C predict example](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification/predict-cpp).

An additional step is to normalize the image NDArrays values to `(0, 1)` and apply mean and standard deviation we just loaded.

```c++
/*
 * The following function runs the forward pass on the model.
 * The executor is created in the constructor.
 *
 */
void Predictor::PredictImage(const std::string& image_file) {
  // Load the input image
  NDArray image_data = LoadInputImage(image_file);

  // Normalize the image
  image_data.Slice(0, 1) /= 255.0;
  image_data -= mean_image_data;
  image_data /= std_dev_image_data;

  LG << "Running the forward pass on model to predict the image";
  /*
   * The executor->arg_arrays represent the arguments to the model.
   *
   * Copying the image_data that contains the NDArray of input image
   * to the arg map of the executor. The input is stored with the key "data" in the map.
   *
   */
  image_data.CopyTo(&(executor->arg_dict()["data"]));
  NDArray::WaitAll();

  // Run the forward pass.
  executor->Forward(false);

  // The output is available in executor->outputs.
  auto array = executor->outputs[0].Copy(global_ctx);
  NDArray::WaitAll();

  /*
   * Find out the maximum accuracy and the index associated with that accuracy.
   * This is done by using the argmax operator on NDArray.
   */
  auto predicted = array.ArgmaxChannel();
  NDArray::WaitAll();

  int best_idx = predicted.At(0, 0);
  float best_accuracy = array.At(0, best_idx);

  if (output_labels.empty()) {
    LG << "The model predicts the highest accuracy of " << best_accuracy << " at index "
       << best_idx;
  } else {
    LG << "The model predicts the input image to be a [" << output_labels[best_idx]
       << " ] with Accuracy = " << best_accuracy << std::endl;
  }
}
```

### Compile and run the inference code

You can find the [full code for the inference example](https://github.com/apache/incubator-mxnet/tree/master/cpp-package/example/inference) in the `cpp-package` folder of the project
, and to compile it use this [Makefile](https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/inference/Makefile).

Make a copy of the example code, rename it to `flower_inference` and apply the changes we mentioned above. Now you will be able to compile and run inference. Run `make all`. Once this is complete, run inference with the following parameters. Remember to set your `LD_LIBRARY_PATH` to point to MXNet library if you have not done so.

```bash
make all
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH=:path/to/incubator-mxnet/lib
./flower_inference --symbol flower-recognition-symbol.json --params flower-recognition-0040.params --synset synset.txt --mean mean_std_224.nd --image ./data/test/lotus/image_01832.jpg
```

Then it will predict your image:

```bash
[17:38:51] resnet.cpp:150: Loading the model from flower-recognition-symbol.json

[17:38:51] resnet.cpp:163: Loading the model parameters from flower-recognition-0040.params

[17:38:52] resnet.cpp:190: Loading the synset file.
[17:38:52] resnet.cpp:211: Load the mean image data that will be used to normalize the image before running forward pass.
[17:38:52] resnet.cpp:263: Loading the image ./data/test/lotus/image_01832.jpg

[17:38:52] resnet.cpp:299: Running the forward pass on model to predict the image
[17:38:52] resnet.cpp:331: The model predicts the input image to be a [lotus ] with Accuracy = 8.63046
```



## What's next

Now you can explore more ways to run inference and deploy your models:
1. [Java Inference examples](https://github.com/apache/incubator-mxnet/tree/master/scala-package/examples/src/main/java/org/apache/mxnetexamples/javaapi/infer)
2. [Scala Inference examples](https://github.com/apache/incubator-mxnet/tree/master/scala-package/examples/src/main/scala/org/apache/mxnetexamples/infer)
3. [ONNX model inference examples](/api/python/docs/tutorials/packages/onnx/inference_on_onnx_model.html)
4. [MXNet Model Server Examples](https://github.com/awslabs/mxnet-model-server/tree/master/examples)

## References

1. [Gluon end to end tutorial](/api/python/docs/tutorials/getting-started/gluon_from_experiment_to_deployment.html)
2. [Gluon C++ inference example](https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/inference/)
3. [Gluon C++ package](https://github.com/apache/incubator-mxnet/tree/master/cpp-package)
