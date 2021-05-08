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

# MXNet Examples

This page contains a curated list of awesome MXNet examples, tutorials and blogs. It is inspired by [awesome-php](https://github.com/ziadoz/awesome-php) and [awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning). See also [Awesome-MXNet](https://github.com/chinakook/Awesome-MXNet) for a similar list.

  - [Contributing](#contributing)
  - [List of examples](#list-of-examples)
    - [Languages Binding Examples](#language-binding-examples)
    - [Deep Learning Examples in the MXNet Project Repository](#deep-learning-examples-mxnet)
    - [Other Deep Learning Examples with MXNet](#deep-learning-examples-other)
    - [IPython Notebooks](#ipython-notebooks)
    - [Mobile App Examples](#mobile-apps-examples)
    - [Web Predictive Services](#web-predictive-services)
  - [List of tutorials](#list-of-tutorials)
    - [GPU Technology Conference 2016 Hands-on session](#gtc2016-hands-on)
    - [Deep learning for hackers with MXnet](#deep-learning-for-hackers)
    - [MXnet setup on AWS](#mxnet-aws)
    - [Kaggle tutorials](#kaggle-tutorials)
    - [Learning Note](#learning-note)
  - [Machine Learning Challenge Winning Solutions](#winning-solutions)
  - [Tools with MXnet](#tools-with-mxnet)

## <a name="Contributing"></a>Contributing

If you want to contribute to this list and the examples, please open a new pull request.


### Examples

Example applications or scripts should be submitted in this `example` folder.


### Tutorials

If you have a tutorial idea for the website, download the [Jupyter notebook tutorial template](https://github.com/dmlc/mxnet/tree/master/example/MXNetTutorialTemplate.ipynb).

#### Tutorial location

Notebook tutorials should be submitted in the `docs/tutorials` folder, so that they maybe rendered in the [web site's tutorial section](https://mxnet.apache.org/tutorials/index.html).

Do not forget to udpdate the `docs/tutorials/index.md` for your tutorial to show up on the website.

#### Tutorial formatting

The site expects the format to be markdown, so export your notebook as a .md via the Jupyter web interface menu (File > Download As > Markdown). Then, to enable the download notebook button in the web site's UI ([example](https://mxnet.apache.org/tutorials/python/linear-regression.html)), add the following as the last line of the file ([example](https://github.com/apache/incubator-mxnet/blame/master/docs/tutorials/python/linear-regression.md#L194)):

```
<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
```

If you want some lines to show-up in the markdown but not in the generated notebooks, add  this comment `<!--notebook-skip-line-->` after your `![png](img_url)`. Like this:

```
![png](img_url.png)<!--notebook-skip-line-->
```

Typically when you have a `plt.imshow()` you want the image tag `[png](img.png)` in the `.md` but not in the downloaded notebook as the user will re-generate the plot at run-time.

#### Tutorial tests

As part of making sure all our tutorials are running correctly with the latest version of MXNet, each tutorial is run automatically through a python2 and python3 jupyter notebook kernel in the CI, in a GPU environment, checking for errors and warnings.

Add your own test here `tests/tutorials/test_tutorials.py`. (If you forget, don't worry your PR will not pass the sanity check).

If your tutorial depends on specific packages, simply add them to this provisionning script: `ci/docker/install/ubuntu_tutorials.sh`

## <a name="list-of-examples"></a>List of examples

### <a name="language-binding-examples"></a>Languages Binding Examples
------------------
* [MXNet C++ API](https://mxnet.apache.org/api/c++/index.html)
   - [C++ examples](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification/predict-cpp) - Example code for using C++ interface, including NDArray, symbolic layer and models.
* [MXNet Python API](https://mxnet.apache.org/api/python/index.html)
* [MXNet Java API](https://mxnet.apache.org/api/java/index.html)
* [MXNet Scala API](https://mxnet.apache.org/api/scala/index.html)
* [MXNet R API](https://mxnet.apache.org/api/r/index.html)
* [MXNet Julia API](https://mxnet.apache.org/api/julia/index.html)
* [MXNet Perl API](https://mxnet.apache.org/api/perl/index.html)
* [go-mxnet-predictor](https://github.com/songtianyi/go-mxnet-predictor) - Go binding for inference
* [MXNet Javascript](https://github.com/dmlc/mxnet.js/) - MXNetJS: Javascript Package for Deep Learning in Browser (without server)

### <a name="deep-learning-examples-mxnet"></a>Deep Learning Examples in the MXNet Project Repository
--------------
* [Autoencoder](autoencoder) - unsupervised feature learning
* [Gluon Examples](gluon) - several examples using the Gluon API
  * [Style Transfer](gluon/style_transfer) - a style transfer example using gluon
  * [Word Language Model](gluon/word_language_model) - an example that trains a multi-layer RNN on the Penn Treebank language modeling benchmark
  * [SN-GAN](gluon/sn_gan) - an example that utilizes spectral normalization to train GAN(Generative adversarial network) using Gluon API
* [Image Classification with R](image-classification) - image classification on MNIST,CIFAR,ImageNet-1k,ImageNet-Full, with multiple GPU and distributed training.
* [Kaggle 2nd national data science bowl](kaggle-ndsb2) - a tutorial for Kaggle Second Nation Data Science Bowl
* [Multi-task Learning](multi-task) - how to use MXNet for multi-task learning
* [Profiling](profiler) - generate profiling results in json files
* [Quantization and Calibration Examples](quantization) - examples of quantizing a FP32 model to INT8 and performing low-precision inference with Intel ONEDNN on CPU or cuDNN on GPU
* [Recommender Systems](recommenders) - examples of how to build various kinds of recommender systems
* [Restricted Boltzmann Machine](restricted-boltzmann-machine) - an example of the binary restricted Boltzmann machine learning MNIST
* [Single Shot MultiBox Detector](ssd) - SSD object recognition example

### <a name="deep-learning-examples-other"></a>Other Deep Learning Examples with MXNet

* [Face Recognition with ArcFace](https://github.com/onnx/models/tree/master/vision/body_analysis/arcface) - ONNX model for face recognition with notebooks for training, validating and running inference in MXNet by [abhinavs95](https://github.com/abhinavs95)
* [Chinese plate recognition](https://github.com/imistyrain/mxnet-mr) - Recognize Chinese vehicle plate, by [imistyrain](https://github.com/imistyrain)
* [Fast R-CNN](https://github.com/precedenceguo/mx-rcnn) by [Jian Guo](https://github.com/precedenceguo)
* "End2End Captcha Recognition (OCR)" by [xlvector](https://github.com/xlvector) [github link](https://github.com/xlvector/learning-dl/tree/master/mxnet/ocr) [Blog in Chinese](http://blog.xlvector.net/2016-05/mxnet-ocr-cnn/)
* "Prediction step of xlvector's lstm ocr" by [melody-rain](https://github.com/melody-rain) [github link](https://github.com/melody-rain/mxnet/commit/46002e31fc34c746c01bcaa7ade999187068ad3c) [Blog in Chinese](https://zhuanlan.zhihu.com/p/22698511)
* "Solving classification + regression with MXnet in Multi Input + Multi Obj" by [xlvector](https://github.com/xlvector) [github link](https://gist.github.com/xlvector/c304d74f9dd6a3b68a3387985482baac) [Blog in Chinese](http://blog.xlvector.net/2016-05/mxnet-regression-classification-for-concret-continuous-features/)
* "Learn to sort by LSTM" by [xlvector](https://github.com/xlvector) [github link](https://github.com/xlvector/learning-dl/tree/master/mxnet/lstm_sort) [Blog in Chinese](http://blog.xlvector.net/2016-05/mxnet-lstm-example/)
* [Neural Art using extremely lightweight (<500K) neural network](https://github.com/pavelgonchar/neural-art-mini) Lightweight version of mxnet neural art implementation by [Pavel Gonchar](https://github.com/pavelgonchar)
* [Neural Art with generative networks](https://github.com/zhaw/neural_style) by [zhaw](https://github.com/zhaw)
* [Faster R-CNN in MXNet with distributed implementation and data parallelization](https://github.com/dmlc/mxnet/tree/master/example/rcnn)
* [Asynchronous Methods for Deep Reinforcement Learning in MXNet](https://github.com/zmonoid/Asyn-RL-MXNet/blob/master/mx_asyn.py) by [zmonoid](https://github.com/zmonoid)
* [Deep Q-learning in MXNet](https://github.com/zmonoid/DQN-MXNet) by [zmonoid](https://github.com/zmonoid)
* [Face Detection with End-to-End Integration of a ConvNet and a 3D Model (ECCV16)](https://github.com/tfwu/FaceDetection-ConvNet-3D) by [tfwu](https://github.com/tfwu), source code for paper Yunzhu Li, Benyuan Sun, Tianfu Wu and Yizhou Wang, "Face Detection with End-to-End Integration of a ConvNet and a 3D Model", ECCV 2016 <https://arxiv.org/abs/1606.00850>
* [End-to-End Chinese plate recognition base on MXNet](https://github.com/szad670401/end-to-end-for-chinese-plate-recognition) by [szad670401](https://github.com/szad670401)
* [Reproduce ResNet-v2 (Identity Mappings in Deep Residual Networks) using MXNet](https://github.com/tornadomeet/ResNet) by [tornadomeet](https://github.com/tornadomeet)
* [Learning similarity among images in MXNet](http://www.jianshu.com/p/70a66c8f73d3) by xlvector in Chinese. GitHub [link](https://github.com/xlvector/learning-dl/tree/master/mxnet/triple-loss)
* [Matrix decomposition (SVD) with MXNet](http://www.jianshu.com/p/ebf7bf53ed3e) by xlvector in Chinese. GitHub [link](https://github.com/xlvector/mxnet/blob/svd/example/svd/svd.py)
* [MultiGPU enabled image generative models (GAN and DCGAN)](https://github.com/tqchen/mxnet-gan) by [Tianqi Chen](https://github.com/tqchen)
* [Deep reinforcement learning for playing flappybird by mxnet](https://github.com/li-haoran/DRL-FlappyBird) by LIHaoran
* [Neural Style in Markov Random Field (MRF) and Perceptual Losses Realtime transfer](https://github.com/zhaw/neural_style) by [zhaw](https://github.com/zhaw)
* [MTCNN Face keypoints detection and alignment](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection) by [yuanyang](https://github.com/YYuanAnyVision), source code for [paper](https://kpzhang93.github.io/papers/spl.pdf) "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks", [Kaipeng Zhang](https://github.com/kpzhang93), Zhanpeng Zhang, Zhifeng Li and Yu Qiao, IEEE Signal Processing Letters, 23(10), 2016
* [SSD: Single Shot MultiBox Object Detector](https://github.com/zhreshold/mxnet-ssd) by [zhreshold](https://github.com/zhreshold)
* [Fast Neural Style in Scala](https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/FastNeuralStyle) by [Ldpe2G](https://github.com/Ldpe2G)
* [LSTM Human Activity Recognition](https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/HumanActivityRecognition) by [Ldpe2G](https://github.com/Ldpe2G)
* [Visual Question Answering](https://github.com/liuzhi136/Visual-Question-Answering) by [liuzhi136](https://github.com/liuzhi136)
* [Deformable ConvNets](https://arxiv.org/abs/1703.06211) ([github](https://github.com/msracver/Deformable-ConvNets)) by [MSRACVer](https://github.com/msracver)
* [OCR with bi-LSTM and CTC Loss in Gluon](https://github.com/ThomasDelteil/Gluon_OCR_LSTM_CTC) by [ThomasDelteil](https://github.com/ThomasDelteil)
* [Visual Search with Gluon and HNSWlib](https://github.com/ThomasDelteil/VisualSearch_MXNet), by [ThomasDelteil](https://github.com/ThomasDelteil), online demo [here](https://thomasdelteil.github.io/VisualSearch_MXNet/)
* [MXNet-face](https://github.com/tornadomeet/mxnet-face) - Using MXNet for a face-related algorithm by [tornadomeet](https://github.com/tornadomeet) where the single model gets 97.13%+-0.88% accuracy on LFW, and with only 20MB size

### <a name="ipython-notebooks"></a>IPython Notebooks
-----------------
* [Predict with Pre-trained model](https://github.com/dmlc/mxnet-notebooks/blob/master/python/moved-from-mxnet/predict-with-pretrained-model.ipynb) - Notebook on how to predict with pretrained model.
* [composite symbol](https://github.com/dmlc/mxnet-notebooks/blob/master/python/moved-from-mxnet/composite_symbol.ipynb) - A demo of how to composite a symbolic Inception-BatchNorm Network
* [cifar-10 recipe](https://github.com/dmlc/mxnet-notebooks/blob/master/python/moved-from-mxnet/cifar10-recipe.ipynb) - A step by step demo of how to use MXNet
* [cifar-100](https://github.com/dmlc/mxnet-notebooks/blob/master/python/moved-from-mxnet/cifar-100.ipynb) - A demo of how to train a 75.68% accuracy CIFAR-100 model
* [simple bind](https://github.com/dmlc/mxnet-notebooks/blob/master/python/moved-from-mxnet/simple_bind.ipynb) - A demo of low level training API.
* [Multi task tutorial](https://github.com/haria/mxnet-multi-task-example/blob/master/multi-task.ipynb) - A demo of how to train and predict multi-task network on both MNIST and your own dataset.
* [class active maps](https://github.com/dmlc/mxnet-notebooks/blob/master/python/moved-from-mxnet/class_active_maps.ipynb) - A demo of how to localize the discriminative regions in an image using global average pooling (GAP) in CNNs.
* [DMLC MXNet Notebooks](https://github.com/dmlc/mxnet-notebooks) DMLC's repo for various notebooks ranging from basic usages of MXNet to state-of-the-art deep learning applications.
* [AWS Seoul Summit 2017 Demos](https://github.com/sxjscience/aws-summit-2017-seoul) The demo codes and ipython notebooks in AWS Seoul Summit 2017.
* [Character-level CNN for text classification](https://github.com/ThomasDelteil/CNN_NLP_MXNet) Performing category classification on Amazon reviews using Gluon and character-level Convolutional Neural Networks. Online demo [here](https://thomasdelteil.github.io/CNN_NLP_MXNet/)

### <a name="mobile-apps-examples"></a>Mobile App Examples
-------------------
* [MXNet Android Classification App](https://github.com/Leliana/WhatsThis) - Image classification on Android with MXNet.
* [MXNet iOS Classification App](https://github.com/pppoe/WhatsThis-iOS) - Image classification on iOS with MXNet.
* [Compile MXnet on Xcode (in Chinese)](http://www.liuxiao.org/2015/12/ios-mxnet-%E7%9A%84-ios-%E7%89%88%E6%9C%AC%E7%BC%96%E8%AF%91/) - a step-by-step tutorial of compiling MXnet on Xcode for iOS app

### <a name="web-predictive-services"></a>Web Predictive Services
-----------------------
* [MXNet Shinny](https://github.com/thirdwing/mxnet_shiny) - Source code for quickly creating a Shiny R app to host online image classification.
* [Machine Eye](http://rupeshs.github.io/machineye/) - Web service for local image file/image URL classification without uploading.

## <a name="list-of-tutorials"></a>List of tutorials

### <a name="gtc2016-hands-on"></a>GPU Technology Conference 2016 Hands-on session

* [Video on GTC 2016 site](http://on-demand.gputechconf.com/gtc/2016/video/L6143.html)
* [Video backup in Mainland China](http://pan.baidu.com/s/1eS58Gue)
* [iPython Notebook](https://github.com/dmlc/mxnet-gtc-tutorial)

### <a name="deep-learning-for-hackers"></a>Deep learning for hackers with MXNet

* Deep learning for hackers with MXNet (1) GPU installation and MNIST [English](https://no2147483647.wordpress.com/2015/12/07/deep-learning-for-hackers-with-mxnet-1/) [Chinese](http://phunter.farbox.com/post/mxnet-tutorial1) - a tutorial of installing MXnet with GPU and introduction to deep learning by MNIST example.
* Deep learning for hackers with MXNet (2): Neural art [English](https://no2147483647.wordpress.com/2015/12/21/deep-learning-for-hackers-with-mxnet-2/) [Chinese](http://phunter.farbox.com/post/mxnet-tutorial2) - a tutorial of generating Van Gogh style cat paintings.

### <a name="mxnet-aws"></a>MXNet on the cloud
* [Setup Amazon AWS GPU instance with MXnet](https://no2147483647.wordpress.com/2016/01/16/setup-amazon-aws-gpu-instance-with-mxnet/) - AWS GPU instance setup with GPU (CUDA with latest cuDNN and S3 support)
* [Intro Guide to AWS (MXNet with Julia)](http://www.datasciencebowl.com/aws_guide/) - A step-by-step guide of using spot instances with Amazon Web Services (AWS) to help you save money when training DSB models on MXNet by [Mike Kim](http://www.datasciencebowl.com/author/mikekim/)
* [Building Deep Neural Networks in the Cloud with Azure GPU VMs, MXNet and Microsoft R Server](https://blogs.technet.microsoft.com/machinelearning/2016/09/15/building-deep-neural-networks-in-the-cloud-with-azure-gpu-vms-mxnet-and-microsoft-r-server/) by [Cortana Intelligence and ML Blog Team](https://social.technet.microsoft.com/profile/Cortana+Intelligence+and+ML+Blog+Team) at Microsoft
* [Applying Deep Learning at Cloud Scale, with Microsoft R Server & Azure Data Lake](https://blogs.technet.microsoft.com/machinelearning/2016/10/31/applying-cloud-deep-learning-at-scale-with-microsoft-r-server-azure-data-lake/) by [Cortana Intelligence and ML Blog Team](https://social.technet.microsoft.com/profile/Cortana+Intelligence+and+ML+Blog+Team) at Microsoft
* [Training Deep Neural Neural Networks on ImageNet Using Microsoft R Server and Azure GPU VMs](https://blogs.technet.microsoft.com/machinelearning/2016/11/15/imagenet-deep-neural-network-training-using-microsoft-r-server-and-azure-gpu-vms/) by [Cortana Intelligence and ML Blog Team](https://social.technet.microsoft.com/profile/Cortana+Intelligence+and+ML+Blog+Team) at Microsoft
* [Cloud-Scale Text Classification with Convolutional Neural Networks on Microsoft Azure](https://blogs.technet.microsoft.com/machinelearning/2017/02/13/cloud-scale-text-classification-with-convolutional-neural-networks-on-microsoft-azure/) by [Cortana Intelligence and ML Blog Team](https://social.technet.microsoft.com/profile/Cortana+Intelligence+and+ML+Blog+Team) at Microsoft
* [Distributed Deep Learning Made Easy](https://aws.amazon.com/blogs/compute/distributed-deep-learning-made-easy/) at AWS/Amazon for deploying deep learning clusters using MXNet

### <a name="kaggle-tutorials"></a>Kaggle tutorials
* [Kaggle 2nd Annual Data Science Bowl End-to-End Deep Learning Tutorial (Python)](https://www.kaggle.com/c/second-annual-data-science-bowl/forums/t/18079/end-to-end-deep-learning-tutorial-0-0392) - an end-to-end python tutorial for Kaggle heart disease diagnose competition (public leaderboard score 0.0392)
* [Kaggle 2nd Annual Data Science Bowl End-to-End Deep Learning Tutorial (R)](https://www.kaggle.com/c/second-annual-data-science-bowl/forums/t/18122/deep-learning-model-in-r) - an end-to-end R tutorial for Kaggle heart disease diagnose competition
* [Dogs vs. Cats classification with mxnet and R](https://statist-bhfz.github.io/cats_dogs_finetune) - end-to-end (not winning) tutorial with an example of fine-tuning in R

### <a name="learning-note"></a>Learning Note
* [Learning Note in Chinese](https://github.com/zhubuntu/MXNet-Learning-Note) - MXNet learning note in Chinese.
* [Getting Started with MXNet](https://indico.io/blog/getting-started-with-mxnet/) by [indico.io](https://indico.io) (Chinese Translation [MXNet实践](http://www.infoq.com/cn/articles/practise-of-mxnet) by [侠天](http://www.infoq.com/cn/author/%E4%BE%A0%E5%A4%A9) )
* [{mxnet} R package from MXnet, an intuitive Deep Learning framework including CNN & RNN](http://tjo-en.hatenablog.com/entry/2016/03/30/233848) by [TJO](http://tjo-en.hatenablog.com/)
* [MXnet with R: combined power of deep learning](http://cos.name/2016/04/mxnet-r/) in Chinese by Tong He
* [Understand MXNet dependency engine](http://yuyang0.github.io/articles/mxnet-engine.html) in Chinese by [Yu Yang](https://github.com/yuyang0)

## <a name="winning-solutions"></a>Machine Learning Challenge Winning Solutions

* Dmitrii Tsybulevskii, 1st place of the [Yelp Restaurant Photo Classification](https://www.kaggle.com/c/yelp-restaurant-photo-classification). Link to [the Kaggle interview](http://blog.kaggle.com/2016/04/28/yelp-restaurant-photo-classification-winners-interview-1st-place-dmitrii-tsybulevskii/).

## <a name="tools-with-mxnet"></a>Tools with MXnet
* [TensorFuse](https://github.com/dementrock/tensorfuse) - Common interface for Theano, CGT, TensorFlow, and mxnet (experimental) by [dementrock](https://github.com/dementrock)
* [MXnet-face](https://github.com/tornadomeet/mxnet-face) - Using mxnet for face-related algorithm by [tornadomeet](https://github.com/tornadomeet) where the single model get 97.13%+-0.88% accuracy on LFW, and with only 20MB size.
* [MinPy](https://github.com/dmlc/minpy) - Pure numpy practice with third party operator Integration and MXnet as backend for GPU computing
* [MXNet Model Server](https://github.com/awslabs/mxnet-model-server) - a flexible and easy to use tool for serving Deep Learning models
