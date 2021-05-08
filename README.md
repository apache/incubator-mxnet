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

<div align="center">
  <a href="https://mxnet.apache.org/"><img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mxnet_logo_2.png"></a><br>
</div>

[![banner](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/banner.png)](https://mxnet.apache.org)

Apache MXNet (incubating) for Deep Learning
===========================================
[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/apache/incubator-mxnet)](https://github.com/apache/incubator-mxnet/releases) [![GitHub stars](https://img.shields.io/github/stars/apache/incubator-mxnet)](https://github.com/apache/incubator-mxnet/stargazers) [![GitHub forks](https://img.shields.io/github/forks/apache/incubator-mxnet)](https://github.com/apache/incubator-mxnet/network) [![GitHub contributors](https://img.shields.io/github/contributors-anon/apache/incubator-mxnet)](https://github.com/apache/incubator-mxnet/graphs/contributors) [![GitHub issues](https://img.shields.io/github/issues/apache/incubator-mxnet)](https://github.com/apache/incubator-mxnet/issues) [![good first issue](https://img.shields.io/github/issues/apache/incubator-mxnet/good%20first%20issue)](https://github.com/apache/incubator-mxnet/labels/good%20first%20issue) [![GitHub pull requests by-label](https://img.shields.io/github/issues-pr/apache/mxnet/pr-awaiting-review)](https://github.com/apache/mxnet/labels/pr-awaiting-review) [![GitHub license](https://img.shields.io/github/license/apache/incubator-mxnet)](https://github.com/apache/incubator-mxnet/blob/master/LICENSE) [![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fapache%2Fincubator-mxnet)](https://twitter.com/intent/tweet?text=Wow:%20https%3A%2F%2Fgithub.com%2Fapache%2Fincubator-mxnet%20@ApacheMXNet) [![Twitter Follow](https://img.shields.io/twitter/follow/ApacheMXNet?style=social)](https://twitter.com/ApacheMXNet)

Apache MXNet is a deep learning framework designed for both *efficiency* and *flexibility*.
It allows you to ***mix*** [symbolic and imperative programming](https://mxnet.apache.org/api/architecture/program_model)
to ***maximize*** efficiency and productivity.
At its core, MXNet contains a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly.
A graph optimization layer on top of that makes symbolic execution fast and memory efficient.
MXNet is portable and lightweight, scalable to many GPUs and machines.

MXNet is more than a deep learning project. It is a [community](https://mxnet.apache.org/versions/master/community)
on a mission of democratizing AI. It is a collection of [blue prints and guidelines](https://mxnet.apache.org/api/architecture/overview)
for building deep learning systems, and interesting insights of DL systems for hackers.

Licensed under an [Apache-2.0](https://github.com/apache/incubator-mxnet/blob/master/LICENSE) license.

| Branch  | Build Status  |
|:-------:|:-------------:|
| [master](https://github.com/apache/incubator-mxnet/tree/master) | [![CentOS CPU Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/centos-cpu/job/master/badge/icon?subject=build%20centos%20cpu)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/centos-cpu/job/master/) [![CentOS GPU Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/centos-gpu/job/master/badge/icon?subject=build%20centos%20gpu)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/centos-gpu/job/master/) [![Clang Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/clang/job/master/badge/icon?subject=build%20clang)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/clang/job/master/) <br> [![Edge Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/edge/job/master/badge/icon?subject=build%20edge)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/edge/job/master/) [![Miscellaneous Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/miscellaneous/job/master/badge/icon?subject=build%20miscellaneous)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/miscellaneous/job/master/) [![Sanity Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/sanity/job/master/badge/icon?subject=build%20sanity)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/sanity/job/master/) <br> [![Unix CPU Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/unix-cpu/job/master/badge/icon?subject=build%20unix%20cpu)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/unix-cpu/job/master/) [![Unix GPU Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/unix-gpu/job/master/badge/icon?subject=build%20unix%20gpu)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/unix-gpu/job/master/) [![Website Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/website/job/master/badge/icon?subject=build%20website)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/website/job/master/) <br> [![Windows CPU Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/windows-cpu/job/master/badge/icon?subject=build%20windows%20cpu)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/windows-cpu/job/master/) [![Windows GPU Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/windows-gpu/job/master/badge/icon?subject=build%20windows%20gpu)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/windows-gpu/job/master/) [![Documentation Status](http://jenkins.mxnet-ci.amazon-ml.com/job/restricted-website-build/badge/icon)](https://mxnet.apache.org/) |
| [v1.x](https://github.com/apache/incubator-mxnet/tree/v1.x) | [![CentOS CPU Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/centos-cpu/job/v1.x/badge/icon?subject=build%20centos%20cpu)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/centos-cpu/job/v1.x/) [![CentOS GPU Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/centos-gpu/job/v1.x/badge/icon?subject=build%20centos%20gpu)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/centos-gpu/job/v1.x/) [![Clang Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/clang/job/v1.x/badge/icon?subject=build%20clang)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/clang/job/v1.x/) <br> [![Edge Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/edge/job/v1.x/badge/icon?subject=build%20edge)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/edge/job/v1.x/) [![Miscellaneous Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/miscellaneous/job/v1.x/badge/icon?subject=build%20miscellaneous)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/miscellaneous/job/v1.x/) [![Sanity Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/sanity/job/v1.x/badge/icon?subject=build%20sanity)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/sanity/job/v1.x/) <br> [![Unix CPU Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/unix-cpu/job/v1.x/badge/icon?subject=build%20unix%20cpu)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/unix-cpu/job/v1.x/) [![Unix GPU Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/unix-gpu/job/v1.x/badge/icon?subject=build%20unix%20gpu)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/unix-gpu/job/v1.x/) [![Website Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/website/job/v1.x/badge/icon?subject=build%20website)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/website/job/v1.x/) <br> [![Windows CPU Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/windows-cpu/job/v1.x/badge/icon?subject=build%20windows%20cpu)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/windows-cpu/job/v1.x/) [![Windows GPU Build Status](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/windows-gpu/job/v1.x/badge/icon?subject=build%20windows%20gpu)](http://jenkins.mxnet-ci.amazon-ml.com/job/mxnet-validation/job/windows-gpu/job/v1.x/) [![Documentation Status](http://jenkins.mxnet-ci.amazon-ml.com/job/restricted-website-build/badge/icon)](https://mxnet.apache.org/) |

Features
--------
* NumPy-like programming interface, and is integrated with the new, easy-to-use Gluon 2.0 interface. NumPy users can easily adopt MXNet and start in deep learning.
* Automatic hybridization provides imperative programming with the performance of traditional symbolic programming.
* Lightweight, memory-efficient, and portable to smart devices through native cross-compilation support on ARM, and through ecosystem projects such as [TVM](https://tvm.ai), [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html), [OpenVINO](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html).
* Scales up to multi GPUs and distributed setting with auto parallelism through [ps-lite](https://github.com/dmlc/ps-lite), [Horovod](https://github.com/horovod/horovod), and [BytePS](https://github.com/bytedance/byteps).
* Extensible backend that supports full customization, allowing integration with custom accelerator libraries and in-house hardware without the need to maintain a fork.
* Support for [Python](https://mxnet.apache.org/api/python), [Java](https://mxnet.apache.org/api/java), [C++](https://mxnet.apache.org/api/cpp), [R](https://mxnet.apache.org/api/r), [Scala](https://mxnet.apache.org/api/scala), [Clojure](https://mxnet.apache.org/api/clojure), [Go](https://github.com/jdeng/gomxnet/), [Javascript](https://github.com/dmlc/mxnet.js/), [Perl](https://mxnet.apache.org/api/perl), and [Julia](https://mxnet.apache.org/api/julia).
* Cloud-friendly and directly compatible with AWS and Azure.

Contents
--------
* [Installation](https://mxnet.apache.org/get_started)
* [Tutorials](https://mxnet.apache.org/api/python/docs/tutorials/)
* [Ecosystem](https://mxnet.apache.org/ecosystem)
* [API Documentation](https://mxnet.apache.org/api)
* [Examples](https://github.com/apache/incubator-mxnet-examples)
* [Stay Connected](#stay-connected)
* [Social Media](#social-media)

What's New
----------
* [1.8.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.8.0) - MXNet 1.8.0 Release.
* [1.7.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.7.0) - MXNet 1.7.0 Release.
* [1.6.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.6.0) - MXNet 1.6.0 Release.
* [1.5.1 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.5.1) - MXNet 1.5.1 Patch Release.
* [1.5.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.5.0) - MXNet 1.5.0 Release.
* [1.4.1 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.4.1) - MXNet 1.4.1 Patch Release.
* [1.4.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.4.0) - MXNet 1.4.0 Release.
* [1.3.1 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.3.1) - MXNet 1.3.1 Patch Release.
* [1.3.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.3.0) - MXNet 1.3.0 Release.
* [1.2.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.2.0) - MXNet 1.2.0 Release.
* [1.1.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.1.0) - MXNet 1.1.0 Release.
* [1.0.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/1.0.0) - MXNet 1.0.0 Release.
* [0.12.1 Release](https://github.com/apache/incubator-mxnet/releases/tag/0.12.1) - MXNet 0.12.1 Patch Release.
* [0.12.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/0.12.0) - MXNet 0.12.0 Release.
* [0.11.0 Release](https://github.com/apache/incubator-mxnet/releases/tag/0.11.0) - MXNet 0.11.0 Release.
* [Apache Incubator](http://incubator.apache.org/projects/mxnet.html) - We are now an Apache Incubator project.
* [0.10.0 Release](https://github.com/dmlc/mxnet/releases/tag/v0.10.0) - MXNet 0.10.0 Release.
* [0.9.3 Release](./docs/architecture/release_note_0_9.md) - First 0.9 official release.
* [0.9.1 Release (NNVM refactor)](./docs/architecture/release_note_0_9.md) - NNVM branch is merged into master now. An official release will be made soon.
* [0.8.0 Release](https://github.com/dmlc/mxnet/releases/tag/v0.8.0)

### Ecosystem News

* [ONEDNN for Faster CPU Performance](docs/python_docs/python/tutorials/performance/backend/mkldnn/mkldnn_readme.md)
* [MXNet Memory Monger, Training Deeper Nets with Sublinear Memory Cost](https://github.com/dmlc/mxnet-memonger)
* [Tutorial for NVidia GTC 2016](https://github.com/dmlc/mxnet-gtc-tutorial)
* [MXNet.js: Javascript Package for Deep Learning in Browser (without server)](https://github.com/dmlc/mxnet.js/)
* [Guide to Creating New Operators (Layers)](https://mxnet.apache.org/api/faq/new_op)
* [Go binding for inference](https://github.com/songtianyi/go-mxnet-predictor)

Stay Connected
--------------

| Channel | Purpose |
|---|---|
| [Follow MXNet Development on GitHub](https://github.com/apache/incubator-mxnet/issues) | See what's going on in the MXNet project. |
| [MXNet Confluence Wiki for Developers](https://cwiki.apache.org/confluence/display/MXNET/Apache+MXNet+Home) <i class="fas fa-external-link-alt"> | MXNet developer wiki for information related to project development, maintained by contributors and developers. To request write access, send an email to [send request to the dev list](mailto:dev@mxnet.apache.org?subject=Requesting%20CWiki%20write%20access) <i class="far fa-envelope"></i>. |
| [dev@mxnet.apache.org mailing list](https://lists.apache.org/list.html?dev@mxnet.apache.org) | The "dev list". Discussions about the development of MXNet. To subscribe, send an email to [dev-subscribe@mxnet.apache.org](mailto:dev-subscribe@mxnet.apache.org) <i class="far fa-envelope"></i>. |
| [discuss.mxnet.io](https://discuss.mxnet.io) <i class="fas fa-external-link-alt"></i> | Asking & answering MXNet usage questions. |
| [Apache Slack #mxnet Channel](https://the-asf.slack.com/archives/C7FN4FCP9) <i class="fas fa-external-link-alt"> | Connect with MXNet and other Apache developers. To join the MXNet slack channel [send request to the dev list](mailto:dev@mxnet.apache.org?subject=Requesting%20slack%20access) <i class="far fa-envelope"></i>. |
| [Follow MXNet on Social Media](#social-media) | Get updates about new features and events. |


### Social Media

Keep connected with the latest MXNet news and updates.

<p>
<a href="https://twitter.com/apachemxnet"><img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/social/twitter.svg?sanitize=true" height="30px"/> Apache MXNet on Twitter</a>
</p>
<p>
<a href="https://medium.com/apache-mxnet"><img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/social/medium_black.svg?sanitize=true" height="30px"/> Contributor and user blogs about MXNet</a>
</p>
<p>
<a href="https://reddit.com/r/mxnet"><img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/social/reddit_blue.svg?sanitize=true" height="30px" alt="reddit"/> Discuss MXNet on r/mxnet</a>
</p>
<p>
<a href="https://www.youtube.com/apachemxnet"><img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/social/youtube_red.svg?sanitize=true" height="30px"/> Apache MXNet YouTube channel</a>
</p>
<p>
<a href="https://www.linkedin.com/company/apache-mxnet"><img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/social/linkedin.svg?sanitize=true" height="30px"/> Apache MXNet on LinkedIn</a>
</p>


History
-------
MXNet emerged from a collaboration by the authors of [cxxnet](https://github.com/dmlc/cxxnet), [minerva](https://github.com/dmlc/minerva), and [purine2](https://github.com/purine/purine2). The project reflects what we have learned from the past projects. MXNet combines aspects of each of these projects to achieve flexibility, speed, and memory efficiency.

Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao,
Bing Xu, Chiyuan Zhang, and Zheng Zhang.
[MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems](https://github.com/dmlc/web-data/raw/master/mxnet/paper/mxnet-learningsys.pdf).
In Neural Information Processing Systems, Workshop on Machine Learning Systems, 2015
