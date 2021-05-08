<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
--->

# SNAIL(A Simple Neural Attentive Meta-Learner)

---

Gluon inplementation of [A Simple Neural Attentive Meta-Learner](https://openreview.net/pdf?id=B1DmUzWAW)

##### network structore
![net_structure](https://github.com/dmlc/web-data/blob/master/mxnet/example/gluon/snail/net_structure.png)

##### building block structure
![block_structure](https://github.com/dmlc/web-data/blob/master/mxnet/example/gluon/snail/blocks.png)

## Requirements
- Python 3.6.1
- mxnet 1.3.1
- mxboard 0.1.0
- tqdm 4.29.0


## Application
-  Omniglot

## Usage

- arguments
  - batch_size : Define batch size (defualt=64)
  - epochs : Define total epoches (default=50)
  - N : the nunber of N-way (default=10)
  - K : the number of K-shot (default=5)
  - iterations : the number of data iteration (default=1000)
  - input_dims : embedding dimension of input data (default=64)
  - download : download omniglot dataset (default=False)
  - GPU_COUNT : use gpu count  (default=1)
  - logdir : location of mxboard log file (default=./log)
  - modeldir : location of model parameter file (default=./models)


###### default setting
```
python main.py
``` 
or

###### manual setting
```
python main.py --batch_size=24 --epochs=200 ..
```

## Results
##### 10-way 5-shot case
![perf_acc](https://github.com/dmlc/web-data/blob/master/mxnet/example/gluon/snail/perf_acc.png)


## Reference
- https://github.com/sagelywizard/snail
- https://github.com/eambutu/snail-pytorch

