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

# WaveNet with Gluon

Gluon implementation of [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)

![net_structure1](https://github.com/dmlc/web-data/blob/master/mxnet/example/gluon/wavenet/net_struc1.png)
![net_structure2](https://github.com/dmlc/web-data/blob/master/mxnet/example/gluon/wavenet/net_struc2.png)

## Dataset
- Deepmind WaveNet blog [link](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)
   - use US English data [audio link](https://storage.googleapis.com/deepmind-media/pixie/us-english/parametric-2.wav)

## Requirements
- python 3.6.1
- mxnet 1.4.0
- tqdm 4.29.0
- scipy 1.2.0
- numpy 1.16.2

## Training

- arguments
  - batch_size : Define batch size (default=64)
  - epochs : Define the total number of epochs (default=1000)
  - mu : Define mu value for [mu-law algorithm](https://en.wikipedia.org/wiki/%CE%9C-law_algorithm) (default=128)
  - n_residue : Define number of residue (default=24)
  - dilation_depth : Define dilation depth (default=10)
  - n_repeat : Define number of repeat (default=2)
  - seq_size : Define sequence size when generating data (default=20000)
  - use_gpu : use gpu for training
  - load_file : file name in loading wave file (default=parametric-2.wav)
  - save_file : file name in saving result (default='')

###### default setting
```
python main.py --use_gpu
``` 
or

###### manual setting
```
python main.py --use_gpu --batch_size=32 --epochs=100 ...
```

## Inference

- arguments
  - seq_size : Define sequence size when generating data (default=3000)
  - use_gpu : use gpu for training
  - model_path : path for best model weigh
  - gen_size : length for data generation (default=10000)
  - save_file : file name in saving result (default=wav.npy)

###### default setting
```
python generate_sound.py --use_gpu
``` 
or

###### manual setting
```
python generate_sound.py --use_gpu --seq_size 3000 ...
```

## Train progress
###### 0 epoch
![epoch0](https://github.com/dmlc/web-data/blob/master/mxnet/example/gluon/wavenet/progress_epoch0.png)

###### 200 epoch
![epoch200](https://github.com/dmlc/web-data/blob/master/mxnet/example/gluon/wavenet/progress_epoch200.png)

###### 400 epoch
![epoch400](https://github.com/dmlc/web-data/blob/master/mxnet/example/gluon/wavenet/progress_epoches400.png)


## Results
![perf_loss](https://github.com/dmlc/web-data/blob/master/mxnet/example/gluon/wavenet/loss.png)


## Samples
- ground truth [link](https://soundcloud.com/seung-hwan-jung-375239472/us-english-ground-truth)
- official result from deepmind [link](https://soundcloud.com/seung-hwan-jung-375239472/official-result-from-deepmind)
- result using this code [link](https://soundcloud.com/seung-hwan-jung-375239472/wavenet-gen-rst)

## References
- WaveNet: A Generative Model for Raw Audio | Deepmind [blog](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)
- WaveNet generation code using pytorch [link](https://gist.github.com/lirnli/4282fcdfb383bb160cacf41d8c783c70#file-pytorch-wavenet-ipynb)

