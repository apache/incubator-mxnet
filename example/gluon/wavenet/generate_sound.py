# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Descrition : main module to generation code
"""
import argparse
import numpy as np
from trainer import Train

def main():
    """
    Description : run wavenet code using argument info
    - seq_size : Define sequence size when generating data (default=3000)
    - use_gpu : use gpu for training
    - model_path : path for best model weigh
    - gen_size : length for data generation (default=10000)
    - save_file : file name in saving result (default=wav.npy)
    """
    parser = argparse.ArgumentParser(description='argument for wavenet generation hyperparameters')
    parser.add_argument('--seq_size', type=int, default=3000, help="number of sequence size")
    parser.add_argument('--use_gpu', action='store_true', help='use gpu for training.')
    parser.add_argument('--model_path', type=str,\
                            default='./models/best_perf_epoch_963_loss_9.889945983886719',\
                            help='path for best model weight')
    parser.add_argument('--gen_size', type=int, default=10000, help='length for data generation')
    parser.add_argument('--save_file', type=str, default='wav.npy',\
                        help="file name in saving result")
    config_gen = parser.parse_args()
    print(type(config_gen))
    config_dict = np.load('./models/commandline_args.npy').item()
    from argparse import Namespace
    config = Namespace(**config_dict)
    config.seq_size = config_gen.seq_size
    config.use_gpu = config_gen.use_gpu
    config.model_path = config_gen.model_path
    config.gen_size = config_gen.gen_size
    config.save_file = config_gen.save_file
    print(config)
    trainer = Train(config)
    trainer.generation(config.model_path, config.gen_size)
if __name__ == "__main__":
    main()
    