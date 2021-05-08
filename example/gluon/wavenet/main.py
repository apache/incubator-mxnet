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
Descrition : main module to run code
"""
import argparse
import numpy as np
from trainer import Train

def main():
    """
    Description : run wavenet code using argument info
    - batch_size : Define batch size (default=64)
    - epochs : Define the total number of epochs (default=1000)
    - mu : Define mu value for mu-law algorithm (default=128)
    - n_residue : Define number of residue (default=24)
    - n_skip : Define number of skip (default=128)
    - dilation_depth : Define dilation depth (default=10)
    - n_repeat : Define number of repeat (default=2)
    - seq_size : Define sequence size when generating data (default=20000)
    - use_gpu : use gpu for training
    - load_file : file name in loading wave file (default=parametric-2.wav)
    - save_file : file name in saving result (default='')
    """
    parser = argparse.ArgumentParser(description='argument for wavenet hyperparameters')
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--epochs', type=int, default=10, help="total number of epochs")
    parser.add_argument('--mu', type=int, default=128, help="mu value for mu-law algorithm")
    parser.add_argument('--n_residue', type=int, default=24, help="number of residue")
    parser.add_argument('--n_skip', type=int, default=128, help="number of skip")
    parser.add_argument('--dilation_depth', type=int, default=10, help="number of dilation depth")
    parser.add_argument('--n_repeat', type=int, default=2, help="number of repeat")
    parser.add_argument('--seq_size', type=int, default=20000, help="number of sequence size")
    parser.add_argument('--use_gpu', action='store_true', help='use gpu for training.')
    parser.add_argument('--load_file', type=str, default='parametric-2.wav', help="file name in loading wave file")
    parser.add_argument('--save_file', type=str, default='', help="file name in saving result")
    config = parser.parse_args()
    print(config)
    np.save('./models/commandline_args.npy', config.__dict__)
    trainer = Train(config)
    trainer.train()

if __name__ == "__main__":
    main()
