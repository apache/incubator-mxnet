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
# under the License
"""
Description : main module to run code
"""
import os
import argparse
from trainer import Train
# pylint: disable=no-member
def main():
    """
    Descrition : main module to run code
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--input_dims', type=int, default=64)
    parser.add_argument('--download', type=bool, default=False)
    parser.add_argument('--GPU_COUNT', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='./log')
    parser.add_argument('--modeldir', type=str, default='./models')
    config = parser.parse_args()

    # create output dir
    try:
        os.makedirs(config.logdir)
        os.makedirs(config.modeldir)
    except OSError:
        pass

    trainer = Train(config)

    trainer.train()
    if config.generation:
        trainer.generation()

if __name__ == "__main__":
    main()
