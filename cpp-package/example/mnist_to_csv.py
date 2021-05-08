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

# Following file converts the mnist data to CSV format.
# Usage:
# mnist_to_csv.py train-images-idx3-ubyte train-labels-idx1-ubyte mnist_train.csv 60000
# mnist_to_csv.py t10k-images-idx3-ubyte t10k-labels-idx1-ubyte mnist_test.csv 10000
#

import argparse

def convert_to_csv(args):
    imageFile = open(args.imageFile, "rb")
    labelFile = open(args.labelFile, "rb")
    outputFile = open(args.outputFile, "w")

    imageFile.read(16)
    labelFile.read(8)
    images = []

    for i in range(args.num_records):
        image = [ord(labelFile.read(1))]
        for j in range(28 * 28):
            image.append(ord(imageFile.read(1)))
        images.append(image)

    for image in images:
        outputFile.write(",".join(str(pix) for pix in image) + "\n")

    imageFile.close()
    outputFile.close()
    labelFile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("imageFile", type=str, help="image file in mnist format e.g. train-images-idx3-ubyte")
    parser.add_argument("labelFile", type=str, help="label file in mnist format e.g train-labels-idx1-ubyte")
    parser.add_argument("outputFile", type=str, help="Output file in CSV format e.g mnist_train_trial.csv")
    parser.add_argument("num_records", type=int, help="Number of images in the input files.e.g 60000")
    args = parser.parse_args()

    try:
        convert_to_csv(args)
    except Exception as e:
        print("Error : Exception {}".format(str(e)))
