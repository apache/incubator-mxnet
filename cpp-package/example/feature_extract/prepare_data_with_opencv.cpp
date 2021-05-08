/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 */
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;

/*read images and store them the NDArray format that MXNet.cpp can handle*/
void Mat2Array() {
  string file_name_list[] = {"./dog.jpg", "./cat.jpg"};

  std::vector<float> array;
  for (auto &t : file_name_list) {
    cv::Mat mat = cv::imread(t);
    /*resize pictures to (224, 224) according to the pretrained model*/
    cv::resize(mat, mat, cv::Size(224, 224));
    for (int c = 0; c < 3; ++c) {
      for (int i = 0; i < 224; ++i) {
        for (int j = 0; j < 224; ++j) {
          array.push_back(static_cast<float>(mat.data[(i * 224 + j) * 3 + c]));
        }
      }
    }
  }
  ofstream outf("./img.dat", ios::binary);
  outf.write(reinterpret_cast<char *>(array.data()), array.size() * sizeof(float));
  outf.close();
}

int main(int argc, char *argv[]) {
  Mat2Array();
  return 0;
}
