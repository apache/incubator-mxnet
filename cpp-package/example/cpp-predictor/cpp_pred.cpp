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

#include <string>
#include <vector>
#include <cassert>

#include "mxnet-cpp/MxNetCpp.h"
#include <opencv/cv.hpp>

#include "predictor.h"

int main() {
    cv::Mat image = cv::imread("test.png");
    image.convertTo(image, CV_32FC3, 1 / 255.);

    mx_uint ch = image.channels();
    mx_uint width = image.cols;
    mx_uint height = image.rows;

    size_t image_size = width * height;

    std::vector<cv::Mat> rgb(3), dst(3);
    cv::split(image, rgb);

    mx_float *data = new mx_float[image_size * ch];
    memcpy(data + image_size * 0, rgb[2].data, image_size * sizeof(mx_float));
    memcpy(data + image_size * 1, rgb[1].data, image_size * sizeof(mx_float));
    memcpy(data + image_size * 2, rgb[0].data, image_size * sizeof(mx_float));

    try {
        std::string symbol_json = "somesr2x_int8_quantized-symbol.json";
        std::string param_data = "somesr2x_int8_quantized-0000.params";

        auto ctx = Context::gpu(0);
        auto input_shape = Shape{ 1, ch, height, width };

        Predictor pred(symbol_json, param_data, input_shape, ctx, MX_FLOAT32);

        if (!pred.SetInput(data, input_shape.Size())) {
            throw std::exception(MXGetLastError());
        }

        pred.Forward();

        auto output_shape = pred.GetOutputShape(0);

        auto buf = new mx_float[output_shape.Size()];
        pred.GetOutput(0, buf, output_shape.Size() * sizeof(mx_float));

        dst[2] = cv::Mat(height * 2, width * 2, CV_32F, buf + image_size * 4 * 0);
        dst[1] = cv::Mat(height * 2, width * 2, CV_32F, buf + image_size * 4 * 1);
        dst[0] = cv::Mat(height * 2, width * 2, CV_32F, buf + image_size * 4 * 2);

        cv::Mat sr;
        cv::merge(dst, sr);

        sr.convertTo(sr, CV_8UC3, 255.0);
        cv::imwrite("result.png", sr);
    } catch (const dmlc::Error &e) {
        printf("Error: %s", MXGetLastError());
    } catch (const std::exception &e) {
        printf("Error: %s", e.what());
    }

    return 0;
}
