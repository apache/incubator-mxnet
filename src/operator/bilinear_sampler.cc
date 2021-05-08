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
 * Copyright (c) 2017 by Contributors
 * \file bilinear_sampler.cc
 * \brief
 * \author Xu Dong
*/

#include <cmath>

#include "./bilinear_sampler-inl.h"

namespace mshadow {
template<typename DType>
bool between(DType value, int lowerBound, int upperBound) {
  return (value >= lowerBound && value <= upperBound);
}
template<typename DType>
inline void BilinearSamplerForward(const Tensor<cpu, 4, DType> &output,
                                    const Tensor<cpu, 4, DType> &input,
                                    const Tensor<cpu, 4, DType> &grid_src) {
  DType *out = output.dptr_;
  const DType *data = input.dptr_;
  const DType *grid = grid_src.dptr_;
  int o_n = output.size(0), o_c = output.size(1), o_h = output.size(2), o_w = output.size(3);
  int i_c = input.size(1), i_h = input.size(2), i_w = input.size(3);
  for (index_t n = 0; n < static_cast<index_t>(o_n); ++n) {
    for (index_t c = 0; c < static_cast<index_t>(o_c); ++c) {
      for (index_t h = 0; h < static_cast<index_t>(o_h); ++h) {
        for (index_t w = 0; w < static_cast<index_t>(o_w); ++w) {
          index_t out_index = n * o_c * o_h * o_w + c * o_h * o_w + h * o_w + w;
          index_t grid_index = n * o_h * o_w * 2 + h * o_w + w;
          DType y_real = (*(grid + grid_index + o_h * o_w) + 1) * (i_h - 1) / 2;
          DType x_real = (*(grid + grid_index) + 1) * (i_w - 1) / 2;
          // NOLINTNEXTLINE
          int top_left_y = static_cast<int>(std::floor(y_real));
          // NOLINTNEXTLINE
          int top_left_x = static_cast<int>(std::floor(x_real));
          DType top_left_y_w = 1.0 - (y_real - top_left_y);
          DType top_left_x_w = 1.0 - (x_real - top_left_x);
          int data_index = n * i_c * i_h * i_w + c * i_h * i_w +
            top_left_y * i_w + top_left_x;
          DType top_left_v = 0;
          DType top_right_v = 0;
          DType bottom_left_v = 0;
          DType bottom_right_v = 0;
          if (between(top_left_x, 0, i_w-1) && between(top_left_y, 0, i_h-1))
            top_left_v = *(data + data_index);
          if (between(top_left_x + 1, 0, i_w-1) && between(top_left_y, 0, i_h-1))
            top_right_v = *(data + data_index + 1);
          if (between(top_left_x, 0, i_w-1) && between(top_left_y + 1, 0, i_h-1))
            bottom_left_v = *(data + data_index + i_w);
          if (between(top_left_x+1, 0, i_w-1) && between(top_left_y + 1, 0, i_h-1))
            bottom_right_v = *(data + data_index + i_w + 1);
          *(out+out_index) = top_left_v * top_left_y_w * top_left_x_w +
                              top_right_v * top_left_y_w * (1.0 - top_left_x_w) +
                              bottom_left_v * (1.0 - top_left_y_w) * top_left_x_w +
                              bottom_right_v * (1.0 - top_left_y_w) * (1.0 - top_left_x_w);
        }
      }
    }
  }
}

template<typename DType>
inline void BilinearSamplerBackward(const Tensor<cpu, 4, DType> &gdata,
                                    const Tensor<cpu, 4, DType> &ggrid,
                                    const Tensor<cpu, 4, DType> &output_grad,
                                    const Tensor<cpu, 4, DType> &input_data,
                                    const Tensor<cpu, 4, DType> &grid,
                                    const mxnet::OpReqType data_req,
                                    const mxnet::OpReqType grid_req) {
  DType *g_input = gdata.dptr_;
  DType *grad_grid = ggrid.dptr_;
  const DType *grid_src = grid.dptr_;
  const DType *grad = output_grad.dptr_;
  const DType *data = input_data.dptr_;
  int o_n = output_grad.size(0), o_c = output_grad.size(1),
      o_h = output_grad.size(2), o_w = output_grad.size(3);
  int i_c = input_data.size(1), i_h = input_data.size(2), i_w = input_data.size(3);
  for (index_t n = 0; n < static_cast<index_t>(o_n); ++n) {
     for (index_t h = 0; h < static_cast<index_t>(o_h); ++h) {
        for (index_t w = 0; w < static_cast<index_t>(o_w); ++w) {
          DType top_left_y_gw = 0.0;
          DType top_left_x_gw = 0.0;
          index_t grid_src_index = n * o_h * o_w * 2 + h * o_w + w;
          DType y_real = (*(grid_src + grid_src_index + o_h * o_w) + 1) * (i_h - 1) / 2;
          DType x_real = (*(grid_src + grid_src_index) + 1) * (i_w - 1) / 2;
          // NOLINTNEXTLINE
          int top_left_y = static_cast<int>(std::floor(y_real));
          // NOLINTNEXTLINE
          int top_left_x = static_cast<int>(std::floor(x_real));
          DType top_left_y_w = 1.0 - (y_real - top_left_y);
          DType top_left_x_w = 1.0 - (x_real - top_left_x);
          for (index_t c = 0; c < static_cast<index_t>(o_c); ++c) {
            index_t grad_index = n * o_c * o_h * o_w + c * o_h * o_w + h * o_w + w;
            int data_index = n * i_c * i_h * i_w + c * i_h * i_w + top_left_y * i_w + top_left_x;
            // calc 4 vertex value in input data
            DType top_left_v = 0;
            DType top_right_v = 0;
            DType bottom_left_v = 0;
            DType bottom_right_v = 0;
            // calc input grad
            if (between(top_left_x, 0, i_w-1) && between(top_left_y, 0, i_h-1)) {
              if (data_req != mxnet::kNullOp) {
                *(g_input + data_index) += *(grad + grad_index) * top_left_y_w * top_left_x_w;
              }
              top_left_v = *(data + data_index);
            }
            if (between(top_left_x+1, 0, i_w-1) && between(top_left_y, 0, i_h-1)) {
              if (data_req != mxnet::kNullOp) {
                *(g_input + data_index + 1) +=
                  *(grad + grad_index) * top_left_y_w * (1.0 - top_left_x_w);
              }
              top_right_v = *(data + data_index + 1);
            }
            if (between(top_left_x, 0, i_w-1) && between(top_left_y+1, 0, i_h-1)) {
              if (data_req != mxnet::kNullOp) {
                *(g_input + data_index+ i_w) +=
                  *(grad + grad_index) * (1.0 - top_left_y_w) * top_left_x_w;
              }
              bottom_left_v = *(data + data_index + i_w);
            }
            if (between(top_left_x+1, 0, i_w-1) && between(top_left_y+1, 0, i_h-1)) {
              if (data_req != mxnet::kNullOp) {
                *(g_input + data_index+ i_w + 1) +=
                  *(grad + grad_index) * (1.0 - top_left_y_w) * (1.0 - top_left_x_w);
              }
              bottom_right_v = *(data + data_index + i_w + 1);
            }
            // calc weight grad of top_left_w, then multiple -1 is the grad of grid_src
            top_left_y_gw -= *(grad + grad_index) * (top_right_v - bottom_right_v +
                              (top_left_v - top_right_v - bottom_left_v + bottom_right_v)
                              * top_left_x_w);
            top_left_x_gw -= *(grad + grad_index) * (bottom_left_v - bottom_right_v +
                              (top_left_v - top_right_v - bottom_left_v + bottom_right_v)
                              * top_left_y_w);
          }
          if (grid_req != mxnet::kNullOp) {
            // calc grad of grid
            *(grad_grid + grid_src_index + o_h * o_w) += top_left_y_gw * (i_h - 1) / 2;
            *(grad_grid + grid_src_index) += top_left_x_gw * (i_w - 1) / 2;
          }
        }
      }
    }
  }
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(BilinearSamplerParam param, int dtype) {
  Operator *op = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new BilinearSamplerOp<cpu, DType>(param);
  })
  return op;
}

Operator *BilinearSamplerProp::CreateOperatorEx(Context ctx, mxnet::ShapeVector *in_shape,
                                     std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(BilinearSamplerParam);

MXNET_REGISTER_OP_PROPERTY(BilinearSampler, BilinearSamplerProp)
.add_argument("data", "NDArray-or-Symbol", "Input data to the BilinearsamplerOp.")
.add_argument("grid", "NDArray-or-Symbol", "Input grid to the BilinearsamplerOp."
                                "grid has two channels: x_src, y_src")
.add_arguments(BilinearSamplerParam::__FIELDS__())
.describe(R"code(Applies bilinear sampling to input feature map.

Bilinear Sampling is the key of  [NIPS2015] \"Spatial Transformer Networks\". The usage of the operator is very similar to remap function in OpenCV,
except that the operator has the backward pass.

Given :math:`data` and :math:`grid`, then the output is computed by

.. math::
  x_{src} = grid[batch, 0, y_{dst}, x_{dst}] \\
  y_{src} = grid[batch, 1, y_{dst}, x_{dst}] \\
  output[batch, channel, y_{dst}, x_{dst}] = G(data[batch, channel, y_{src}, x_{src})

:math:`x_{dst}`, :math:`y_{dst}` enumerate all spatial locations in :math:`output`, and :math:`G()` denotes the bilinear interpolation kernel.
The out-boundary points will be padded with zeros.The shape of the output will be (data.shape[0], data.shape[1], grid.shape[2], grid.shape[3]).

The operator assumes that :math:`data` has 'NCHW' layout and :math:`grid` has been normalized to [-1, 1].

BilinearSampler often cooperates with GridGenerator which generates sampling grids for BilinearSampler.
GridGenerator supports two kinds of transformation: ``affine`` and ``warp``.
If users want to design a CustomOp to manipulate :math:`grid`, please firstly refer to the code of GridGenerator.

Example 1::

  ## Zoom out data two times
  data = array([[[[1, 4, 3, 6],
                  [1, 8, 8, 9],
                  [0, 4, 1, 5],
                  [1, 0, 1, 3]]]])

  affine_matrix = array([[2, 0, 0],
                         [0, 2, 0]])

  affine_matrix = reshape(affine_matrix, shape=(1, 6))

  grid = GridGenerator(data=affine_matrix, transform_type='affine', target_shape=(4, 4))

  out = BilinearSampler(data, grid)

  out
  [[[[ 0,   0,     0,   0],
     [ 0,   3.5,   6.5, 0],
     [ 0,   1.25,  2.5, 0],
     [ 0,   0,     0,   0]]]


Example 2::

  ## shift data horizontally by -1 pixel

  data = array([[[[1, 4, 3, 6],
                  [1, 8, 8, 9],
                  [0, 4, 1, 5],
                  [1, 0, 1, 3]]]])

  warp_matrix = array([[[[1, 1, 1, 1],
                         [1, 1, 1, 1],
                         [1, 1, 1, 1],
                         [1, 1, 1, 1]],
                        [[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]]]])

  grid = GridGenerator(data=warp_matrix, transform_type='warp')
  out = BilinearSampler(data, grid)

  out
  [[[[ 4,  3,  6,  0],
     [ 8,  8,  9,  0],
     [ 4,  1,  5,  0],
     [ 0,  1,  3,  0]]]
)code" ADD_FILELINE);
}  // namespace op
}  // namespace mxnet
