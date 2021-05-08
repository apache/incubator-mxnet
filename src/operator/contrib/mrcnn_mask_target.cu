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
 *  Copyright (c) 2019 by Contributors
 * \file mrcnn_mask_target.cu
 * \brief Mask-RCNN target generator
 * \author Serge Panev
 */

#include "./mrcnn_mask_target-inl.h"

namespace mxnet {
namespace op {

using namespace mshadow::cuda;

// The maximum number of blocks to use in the default kernel call.
constexpr int MAXIMUM_NUM_BLOCKS = 4096;

inline int CUDA_GET_BLOCKS(const int N) {
  return std::max(
      std::min(
          (N + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock,
          MAXIMUM_NUM_BLOCKS),
      // Use at least 1 block, since CUDA does not allow empty block.
      1);
}

// Kernels

template <typename T>
__device__ T bilinear_interpolate(
    const T* in_data,
    const int height,
    const int width,
    T y,
    T x,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  int y_low = static_cast<int>(y);
  int x_low = static_cast<int>(x);
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = in_data[y_low * width + x_low];
  T v2 = in_data[y_low * width + x_high];
  T v3 = in_data[y_high * width + x_low];
  T v4 = in_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

// Modified version of RoIAlignForwardKernel from Caffe (in roi_align.cu)
// Main modifications:
// - We don't need position_sensitive neither spatial_scale from the original RoIAlign kernel.
// - We replace `channels` by `num_classes` and modify the logic consequently (e.g. offset_in_data
//   does not use `c` anymore).
template <typename T>
__device__ void RoIAlignForward(
    const T* in_data,  // (B, M, H, W)
    const T* rois,  // (B, N, 4)
    const T* matches,  // (B, N)
    const int num_el,
    const int num_classes,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const int num_rois,
    const int num_gtmasks,
    const bool continuous_coordinate,
    T* out_data) {  // (B, N, C, H, W)
  // Update kernel
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < num_el;
       index += blockDim.x * gridDim.x) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    // int c = (index / pooled_width / pooled_height) % num_classes;
    int n = (index / pooled_width / pooled_height / num_classes) % num_rois;
    int batch_idx = (index / pooled_width / pooled_height / num_classes / num_rois);

    int roi_batch_ind = matches[batch_idx * num_rois + n];

    const T* offset_rois = rois + batch_idx * (4 * num_rois) + n * 4;
    // Do not using rounding; this implementation detail is critical
    T roi_offset = continuous_coordinate ? static_cast<T>(0.5) : static_cast<T>(0);
    T roi_start_w = offset_rois[0] - roi_offset;
    T roi_start_h = offset_rois[1] - roi_offset;
    T roi_end_w = offset_rois[2] - roi_offset;
    T roi_end_h = offset_rois[3] - roi_offset;

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;

    // Force malformed ROIs to be 1x1
    if (!continuous_coordinate) {  // backward compatiblity
      // Force malformed ROIs to be 1x1
      roi_width = max(roi_width, (T)1.);
      roi_height = max(roi_height, (T)1.);
    }
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_in_data =
        in_data + batch_idx * num_gtmasks * height * width
        + roi_batch_ind * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate(
            offset_in_data, height, width, y, x, index);
        output_val += val;
      }
    }
    output_val /= count;

    out_data[index] = output_val;
  }
}


template<typename DType>
__global__ void MRCNNMaskTargetKernel(const DType *rois,
                                      const DType *gt_masks,
                                      const DType *matches,
                                      const DType *cls_targets,
                                      DType* sampled_masks,
                                      DType* mask_cls,
                                      const int total_out_el,
                                      const bool aligned,
                                      int batch_size,
                                      int num_classes,
                                      int num_rois,
                                      int num_gtmasks,
                                      int gt_height,
                                      int gt_width,
                                      int mask_size_h,
                                      int mask_size_w,
                                      int sample_ratio) {
  // computing sampled_masks
  RoIAlignForward(gt_masks, rois, matches, total_out_el,
                  num_classes, gt_height, gt_width, mask_size_h, mask_size_w,
                  sample_ratio, num_rois, num_gtmasks, aligned, sampled_masks);
  // computing mask_cls
  int num_masks = batch_size * num_rois * num_classes;
  int mask_vol = mask_size_h * mask_size_w;
  for (int mask_idx = blockIdx.x; mask_idx < num_masks; mask_idx += gridDim.x) {
    int cls_idx = mask_idx % num_classes;
    int roi_idx = (mask_idx / num_classes) % num_rois;
    int batch_idx = (mask_idx / num_classes / num_rois);

    DType* mask_cls_out = mask_cls + mask_idx * mask_vol;

    DType cls_target = cls_targets[batch_idx * num_rois + roi_idx];
    DType out_val = (cls_target == cls_idx);
    for (int mask_pixel = threadIdx.x; mask_pixel < mask_vol; mask_pixel += blockDim.x) {
      mask_cls_out[mask_pixel] = out_val;
    }
  }
}

template<>
void MRCNNMaskTargetRun<gpu>(const MRCNNMaskTargetParam& param, const std::vector<TBlob> &inputs,
                             const std::vector<TBlob> &outputs, mshadow::Stream<gpu> *s) {
  const int block_dim_size = kMaxThreadsPerBlock;
  using namespace mxnet_op;
  using mshadow::Tensor;

  auto stream = mshadow::Stream<gpu>::GetStream(s);

  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    auto rois = inputs[mrcnn_index::kRoi].FlatToKD<gpu, 3, DType>(s);
    auto gt_masks = inputs[mrcnn_index::kGtMask].FlatToKD<gpu, 4, DType>(s);
    auto matches = inputs[mrcnn_index::kMatches].FlatTo2D<gpu, DType>(s);
    auto cls_targets = inputs[mrcnn_index::kClasses].FlatTo2D<gpu, DType>(s);

    auto out_masks = outputs[mrcnn_index::kMask].FlatToKD<gpu, 5, DType>(s);
    auto out_mask_cls = outputs[mrcnn_index::kMaskClasses].FlatToKD<gpu, 5, DType>(s);

    int batch_size = gt_masks.shape_[0];
    int num_gtmasks = gt_masks.shape_[1];
    int gt_height = gt_masks.shape_[2];
    int gt_width = gt_masks.shape_[3];

    int num_el = outputs[mrcnn_index::kMask].Size();

    dim3 dimGrid = dim3(CUDA_GET_BLOCKS(num_el));
    dim3 dimBlock = dim3(block_dim_size);

    MRCNNMaskTargetKernel<<<dimGrid, dimBlock, 0, stream>>>
    (rois.dptr_, gt_masks.dptr_, matches.dptr_, cls_targets.dptr_,
    out_masks.dptr_, out_mask_cls.dptr_, num_el, param.aligned,
    batch_size, param.num_classes, param.num_rois,
    num_gtmasks, gt_height, gt_width,
    param.mask_size[0], param.mask_size[1], param.sample_ratio);
    MSHADOW_CUDA_POST_KERNEL_CHECK(MRCNNMaskTargetKernel);
  });
}

DMLC_REGISTER_PARAMETER(MRCNNMaskTargetParam);

NNVM_REGISTER_OP(_contrib_mrcnn_mask_target)
.describe("Generate mask targets for Mask-RCNN.")
.set_num_inputs(4)
.set_num_outputs(2)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"rois", "gt_masks", "matches", "cls_targets"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"masks_output", "mask_classes_output"};
})
.set_attr_parser(ParamParser<MRCNNMaskTargetParam>)
.set_attr<mxnet::FInferShape>("FInferShape", MRCNNMaskTargetShape)
.set_attr<nnvm::FInferType>("FInferType", MRCNNMaskTargetType)
.set_attr<FCompute>("FCompute<gpu>", MRCNNMaskTargetCompute<gpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("rois", "NDArray-or-Symbol", "Bounding box coordinates, a 3D array")
.add_argument("gt_masks", "NDArray-or-Symbol", "Input masks of full image size, a 4D array")
.add_argument("matches", "NDArray-or-Symbol", "Index to a gt_mask, a 2D array")
.add_argument("cls_targets", "NDArray-or-Symbol",
              "Value [0, num_class), excluding background class, a 2D array")
.add_arguments(MRCNNMaskTargetParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
