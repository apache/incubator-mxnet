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

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_RNN_IMPL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_RNN_IMPL_H_
#if MXNET_USE_MKLDNN == 1
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/storage.h>
#include <algorithm>
#include <map>
#include <vector>
#include <utility>
#include <string>
#include "../../math_functions-inl.h"
#include "../../operator_common.h"
#include "../../rnn_impl.h"
#include "../../rnn-inl.h"
#include "mkldnn.hpp"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

struct MKLDNNRNNMemory {
  std::vector<mkldnn::memory> concat_weight_memory;
  std::vector<mkldnn::memory> concat_iter_memory;
  std::vector<mkldnn::memory> x_memory;
  std::vector<mkldnn::memory> hcx_memory;
  std::vector<mkldnn::memory> wx_memory;
  std::vector<mkldnn::memory> wh_memory;
  std::vector<mkldnn::memory> bias_memory;
  std::vector<mkldnn::memory> y_memory;
  std::vector<mkldnn::memory> hcy_memory;
  std::vector<mkldnn::memory> uni_states_memory;
  std::vector<mkldnn::memory> concat_states_memory;
  std::vector<mkldnn::memory> weight_layer_mems;
  std::vector<mkldnn::memory> weight_iter_mems;
  mkldnn::memory user_src_layer_memory_l;

  MKLDNNRNNMemory() : user_src_layer_memory_l(
      null_memory(CpuEngine::Get()->get_engine())) {}
};

static algorithm GetMKLDNNRNNAlgo(int mode,
                                  int* ngates,
                                  int* nstates) {
  algorithm algo = algorithm::vanilla_rnn;
  switch (mode) {
    case rnn_enum::kLstm:
      *ngates = 4;
      *nstates = 2;
      algo = algorithm::vanilla_lstm;
      break;
    case rnn_enum::kGru:
      *ngates = 3;
      *nstates = 1;
      algo = algorithm::gru_linear_before_reset;
      break;
    case rnn_enum::kRnnRelu:
    case rnn_enum::kRnnTanh:
      *ngates = 1;
      *nstates = 1;
      algo = algorithm::vanilla_rnn;
      break;
    default:
      LOG(FATAL) << "unsupported RNN mode:" << mode;
      break;
  }
  return algo;
}

static void ConcatData(mkldnn::memory::format src_format,
                       mkldnn::memory::format dst_format,
                       std::vector<mkldnn::memory::dims> srcs_cds,
                       mkldnn::memory::dims dst_cds,
                       mkldnn::memory::data_type mkldnn_dtype,
                       int concat_dimension,
                       const std::vector<void*> &srcs_data,
                       const mkldnn::memory &dst,
                       std::vector<mkldnn::memory> *tmp_src_mems) {
  auto cpu_engine = CpuEngine::Get()->get_engine();
  std::vector<mkldnn::memory::primitive_desc> srcs_pd;
  bool initialized = tmp_src_mems->size() > 0;
  for (size_t i = 0; i < srcs_cds.size(); i++) {
    auto desc = mkldnn::memory::desc(srcs_cds[i], mkldnn_dtype, src_format);
    auto mpd = mkldnn::memory::primitive_desc(desc, cpu_engine);
    srcs_pd.push_back(mpd);
    if (initialized) {
      tmp_src_mems->at(i).set_data_handle(srcs_data[i]);
    } else {
      auto src_memory = mkldnn::memory(mpd, srcs_data[i]);
      tmp_src_mems->push_back(src_memory);
    }
  }
  std::vector<primitive::at> inputs(tmp_src_mems->begin(), tmp_src_mems->end());
  auto dst_desc = mkldnn::memory::desc(dst_cds, mkldnn_dtype, dst_format);
  auto concat_pd = concat::primitive_desc(dst_desc, concat_dimension, srcs_pd);
  MKLDNNStream::Get()->RegisterPrim(concat(concat_pd, inputs, dst));
}

/**
 * Size of cached memory
 * 
 * Cache memory of wx, wh from the first layer and next L - 1 layers
 * seperately, as well as the layer and iter memory for src and dst.
 * Output states memory hx, hc and bias memory are also cached. It
 * will prepare memory on before and after reorder and concat. For
 * unidirectional, it will fused as dim like 1  + (L - 1) when I != H.
 * For bidirectional, it will fused as data + back_data (weight, bias,
 * iter etc)
 * 
 * @param num_layer Number of Layers
 * @param direction Direction of the RNN implement. It should be 1 or 2.
 * @param seq_len The maximum sequence length.
 * @param batch_size Batch size.
 * @param input_size Input channel. Also the dimension of the input feature.
 * @param hidden_size Hidden state size.
 * @return The required cache size.
 */
static size_t GetMKLDNNRNNCacheMemorySize(int num_layer,
                                          int direction,
                                          int seq_len,
                                          int batch_size,
                                          int input_size,
                                          int hidden_size,
                                          int mode) {
  int n_gates = 0, n_states = 0;
  GetMKLDNNRNNAlgo(mode, &n_gates, &n_states);
  int n_bias = mode == rnn_enum::kGru ? n_gates + 1 : n_gates;
  // sizes of single gates from a single cell
  const size_t weights_size_0 = direction * (input_size + hidden_size) * hidden_size;
  const size_t weights_size_n = direction * (direction * hidden_size + hidden_size) * hidden_size;
  const size_t bias_size      = direction * hidden_size;
  const size_t src_iter_size  = direction * batch_size * hidden_size;
  const size_t dst_iter_size  = direction * batch_size * hidden_size;
  const size_t dst_layer_size = seq_len * batch_size * direction * hidden_size;

  size_t size = (weights_size_0 + weights_size_n * (num_layer - 1)) * n_gates * 2 +
      bias_size * num_layer * n_bias + src_iter_size * num_layer * n_states * 2 +
      dst_iter_size * num_layer * n_states + dst_layer_size * 2;
  return size;
}

template <typename DType>
static void AdjustGruWeightGateOrder(DType* weight,
                                     const int I,
                                     const int H) {
  // mxnet gru gate order is reset, update and new gates
  // mkldnn gru gate order is update, reset and new gates
  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  DType* weight_reset = weight;
  DType* weight_update = weight + I * H;
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < I * H; i++) {
    DType tmp = weight_update[i];
    weight_update[i] = weight_reset[i];
    weight_reset[i] = tmp;
  }
}

template <typename DType>
static void AdjustGruBiasGateOrder(DType* bias,
                                   const int H) {
  // mxnet gru gate order is reset, update and new gates
  // mkldnn gru gate order is update, reset and new gates
  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  DType* bias_reset = bias;
  DType* bias_update = bias + H;
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < H; i++) {
    DType tmp = bias_update[i];
    bias_update[i] = bias_reset[i];
    bias_reset[i] = tmp;
  }
}
// since there is different sematics of MKLDNN's Fused RNN and MXNet FusedRNN,
// bidirectional will be fused layer by layer,
// unidirectional will be done by fused 1 + fused (L - 1) layers or fused L layers(when I = H)

template <typename DType>
static void MKLDNNRNNForwardSingleLayerBi(bool state_outputs,
                                          const int T,
                                          const int N,
                                          const int I,
                                          const int H,
                                          DType* x_ptr,
                                          DType* hx_ptr,
                                          DType* cx_ptr,
                                          DType* w_ptr,
                                          DType* b_ptr,
                                          DType* y_ptr,
                                          DType* hy_ptr,
                                          DType* cy_ptr,
                                          MKLDNNRNNMemory *mkldnn_mems,
                                          std::vector<primitive> *rnn_forward_prim,
                                          int layer_index,
                                          bool *has_cache,
                                          int dtype,
                                          bool is_train,
                                          int mode) {
  int ngates = 0, nstates = 0;
  algorithm nalgorithm = GetMKLDNNRNNAlgo(mode, &ngates, &nstates);
  const int nbias = mode == rnn_enum::kGru ? ngates + 1 : ngates;
  mkldnn::memory::data_type mkldnn_dtype = get_mkldnn_type(dtype);
  const int single_cell_size = N * H;
  const int mx_single_b_sz = ngates * H;
  DType* wx = w_ptr;  //  ngates * H, I
  DType* wh = w_ptr + I * H * ngates;  //  ngates * H, H
  DType* back_wx = w_ptr + ngates * H * (I + H);
  DType* back_wh = back_wx + I * H * ngates;
  DType* bx = b_ptr;
  DType* bh = b_ptr + H * ngates;
  DType* back_bx = b_ptr + mx_single_b_sz * 2;
  DType* back_bh = back_bx + H * ngates;
  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto null_memory_ = null_memory(cpu_engine);
  int offset1 = 0, offset2 = 0;
  bool initialized = *has_cache;
  mkldnn::memory::dims src_layer_tz = {T, N, I};
  mkldnn::memory::dims dst_layer_tz = {T, N, 2 * H};
  mkldnn::memory::dims weights_layer_tz = {1, 2, I, ngates, H};  //  ldigo
  mkldnn::memory::dims weights_layer_r_tz = {1, 1, I, ngates, H};  //  ldigo for reorder
  mkldnn::memory::dims weights_iter_tz = {1, 2, H, ngates, H};  //  ldigo
  mkldnn::memory::dims weights_iter_r_tz = {1, 1, H, ngates, H};  //  ldigo for reorder
  mkldnn::memory::dims bias_tz = {1, 2, nbias, H};  // ldgo
  mkldnn::memory::dims src_iter_tz = {1, 2, nstates, N, H};  //  ldsnc
  mkldnn::memory::dims dst_iter_tz = {1, 2, nstates, N, H};  //  ldsnc

  bool has_adjusted = false;
  if (!initialized || is_train) {
    if (mode == rnn_enum::kGru) {
      AdjustGruWeightGateOrder(wx, I, H);
      AdjustGruWeightGateOrder(back_wx, I, H);
      AdjustGruWeightGateOrder(wh, H, H);
      AdjustGruWeightGateOrder(back_wh, H, H);
      has_adjusted = true;
    }
    auto src_wx = mkldnn_mems->concat_weight_memory[2 * layer_index];
    auto src_wh = mkldnn_mems->concat_weight_memory[2 * layer_index + 1];
    std::vector<void*> srcs_data1;
    srcs_data1.push_back(wx);
    srcs_data1.push_back(back_wx);
    ConcatData(mkldnn::memory::format::ldgoi, mkldnn::memory::format::ldgoi,
        {weights_layer_r_tz, weights_layer_r_tz}, weights_layer_tz,
        mkldnn_dtype, 1, srcs_data1, src_wx, &(mkldnn_mems->weight_layer_mems));
    srcs_data1.clear();
    srcs_data1.push_back(wh);
    srcs_data1.push_back(back_wh);
    ConcatData(mkldnn::memory::format::ldgoi, mkldnn::memory::format::ldgoi,
        {weights_iter_r_tz, weights_iter_r_tz}, weights_iter_tz,
        mkldnn_dtype, 1, srcs_data1, src_wh, &(mkldnn_mems->weight_iter_mems));

    MKLDNNStream::Get()->RegisterPrim(reorder(src_wx, mkldnn_mems->wx_memory[layer_index]));
    MKLDNNStream::Get()->RegisterPrim(reorder(src_wh, mkldnn_mems->wh_memory[layer_index]));

    DType* user_bias = reinterpret_cast<DType *>
        (mkldnn_mems->bias_memory[layer_index].get_data_handle());
    if (mode == rnn_enum::kGru) {
      // While mxnet gru gate order is reset, update and new gates,
      // mkldnn gru gate order is update, reset and new gates. So
      // we need to swap the order of reset and update from mxnet.
      const index_t single_b_sz = nbias * H;
      #pragma omp parallel for num_threads(omp_threads)
      for (int j = 0; j < H; j++) {
        user_bias[j + H] = bx[j] + bh[j];
        user_bias[single_b_sz + j + H] = back_bx[j] + back_bh[j];
        user_bias[j] = bx[j + H] + bh[j + H];
        user_bias[single_b_sz + j] = back_bx[j + H] + back_bh[j + H];
      }
      #pragma omp parallel for num_threads(omp_threads)
      for (int j = 2 * H; j < 3 * H; j++) {
        user_bias[j] = bx[j];
        user_bias[j + H] = bh[j];
        user_bias[single_b_sz + j] = back_bx[j];
        user_bias[single_b_sz + j + H] = back_bh[j];
      }
    } else {
      #pragma omp parallel for num_threads(omp_threads)
      for (int j = 0; j < mx_single_b_sz; j++) {
        user_bias[j] = bx[j] + bh[j];
        user_bias[mx_single_b_sz + j] = back_bx[j] + back_bh[j];
      }
    }
  }

  auto src_layer_md = mkldnn::memory::desc(
      { src_layer_tz }, mkldnn_dtype, mkldnn::memory::format::tnc);
  auto weight_layer_md = mkldnn::memory::desc(
      { weights_layer_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
  auto weight_iter_md = mkldnn::memory::desc(
      { weights_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
  auto dst_layer_md = mkldnn::memory::desc(
      { dst_layer_tz }, mkldnn_dtype, mkldnn::memory::format::tnc);
  auto dst_iter_md = mkldnn::memory::desc(
      { dst_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldsnc);
  auto src_iter_md = mkldnn::memory::desc(
      {src_iter_tz}, mkldnn_dtype, mkldnn::memory::format::ldsnc);
  auto bias_md = mkldnn::memory::desc({bias_tz},
      mkldnn_dtype, mkldnn::memory::format::ldgo);

  auto user_src_iter_memory = mkldnn_mems->concat_iter_memory[2];
  if (mode == rnn_enum::kLstm) {
    std::vector<void*> srcs_data1;
    srcs_data1.push_back(hx_ptr);
    srcs_data1.push_back(cx_ptr);
    auto tmp1_src_iter_memory = mkldnn_mems->concat_iter_memory[0];
    ConcatData(mkldnn::memory::format::ldsnc, mkldnn::memory::format::ldsnc,
        {{1, 1, 1, N, H}, {1, 1, 1, N, H}}, {1, 1, nstates, N, H}, mkldnn_dtype, 2,
        srcs_data1, tmp1_src_iter_memory, &(mkldnn_mems->uni_states_memory));
    std::vector<void*> srcs_data2;
    srcs_data2.push_back(hx_ptr + single_cell_size);
    srcs_data2.push_back(cx_ptr + single_cell_size);
    auto tmp2_src_iter_memory = mkldnn_mems->concat_iter_memory[1];
    ConcatData(mkldnn::memory::format::ldsnc, mkldnn::memory::format::ldsnc,
        {{1, 1, 1, N, H}, {1, 1, 1, N, H}}, {1, 1, nstates, N, H}, mkldnn_dtype, 2,
        srcs_data2, tmp2_src_iter_memory, &(mkldnn_mems->uni_states_memory));
    std::vector<void*> srcs_data3;
    srcs_data3.push_back(reinterpret_cast<DType *>(tmp1_src_iter_memory.get_data_handle()));
    srcs_data3.push_back(reinterpret_cast<DType *>(tmp2_src_iter_memory.get_data_handle()));
    ConcatData(mkldnn::memory::format::ldsnc, mkldnn::memory::format::ldsnc,
        {{1, 1, nstates, N, H}, {1, 1, nstates, N, H}}, {1, 2, nstates, N, H},
        mkldnn_dtype, 1, srcs_data3, user_src_iter_memory, &(mkldnn_mems->concat_states_memory));
  } else {
    user_src_iter_memory.set_data_handle(hx_ptr);
  }
  mkldnn_mems->hcx_memory[layer_index].set_data_handle(user_src_iter_memory.get_data_handle());

  rnn_cell::desc rnn_cell(nalgorithm,
      mode == rnn_enum::kRnnRelu ? algorithm::eltwise_relu : algorithm::eltwise_tanh);

  rnn_forward::desc layer_desc(prop_kind::forward_inference, rnn_cell,
      rnn_direction::bidirectional_concat, src_layer_md,
      src_iter_md, weight_layer_md, weight_iter_md,
      bias_md, dst_layer_md, dst_iter_md);

  auto prim_desc
       = rnn_forward::primitive_desc(layer_desc, cpu_engine);

  if (x_ptr && layer_index == 0) {
    mkldnn_mems->x_memory[layer_index].set_data_handle(x_ptr);
  } else {
    mkldnn_mems->x_memory[layer_index].set_data_handle(
        mkldnn_mems->user_src_layer_memory_l.get_data_handle());
  }
  mkldnn_mems->y_memory[layer_index].set_data_handle(y_ptr);
  if (rnn_forward_prim->size() <= (size_t)layer_index) {
    primitive rnn_prim = rnn_forward(prim_desc, mkldnn_mems->x_memory[layer_index],
          mkldnn_mems->hcx_memory[layer_index], mkldnn_mems->wx_memory[layer_index],
          mkldnn_mems->wh_memory[layer_index], mkldnn_mems->bias_memory[layer_index],
          mkldnn_mems->y_memory[layer_index],
          mkldnn_mems->hcy_memory[layer_index], null_memory_);
    rnn_forward_prim->push_back(rnn_prim);
  }
  MKLDNNStream::Get()->RegisterPrim((*rnn_forward_prim)[layer_index]);
  MKLDNNStream::Get()->Submit();

  if (state_outputs) {
    DType* dst_hcy = reinterpret_cast<DType *>(mkldnn_mems->hcy_memory[layer_index].get_data_handle());
    if (mode == rnn_enum::kLstm) {
      offset1 = nstates * single_cell_size;
      offset2 = (nstates + 1) * single_cell_size;
      #pragma omp parallel for num_threads(omp_threads)
      for (int n = 0; n < single_cell_size; n++) {
        hy_ptr[n] = dst_hcy[n];
        hy_ptr[n + single_cell_size] = dst_hcy[n + offset1];
        cy_ptr[n] = dst_hcy[n + single_cell_size];
        cy_ptr[n + single_cell_size] = dst_hcy[n + offset2];
      }
    } else {
      #pragma omp parallel for num_threads(omp_threads)
      for (int n = 0; n < 2 * single_cell_size; n++) {
        hy_ptr[n] = dst_hcy[n];
      }
    }
  }
  if (has_adjusted) {
    AdjustGruWeightGateOrder(wx, I, H);
    AdjustGruWeightGateOrder(back_wx, I, H);
    AdjustGruWeightGateOrder(wh, H, H);
    AdjustGruWeightGateOrder(back_wh, H, H);
  }
}


template <typename DType>
static void MKLDNNRNNForwardUnidi(bool state_outputs,
                                  const int L,
                                  const int T,
                                  const int N,
                                  const int I,
                                  const int H,
                                  DType* x_ptr,
                                  DType* hx_ptr,
                                  DType* cx_ptr,
                                  DType* w_ptr,
                                  DType* b_ptr,
                                  DType* y_ptr,
                                  DType* hy_ptr,
                                  DType* cy_ptr,
                                  MKLDNNRNNMemory *mkldnn_mems,
                                  std::vector<primitive> *rnn_forward_prim,
                                  int layer_index,
                                  bool *has_cache,
                                  int dtype,
                                  bool is_train,
                                  int mode) {
  int ngates = 0, nstates = 0;
  algorithm nalgorithm = GetMKLDNNRNNAlgo(mode, &ngates, &nstates);
  const int nbias = (mode == rnn_enum::kGru ? ngates + 1 : ngates);
  mkldnn::memory::data_type mkldnn_dtype = get_mkldnn_type(dtype);
  const int cell_size = N * H;
  const int single_cell_size = N * H;
  const int single_b_size = nbias * H;
  int w_size = (I + H) * H * ngates;
  const int omp_threads = mxnet::engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto null_memory_ = null_memory(cpu_engine);
  int offset1 = 0, offset2 = 0;
  bool initialized = *has_cache;

  mkldnn::memory::dims src_layer_tz = {T, N, I};
  mkldnn::memory::dims dst_layer_tz = {T, N, H};
  mkldnn::memory::dims weights_layer_tz = {L, 1, I, ngates, H};  //  ldigo
  mkldnn::memory::dims weights_iter_tz = {L, 1, H, ngates, H};  //  ldigo
  mkldnn::memory::dims bias_tz = {L, 1, nbias, H};  // ldgo
  mkldnn::memory::dims src_iter_tz = {L, 1, nstates, N, H};  //  ldsnc
  mkldnn::memory::dims dst_iter_tz = {L, 1, nstates, N, H};  //  ldsnc
  mkldnn::memory::dims weights_layer_r_tz = {1, 1, I, ngates, H};  //  ldigo for reorder
  mkldnn::memory::dims weights_iter_r_tz = {1, 1, H, ngates, H};  //  ldigo for reorder

  auto weight_layer_md = mkldnn::memory::desc(
      { weights_layer_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
  auto weight_iter_md = mkldnn::memory::desc(
      { weights_iter_tz }, mkldnn_dtype, mkldnn::memory::format::ldigo);
  auto src_layer_md = mkldnn::memory::desc(
      { src_layer_tz }, mkldnn_dtype, mkldnn::memory::format::tnc);
  auto dst_layer_md = mkldnn::memory::desc(
      {dst_layer_tz}, mkldnn_dtype, mkldnn::memory::format::tnc);
  auto src_iter_md = mkldnn::memory::desc(
      {src_iter_tz}, mkldnn_dtype, mkldnn::memory::format::ldsnc);
  auto bias_md = mkldnn::memory::desc({bias_tz},
      mkldnn_dtype, mkldnn::memory::format::ldgo);
  auto dst_iter_md = mkldnn::memory::desc(
      {dst_iter_tz}, mkldnn_dtype, mkldnn::memory::format::ldsnc);

  for (int l = 0; l < L; l++) {
    if (mode == rnn_enum::kLstm) {
      std::vector<void*> srcs_data;
      srcs_data.push_back(hx_ptr);
      srcs_data.push_back(cx_ptr);
      auto tmp_src_iter_memory = mkldnn_mems->concat_iter_memory[l + layer_index];
      ConcatData(mkldnn::memory::format::ldsnc, mkldnn::memory::format::ldsnc,
          {{1, 1, 1, N, H}, {1, 1, 1, N, H}}, {1, 1, nstates, N, H}, mkldnn_dtype,
          2, srcs_data, tmp_src_iter_memory, &(mkldnn_mems->uni_states_memory));
    } else {
      mkldnn_mems->concat_iter_memory[l + layer_index].set_data_handle(hx_ptr);
    }
    hx_ptr += cell_size;
    if (mode == rnn_enum::kLstm) {
      cx_ptr += cell_size;
    }
  }

  auto user_src_iter_memory = null_memory_;
  if (L == 1) {
    user_src_iter_memory = mkldnn_mems->concat_iter_memory[layer_index];
  } else {
    user_src_iter_memory = mkldnn_mems->concat_iter_memory[L + layer_index];
    std::vector<void*> src_l_data;
    std::vector<mkldnn::memory::dims> src_l_dim;
    for (int l = 0; l < L; l++) {
      src_l_data.push_back(reinterpret_cast<DType *>
          (mkldnn_mems->concat_iter_memory[l + layer_index].get_data_handle()));
      src_l_dim.push_back({1, 1, nstates, N, H});
    }
    ConcatData(mkldnn::memory::format::ldsnc, mkldnn::memory::format::ldsnc, src_l_dim,
        {L, 1, nstates, N, H}, mkldnn_dtype, 0, src_l_data, user_src_iter_memory,
        &(mkldnn_mems->concat_states_memory));
  }
  mkldnn_mems->hcx_memory[layer_index].set_data_handle(user_src_iter_memory.get_data_handle());

  auto src_wx_f = mkldnn_mems->concat_weight_memory[2 * layer_index];
  auto src_wh_f = mkldnn_mems->concat_weight_memory[2 * layer_index + 1];

  std::vector<void*> srcs_data_x;
  std::vector<void*> srcs_data_h;
  std::vector<mkldnn::memory::dims> src_l_dim_x;
  std::vector<mkldnn::memory::dims> src_l_dim_h;

  bool has_adjusted = false;
  if (!initialized) {
    if (L == 1) {
      DType* wx = w_ptr;
      DType* wh = wx + I * H * ngates;
      if (mode == rnn_enum::kGru) {
        AdjustGruWeightGateOrder(wx, I, H);
        AdjustGruWeightGateOrder(wh, H, H);
        has_adjusted = true;
      }
      src_wx_f.set_data_handle(wx);
      src_wh_f.set_data_handle(wh);
    } else {
      for (int l = 0; l < L; l++) {
        DType* wx = w_ptr + l * w_size;
        DType* wh = wx + I * H * ngates;
        if (mode == rnn_enum::kGru) {
          AdjustGruWeightGateOrder(wx, I, H);
          AdjustGruWeightGateOrder(wh, H, H);
          has_adjusted = true;
        }
        srcs_data_x.push_back(wx);
        srcs_data_h.push_back(wh);
        src_l_dim_x.push_back(weights_layer_r_tz);
        src_l_dim_h.push_back(weights_iter_r_tz);
      }
      ConcatData(mkldnn::memory::format::ldgoi, mkldnn::memory::format::ldgoi,
          src_l_dim_x, weights_layer_tz, mkldnn_dtype, 0, srcs_data_x, src_wx_f,
          &(mkldnn_mems->weight_layer_mems));
      ConcatData(mkldnn::memory::format::ldgoi, mkldnn::memory::format::ldgoi,
          src_l_dim_h, weights_iter_tz, mkldnn_dtype, 0, srcs_data_h, src_wh_f,
          &(mkldnn_mems->weight_iter_mems));
    }
    MKLDNNStream::Get()->RegisterPrim(reorder(src_wx_f, mkldnn_mems->wx_memory[layer_index]));
    MKLDNNStream::Get()->RegisterPrim(reorder(src_wh_f, mkldnn_mems->wh_memory[layer_index]));

    DType* user_bias_f = reinterpret_cast<DType *>(mkldnn_mems->bias_memory[layer_index].get_data_handle());
    if (mode == rnn_enum::kGru) {
      const int mx_single_b_sz = ngates * H;
      for (int l = 0; l < L; l++) {
        #pragma omp parallel for num_threads(omp_threads)
        for (int g = 0; g < H; g++) {
          // While mxnet gru gate order is reset, update and new gates,
          // mkldnn gru gate order is update, reset and new gates. So
          // we need to swap the order of reset and update from mxnet.
          user_bias_f[g + H + l * single_b_size] =
              b_ptr[g + l * mx_single_b_sz * 2]
              + b_ptr[g + l * mx_single_b_sz * 2 + mx_single_b_sz];
          user_bias_f[g + l * single_b_size] =
              b_ptr[g + H + l * mx_single_b_sz * 2]
              + b_ptr[g + H + l * mx_single_b_sz * 2 + mx_single_b_sz];
        }
        #pragma omp parallel for num_threads(omp_threads)
        for (int g = 2 * H; g < 3 * H; g++) {
          user_bias_f[g + l * single_b_size] = b_ptr[g + l * mx_single_b_sz * 2];
          user_bias_f[g + l * single_b_size + H] =
              b_ptr[g + l * mx_single_b_sz * 2 + mx_single_b_sz];
        }
      }
    } else {
      #pragma omp parallel for num_threads(omp_threads)
      for (int j = 0; j < L * single_b_size; j++) {
        int k = j / single_b_size;
        user_bias_f[j] = b_ptr[j + k * single_b_size] + b_ptr[j + k * single_b_size + single_b_size];
      }
    }
  }

  rnn_cell::desc rnn_cell(nalgorithm,
      mode == rnn_enum::kRnnRelu ? algorithm::eltwise_relu : algorithm::eltwise_tanh);

  rnn_forward::desc layer_desc(prop_kind::forward_inference, rnn_cell,
      rnn_direction::unidirectional, src_layer_md,
      src_iter_md, weight_layer_md, weight_iter_md,
      bias_md, dst_layer_md, dst_iter_md);

  auto prim_desc
       = rnn_forward::primitive_desc(layer_desc, cpu_engine);

  if (x_ptr && layer_index == 0) {
    mkldnn_mems->x_memory[layer_index].set_data_handle(x_ptr);
  } else {
    mkldnn_mems->x_memory[layer_index].set_data_handle(
        mkldnn_mems->user_src_layer_memory_l.get_data_handle());
  }
  mkldnn_mems->y_memory[layer_index].set_data_handle(y_ptr);
  if (rnn_forward_prim->size() <= (size_t)layer_index) {
    primitive rnn_prim = rnn_forward(prim_desc, mkldnn_mems->x_memory[layer_index],
          mkldnn_mems->hcx_memory[layer_index], mkldnn_mems->wx_memory[layer_index],
          mkldnn_mems->wh_memory[layer_index], mkldnn_mems->bias_memory[layer_index],
          mkldnn_mems->y_memory[layer_index],
         mkldnn_mems->hcy_memory[layer_index], null_memory_);
    rnn_forward_prim->push_back(rnn_prim);
  }
  MKLDNNStream::Get()->RegisterPrim((*rnn_forward_prim)[layer_index]);
  MKLDNNStream::Get()->Submit();

  if (state_outputs) {
    DType* dst_hcy = reinterpret_cast<DType *>(mkldnn_mems->hcy_memory[layer_index].get_data_handle());
    if (mode == rnn_enum::kLstm) {
      for (int l = 0; l < L; l++) {
        offset1 = l * single_cell_size;
        offset2 = l * nstates * single_cell_size;
        #pragma omp parallel for num_threads(omp_threads)
        for (int n = 0; n < single_cell_size; n++) {
          hy_ptr[offset1 + n] = dst_hcy[offset2 + n];
          cy_ptr[offset1 + n] = dst_hcy[offset2 + n + single_cell_size];
        }
      }
    } else {
      #pragma omp parallel for num_threads(omp_threads)
      for (int n = 0; n < L * single_cell_size; n++) {
        hy_ptr[n] = dst_hcy[n];
      }
    }
  }
  if (has_adjusted) {
    for (int l = 0; l < L; l++) {
      DType* wx = w_ptr + l * w_size;
      DType* wh = wx + I * H * ngates;
      AdjustGruWeightGateOrder(wx, I, H);
      AdjustGruWeightGateOrder(wh, H, H);
    }
  }
}

template <typename DType>
static void MKLDNNRNNForward(bool state_outputs,
                             const int L,
                             const int D,
                             const int T,
                             const int N,
                             const int I,
                             const int H,
                             DType* x_ptr,
                             DType* hx_ptr,
                             DType* cx_ptr,
                             DType* w_ptr,
                             DType* b_ptr,
                             DType* y_ptr,
                             DType* hy_ptr,
                             DType* cy_ptr,
                             MKLDNNRNNMemory *mkldnn_mems,
                             std::vector<primitive> *rnn_forward_prim,
                             bool *has_cache,
                             int dtype,
                             bool is_train,
                             int mode) {
  int ngates = 0, nstates = 0;
  GetMKLDNNRNNAlgo(mode, &ngates, &nstates);
  const int b_size = 2 * H * ngates * D;
  const int cell_size = N * H * D;
  //  First layer
  int w_size = (I + H) * H * ngates * D;
  DType* tmpNull = NULL;
  // when D = 1 and I == H, L layers can be fused together
  if (D == 1 && I == H) {
    MKLDNNRNNForwardUnidi(state_outputs, L, T, N, I, H, x_ptr,
        hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr,
        mkldnn_mems, rnn_forward_prim,
        0, has_cache, dtype, is_train, mode);
  } else {
    if (D == 2) {
      MKLDNNRNNForwardSingleLayerBi(state_outputs, T, N, I, H, x_ptr,
          hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr,
          mkldnn_mems, rnn_forward_prim,
          0, has_cache, dtype, is_train, mode);
    } else {
      MKLDNNRNNForwardUnidi(state_outputs, 1, T, N, I, H, x_ptr,
          hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr,
          mkldnn_mems, rnn_forward_prim,
          0, has_cache, dtype, is_train, mode);
    }
    if (L > 1) {
      mkldnn_mems->user_src_layer_memory_l = mkldnn_mems->y_memory[0];
      //  go to next L - 1 layers.
      //  If D = 2, do it layer by layer. If D = 1, fused L - 1 layers
      w_ptr += w_size;
      b_ptr += b_size;
      if (D == 2) {
        w_size = (H * D + H) * H * ngates * D;
        for (int l = 0; l < L - 1; l++) {
          if (state_outputs) {
            hy_ptr += cell_size;
            if (mode == rnn_enum::kLstm) {
              cy_ptr += cell_size;
            }
          }
          hx_ptr += cell_size;
          if (mode == rnn_enum::kLstm) {
            cx_ptr += cell_size;
          }
          MKLDNNRNNForwardSingleLayerBi(state_outputs, T, N, D * H, H, tmpNull,
              hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr,
              cy_ptr, mkldnn_mems, rnn_forward_prim,
              1, has_cache, dtype, is_train, mode);
          mkldnn_mems->user_src_layer_memory_l = mkldnn_mems->y_memory[1];
          w_ptr += w_size;
          b_ptr += b_size;
        }
      }
      if (D == 1) {
        if (state_outputs) {
          hy_ptr += cell_size;
          if (mode == rnn_enum::kLstm) {
            cy_ptr += cell_size;
          }
        }
        w_size = (H + H) * H * ngates;
        MKLDNNRNNForwardUnidi(state_outputs, L - 1, T, N, H, H, tmpNull,
            hx_ptr, cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr, mkldnn_mems,
            rnn_forward_prim, 1, has_cache, dtype, is_train, mode);
      }
    }
  }
  *has_cache = true;
}

template <typename DType>
static void MKLDNNRNNForwardInference(bool state_outputs,
                                      const int num_layers,
                                      const int direction,
                                      const int seq_length,
                                      const int batch_size,
                                      const int input_size,
                                      const int state_size,
                                      DType* x_ptr,
                                      DType* hx_ptr,
                                      DType* cx_ptr,
                                      DType* w_ptr,
                                      DType* b_ptr,
                                      DType* y_ptr,
                                      DType* hy_ptr,
                                      DType* cy_ptr,
                                      MKLDNNRNNMemory *mkldnn_mems,
                                      std::vector<primitive> *rnn_forward_prim,
                                      bool *has_cache,
                                      int dtype,
                                      bool is_train,
                                      int mode) {
  switch (mode) {
    case rnn_enum::kLstm:
    case rnn_enum::kGru:
    case rnn_enum::kRnnTanh:
    case rnn_enum::kRnnRelu:
      MKLDNNRNNForward<DType>(state_outputs, num_layers, direction, seq_length,
                              batch_size, input_size, state_size, x_ptr, hx_ptr,
                              cx_ptr, w_ptr, b_ptr, y_ptr, hy_ptr, cy_ptr,
                              mkldnn_mems, rnn_forward_prim,
                              has_cache, dtype, is_train, mode);
      break;
    default:
      LOG(FATAL) << "unknown RNN mode" << mode;
      break;
  }
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_RNN_IMPL_H_
