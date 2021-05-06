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
 *  \file mkldnn_test.cc
 *  \brief test functions in mkldnn.
 *  \author Da Zheng
 */

#if MXNET_USE_MKLDNN == 1

#include <mkldnn_types.h>
#include <cmath>
#include <climits>
#include <set>
#include "gtest/gtest.h"
#include "mxnet/imperative.h"
#include "../../src/operator/nn/mkldnn/mkldnn_ops-inl.h"
#include "../../src/operator/nn/mkldnn/mkldnn_base-inl.h"
#include "../include/test_mkldnn.h"

using namespace mxnet;

#if __GNUC__ >= 5
bool test_mem_align(void *mem, size_t size, size_t alignment, size_t space) {
  void *ret1, *ret2;
  size_t space1, space2;
  space1 = space;
  space2 = space;
  ret1 = mxnet::AlignMem(mem, size, alignment, &space1);
  ret2 = std::align(alignment, size, mem, space2);
  EXPECT_EQ(ret1, ret2);
  EXPECT_EQ(space1, space2);
  return ret1 == ret2;
}
#endif

TEST(MKLDNN_UTIL_FUNC, AlignMem) {
#if __GNUC__ >= 5
  size_t alignment = 4096;
  void *mem;
  size_t size, space;
  // When mem has been aligned.
  mem = reinterpret_cast<void *>(0x10000);
  size = 1000;
  space = 10000;
  test_mem_align(mem, size, alignment, space);

  // When mem isn't aligned and we have enough space for alignment.
  mem = reinterpret_cast<void *>(0x10010);
  size = 1000;
  space = 10000;
  test_mem_align(mem, size, alignment, space);

  // When mem isn't aligned and we don't have enough memory for alignment
  mem = reinterpret_cast<void *>(0x10010);
  size = 1000;
  space = 1001;
  test_mem_align(mem, size, alignment, space);

  for (size_t i = 0; i < 10000; i++) {
    mem = reinterpret_cast<void *>(random());
    size = random() % 2000;
    space = random() % 2000;
    test_mem_align(mem, size, alignment, space);
  }
#else
  // std::align is not supported in GCC < 5.0, this test case will be checked
  // with newer version
  LOG(INFO) << "Skipped for GCC " << __GNUC__ << "." << __GNUC_MINOR__;
#endif
}

static void VerifyDefMem(const mkldnn::memory &mem) {
  mkldnn::memory::desc desc = mem.get_desc();
  mshadow::default_real_t *data
      = static_cast<mshadow::default_real_t *>(mem.get_data_handle());
  size_t size = desc.get_size() / sizeof(mshadow::default_real_t);
  size_t num_same = 0;
  for (int i = 0; i < size; i++)
    num_same += data[i] == static_cast<mshadow::default_real_t>(i % 100 - 50);
  EXPECT_EQ(num_same, size);
}

TEST(MKLDNN_UTIL_FUNC, MemFormat) {
  // Check whether the number of format is correct.
  CHECK_EQ(mkldnn_format_tag_last, 363);
  CHECK_EQ(mkldnn_nchw, 5);
  CHECK_EQ(mkldnn_oihw, 5);
}

static void VerifyMem(const mkldnn::memory &mem) {
  mkldnn::memory::desc desc = mem.get_desc();
  mkldnn::memory::dims dims(desc.data.ndims);
  for (size_t i = 0; i < dims.size(); i++)
    dims[i] = desc.data.dims[i];
  mkldnn::memory::desc new_desc{dims,
      static_cast<mkldnn::memory::data_type>(desc.data.data_type),
      static_cast<mkldnn::memory::format_tag>(GetDefaultFormat(desc))};

  if (desc == new_desc) {
    VerifyDefMem(mem);
  } else {
    mkldnn::memory* src_mem = const_cast<mkldnn::memory*>(&mem);
    mkldnn::memory new_mem(new_desc, CpuEngine::Get()->get_engine());

    mkldnn::stream s(CpuEngine::Get()->get_engine());
    mkldnn::reorder(*src_mem, new_mem)
        .execute(s, *src_mem, new_mem);

    VerifyDefMem(new_mem);
  }
}

TEST(MKLDNN_NDArray, GetDataReorder) {
  TestArrayShapes tas = GetTestArrayShapes();
  mxnet::ShapeVector shapes = tas.shapes;
  std::vector<mkldnn::memory::desc> mds = tas.mds;


  // Reorder from the default to any other layout.
  for (auto s : shapes) {
    NDArray arr(s, Context());
    InitDefaultArray(&arr);
    for (auto md : mds) {
      if (s.Size() == md.get_size() / sizeof(mshadow::default_real_t)) {
        const mkldnn::memory *mem = arr.GetMKLDNNDataReorder(md);
        printf("reorder from (");
        for (size_t i = 0; i < s.ndim(); i++)
          printf("%ld, ", s[i]);
        printf(") to (");
        for (int i = 0; i < md.data.ndims; i++)
          printf("%ld, ", md.data.dims[i]);
        printf("), format: %d\n", static_cast<int>(GetDefaultFormat(md)));
        MKLDNNStream::Get()->Submit(false);
        VerifyMem(*mem);
        MKLDNNStream::Get()->Cleanup();
      }
    }
  }

  // Reorder from a special layout to another layout.
  for (auto s : shapes) {
    for (auto md : mds) {
      if (md.get_size() / sizeof(mshadow::default_real_t) == s.Size()) {
        NDArray arr(s, Context());
        // There is possibility that the dimensions of an NDArray doesn't match
        // with the MKLDNN memory inside.
        printf("Init array (");
        for (size_t i = 0; i < s.ndim(); i++)
          printf("%ld, ", s[i]);
        printf(") with MKLDNN memory (");
        for (int i = 0; i < md.data.ndims; i++)
          printf("%ld, ", md.data.dims[i]);
        printf("), format: %d\n", static_cast<int>(GetDefaultFormat(md)));
        InitMKLDNNArray(&arr, md);
        for (auto to_md : mds) {
          if (to_md.get_size() / sizeof(mshadow::default_real_t) == s.Size()) {
            const mkldnn::memory *mem = arr.GetMKLDNNDataReorder(to_md);
            printf("reorder from (");
            for (size_t i = 0; i < s.ndim(); i++)
              printf("%ld, ", s[i]);
            printf("), format: %d to (", static_cast<int>(GetDefaultFormat(to_md)));
            for (int i = 0; i < to_md.data.ndims; i++)
              printf("%ld, ", to_md.data.dims[i]);
            printf("), format: %d\n", static_cast<int>(GetDefaultFormat(to_md)));
            MKLDNNStream::Get()->Submit(false);
            VerifyMem(*mem);
            MKLDNNStream::Get()->Cleanup();
          }
        }
      }
    }
  }
}

TEST(MKLDNN_BASE, MKLDNNSum) {
  std::vector<NDArrayAttrs> in_arrs = GetTestInputArrays();
  std::vector<NDArrayAttrs> in_arrs2 = GetTestInputArrays(ArrayTypes::All, true);
  TestArrayShapes tas = GetTestArrayShapes();
  std::vector<mkldnn::memory::desc> mds = tas.mds;

  for (int i = 0; i < in_arrs.size(); i++) {
    auto in_arr = in_arrs[i];
    auto in_arr2 = in_arrs2[i];
    if (!SupportMKLDNN(in_arr.arr))
      continue;
    if (in_arr.arr.IsMKLDNNData() && in_arr.arr.IsView()) {
      continue;
    }
    std::vector<NDArrayAttrs> out_arrs = GetTestOutputArrays(in_arr.arr.shape(), mds);
    for (auto &out_arr : out_arrs) {
      auto in_mem1 = in_arr.arr.GetMKLDNNData();
      auto in_mem2 = in_arr2.arr.GetMKLDNNData();
      if (out_arr.arr.IsView())
        continue;
      auto out_mem = out_arr.arr.GetMKLDNNData();
      PrintVerifyMsg(in_arr, in_arr);
      op::MKLDNNSum(*in_mem1, *in_mem2, *out_mem);
      MKLDNNStream::Get()->Submit();
      VerifySumResult({&in_arr.arr, &in_arr2.arr}, {&out_arr.arr});
    }
  }

  // in place
  for (int i = 0; i < in_arrs.size(); i++) {
    auto in_arr = in_arrs[i];
    auto in_arr2 = in_arrs2[i];
    if (!SupportMKLDNN(in_arr.arr))
      continue;
    if (in_arr.arr.IsMKLDNNData() && in_arr.arr.IsView()) {
      continue;
    }
    auto input_mem = in_arr.arr.GetMKLDNNData();
    auto input_mem2 = in_arr2.arr.GetMKLDNNData();
    NDArrayAttrs orig_arr(in_arr.arr.Copy(in_arr.arr.ctx()), "In Place Copy");
    orig_arr.arr.WaitToRead();
    PrintVerifyMsg(orig_arr, in_arr);
    InitMKLDNNArray(&orig_arr.arr, input_mem->get_desc());
    orig_arr.arr.CopyFrom(*input_mem);
    op::MKLDNNSum(*input_mem, *input_mem2, *input_mem);
    MKLDNNStream::Get()->Submit();
    VerifySumResult({&orig_arr.arr, &in_arr2.arr}, {&in_arr.arr});
  }
}

TEST(MKLDNN_BASE, CreateMKLDNNMem) {
  std::vector<NDArrayAttrs> in_arrs = GetTestInputArrays();
  std::vector<NDArrayAttrs> in_arrs2 = GetTestInputArrays(ArrayTypes::All, true);
  TestArrayShapes tas = GetTestArrayShapes();
  std::vector<mkldnn::memory::desc> mds = tas.mds;
  MKLDNNStream *stream = MKLDNNStream::Get();

  // kWriteTo
  for (int i = 0; i < in_arrs.size(); i++) {
    auto in_arr = in_arrs[i];
    auto in_arr2 = in_arrs2[i];
    if (!SupportMKLDNN(in_arr.arr))
      continue;
    if (in_arr.arr.IsMKLDNNData() && in_arr.arr.IsView()) {
      continue;
    }
    std::vector<NDArrayAttrs> out_arrs = GetTestOutputArrays(in_arr.arr.shape(), mds);
    for (auto &out_arr : out_arrs) {
      auto in_mem = in_arr.arr.GetMKLDNNData();
      auto in_mem2 = in_arr2.arr.GetMKLDNNData();
      NDArray orig_output = out_arr.arr.Copy(out_arr.arr.ctx());
      orig_output.WaitToRead();
      PrintVerifyMsg(in_arr, out_arr);
      auto out_mem = out_arr.arr.GetMKLDNNData();
      auto output_mem_t = CreateMKLDNNMem(out_arr.arr, out_mem->get_desc(), kWriteTo);
      op::MKLDNNSum(*in_mem, *in_mem2, *output_mem_t.second);
      CommitOutput(out_arr.arr, output_mem_t);
      stream->Submit();
      VerifySumResult({&in_arr.arr, &in_arr2.arr}, {&out_arr.arr});
    }
  }

  // kWriteInPlace
  for (int i = 0; i < in_arrs.size(); i++) {
    auto in_arr = in_arrs[i];
    auto in_arr2 = in_arrs2[i];
    if (!SupportMKLDNN(in_arr.arr))
      continue;
    if (in_arr.arr.IsMKLDNNData() && in_arr.arr.IsView()) {
      continue;
    }
    auto input_mem = in_arr.arr.GetMKLDNNData();
    auto input_mem2 = in_arr2.arr.GetMKLDNNData();
    NDArrayAttrs orig_arr(in_arr.arr.Copy(in_arr.arr.ctx()), "In Place Copy");
    orig_arr.arr.WaitToRead();
    PrintVerifyMsg(orig_arr, in_arr);
    InitMKLDNNArray(&orig_arr.arr, input_mem->get_desc());
    orig_arr.arr.CopyFrom(*input_mem);
    auto output_mem_t = CreateMKLDNNMem(in_arr.arr,
        input_mem->get_desc(), kWriteInplace, &in_arr.arr);
    op::MKLDNNSum(*input_mem, *input_mem2, *output_mem_t.second);
    CommitOutput(in_arr.arr, output_mem_t);
    stream->Submit();
    VerifySumResult({&orig_arr.arr, &in_arr2.arr}, {&in_arr.arr});
  }

  // kAddTo
  for (int i = 0; i < in_arrs.size(); i++) {
    auto in_arr = in_arrs[i];
    auto in_arr2 = in_arrs2[i];
    if (!SupportMKLDNN(in_arr.arr))
      continue;
    if (in_arr.arr.IsMKLDNNData() && in_arr.arr.IsView()) {
      continue;
    }
    std::vector<NDArrayAttrs> out_arrs = GetTestOutputArrays(in_arr.arr.shape(), mds);
    for (auto &out_arr : out_arrs) {
      auto in_mem = in_arr.arr.GetMKLDNNData();
      auto in_mem2 = in_arr2.arr.GetMKLDNNData();
      NDArray orig_output = out_arr.arr.Copy(out_arr.arr.ctx());
      orig_output.WaitToRead();
      PrintVerifyMsg(in_arr, out_arr);
      auto out_mem = out_arr.arr.GetMKLDNNData();
      auto output_mem_t = CreateMKLDNNMem(out_arr.arr, out_mem->get_desc(), kAddTo);
      op::MKLDNNSum(*in_mem, *in_mem2, *output_mem_t.second);
      CommitOutput(out_arr.arr, output_mem_t);
      stream->Submit();
      VerifyAddRequest(
          {&in_arr.arr, &in_arr2.arr}, {&orig_output}, {&out_arr.arr}, VerifySumResult);
    }
  }

  // kNullOp
  for (int i = 0; i < in_arrs.size(); i++) {
    auto in_arr = in_arrs[i];
    auto in_arr2 = in_arrs2[i];
    if (!SupportMKLDNN(in_arr.arr))
      continue;
    if (in_arr.arr.IsMKLDNNData() && in_arr.arr.IsView()) {
      continue;
    }
    auto input_mem = in_arr.arr.GetMKLDNNData();
    auto input_mem2 = in_arr2.arr.GetMKLDNNData();
    NDArrayAttrs orig_arr(in_arr.arr.Copy(in_arr.arr.ctx()), "In Place Copy");
    orig_arr.arr.WaitToRead();
    PrintVerifyMsg(orig_arr, in_arr);
    InitMKLDNNArray(&orig_arr.arr, input_mem->get_desc());
    orig_arr.arr.CopyFrom(*input_mem);
    auto output_mem_t = CreateMKLDNNMem(in_arr.arr, input_mem->get_desc(), kNullOp);
    op::MKLDNNSum(*input_mem, *input_mem2, *output_mem_t.second);
    CommitOutput(in_arr.arr, output_mem_t);
    stream->Submit();
    // original and input should be the same since noop
    VerifyCopyResult({&orig_arr.arr}, {&in_arr.arr});
  }
}

TEST(MKLDNN_NDArray, GetTestInputArraysConcat) {
  auto in_arrs = GetTestInputArrays();
  for (int dim = 0; dim < 5; dim++) {
    for (int num_inputs = 2; num_inputs < 5; num_inputs++) {
      std::vector<float> scale_vector(dim + 1);
      for (size_t i = 0; i < dim + 1; ++i)
        scale_vector[i] = 1;
      scale_vector[dim] = num_inputs;
      std::vector<NDArrayAttrs> expanded_arrs = GetTestInputArrays(
          ArrayTypes::All, false, scale_vector);
      int i = 0;
      for (auto &arr : in_arrs) {
        if (dim >= arr.arr.shape().ndim())
          continue;
        auto ex_arr = expanded_arrs[i];
        PrintVerifyMsg(arr, ex_arr);
        EXPECT_EQ(arr.arr.shape().Size() * num_inputs, ex_arr.arr.shape().Size());
        EXPECT_EQ(arr.arr.shape()[dim] * num_inputs, ex_arr.arr.shape()[dim]);
        i++;
      }
    }
  }
}

TEST(MKLDNN_NDArray, GetTestOutputArraysConcat) {
  auto shapes_pds = GetTestArrayShapes();
  std::vector<mxnet::TShape> shapes = shapes_pds.shapes;
  std::vector<mkldnn::memory::desc> mds = shapes_pds.mds;
  for (auto &shape : shapes) {
    for (int dim = 0; dim < 5; dim++) {
      for (int num_inputs = 2; num_inputs < 5; num_inputs++) {
        if (shape.ndim() <= dim)
          continue;
        std::cout << "Extending " << shape << " dim " <<
                  dim << " and " << num_inputs << "num_inputs\n";
        std::vector<float> scale_vector(shape.ndim());
        for (int i = 0; i < shape.ndim(); i++)
          scale_vector[i] = 1;
        scale_vector[dim] = num_inputs;
        auto output_arrs = GetTestOutputArrays(shape, mds, scale_vector);
        for (auto &out_arr : output_arrs) {
          auto out_shape = out_arr.arr.shape();
          EXPECT_EQ(shape.Size() * num_inputs, out_arr.arr.shape().Size());
          EXPECT_EQ(shape[dim] * num_inputs, out_arr.arr.shape()[dim]);
        }
      }
    }
  }
}

TEST(MKLDNN_NDArray, CopyFrom) {
  TestArrayShapes tas = GetTestArrayShapes();
  std::vector<mkldnn::memory::desc> mds = tas.mds;

  std::vector<NDArrayAttrs> in_arrs = GetTestInputArrays();
  for (auto &in_arr : in_arrs) {
    if (in_arr.arr.IsMKLDNNData() && in_arr.arr.IsView())
      continue;
    std::vector<NDArrayAttrs> out_arrs = GetTestOutputArrays(in_arr.arr.shape(), mds);
    for (auto &out_arr : out_arrs) {
      const mkldnn::memory *mem = in_arr.arr.GetMKLDNNData();
      out_arr.arr.CopyFrom(*mem);
      MKLDNNStream::Get()->Submit();
      std::vector<NDArray *> inputs(1);
      inputs[0] = &in_arr.arr;
      VerifyCopyResult(inputs, {&out_arr.arr});
    }
  }
}

#endif  // MXNET_USE_MKLDNN  == 1
