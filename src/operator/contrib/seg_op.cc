#include "./seg_op.h"
namespace mxnet {
namespace op {

namespace seg_op {

template<int reduce_type>
void SegReduceImpl(const Tensor<cpu, 2, float> &dst,
                   const Tensor<cpu, 2, float> &data,
                   const Tensor<cpu, 1, int> &indptr,
                   const OpReqType req,
                   const OpContext& ctx,
                   Stream<cpu>* s) {
  if (req == kNullOp) return;

  int batch_num = data.shape_[0];
  int nnz = data.shape_[1];
  int seg_num = indptr.shape_[0] - 1;
  for (int k = 0; k < batch_num; k++) {
    for (int i = 0; i < seg_num; i++) {
      float res = 0.0;
      if (reduce_type == SegReduceType::kMax) {
        res = std::numeric_limits<float>::lowest();
      } else if (reduce_type == SegReduceType::kMin) {
        res = std::numeric_limits<float>::max();
      } else if (reduce_type == SegReduceType::kSum) {
        res = 0.0;
      } else {
        LOG(FATAL) << "reduce_type = " << reduce_type << " is not supported!";
      }
      for (int j = indptr[i]; j < indptr[i + 1]; j++) {
        if (reduce_type == SegReduceType::kSum) {
          res += data[k][j];
        } else if (reduce_type == SegReduceType::kMax) {
          res = std::max(res, data[k][j]);
        } else if (reduce_type == SegReduceType::kMin) {
          res = std::min(res, data[k][j]);
        } else {
          LOG(FATAL) << "reduce_type = " << reduce_type << " is not supported!";
        }
      }
      if (req == kAddTo) {
        dst[k][i] += res;
      } else {
        dst[k][i] = res;
      }
    }
  }
}

template<typename OP>
void SegBroadcastBinaryImpl(const Tensor<cpu, 2, float> &dst,
                            const Tensor<cpu, 2, float> &lhs,
                            const Tensor<cpu, 2, float> &rhs,
                            const Tensor<cpu, 1, int> &indptr,
                            const OpReqType req,
                            const OpContext& ctx,
                            Stream<cpu>* s) {
  if (req == kNullOp) return;
  int batch_num = lhs.shape_[0];
  int nnz = lhs.shape_[1];
  int seg_num = rhs.shape_[1];
  if(req != kAddTo) {
    std::memset(dst.dptr_, 0, sizeof(float) * batch_num * nnz);
  }
  for (int k = 0; k < batch_num; k++) {
    for (int i = 0; i < seg_num; i++) {
      for (int j = indptr[i]; j < indptr[i + 1]; j++) {
        if(req == kAddTo) {
          dst[k][j] += OP::Map(lhs[k][j], rhs[k][i]);
        } else {
          dst[k][j] = OP::Map(lhs[k][j], rhs[k][i]);
        }
      }
    }
  }
}

void SegSoftmaxImpl(const Tensor<cpu, 2, float> &dst,
                    const Tensor<cpu, 2, float> &data,
                    const Tensor<cpu, 1, int> &indptr,
                    const OpReqType req,
                    const OpContext& ctx,
                    Stream<cpu>* s) {
  if (req == kNullOp) return;
  CHECK_NE(req, kAddTo) << "AddTo for seg_softmax is currently not supported!";
  int batch_num = data.shape_[0];
  int nnz = data.shape_[1];
  int seg_num = indptr.shape_[0] - 1;
  for (int k = 0; k < batch_num; k++) {
    for (int i = 0; i < nnz; i++) {
      dst[k][i] = 0;
    }
  }
  for (int k = 0; k < batch_num; k++) {
    for (int i = 0; i < seg_num; i++) {
      float sum_val;
      float max_val;
      red::sum::SetInitValue(sum_val);
      red::maximum::SetInitValue(max_val);
      for (int j = indptr[i]; j < indptr[i + 1]; j++) {
        red::maximum::Reduce(max_val, data[k][j]);
      }
      for (int j = indptr[i]; j < indptr[i + 1]; j++) {
        dst[k][j] = expf(data[k][j] - max_val);
      }
      for (int j = indptr[i]; j < indptr[i + 1]; j++) {
        red::sum::Reduce(sum_val, dst[k][j]);
      }
      for (int j = indptr[i]; j < indptr[i + 1]; j++) {
        dst[k][j] /= sum_val;
      }
    }
  }
}

/* Backward pass of the softmax
dst: Shape(batch_num, nnz)
ograd: Shape(batch_num, seg_num)
*/
void SegSoftmaxBackwardImpl(const Tensor<cpu, 2, float> &dst,
                            const Tensor<cpu, 2, float> &ograd,
                            const Tensor<cpu, 2, float> &val,
                            const Tensor<cpu, 1, int> &indptr,
                            const OpReqType req,
                            const OpContext& ctx,
                            Stream<cpu>* s) {
  if (req == kNullOp) return;
  int batch_num = ograd.shape_[0];
  int nnz = ograd.shape_[1];
  int seg_num = indptr.shape_[1] - 1;
  for (int k = 0; k < batch_num; k++) {
    for(int i = 0; i < seg_num; i++) {
      float sum_val = 0;
      for(int j = indptr[i]; j < indptr[i + 1]; j++) {
        sum_val += ograd[k][j] * val[k][j];
      }
      for(int j = indptr[i]; j < indptr[i + 1]; j++) {
        float g_val = val[k][j] * (ograd[k][j] - sum_val);
        if(req == kAddTo) {
          dst[k][j] += g_val;
        } else {
          dst[k][j] = g_val;
        }
      }
    }
  }
}

void SegTakeKCorrImpl(const Tensor<cpu, 2, float> &dst,
                      const Tensor<cpu, 3, float> &embed1,
                      const Tensor<cpu, 3, float> &embed2,
                      const Tensor<cpu, 1, int> &neighbor_ids,
                      const Tensor<cpu, 1, int> &neighbor_ind_ptr,
                      const OpReqType req,
                      const OpContext& ctx,
                      Stream<cpu>* s) {
  if (req == kNullOp) return;
  int K = embed1.shape_[0];
  int node_num = embed1.shape_[1];
  int feat_dim = embed1.shape_[2];
  int neighbor_node_num = embed2.shape_[1];
  int nnz = neighbor_ids.shape_[0];
  if(req != kAddTo) {
    std::memset(dst.dptr_, 0, sizeof(float) * K * nnz);
  }
  for(int k = 0; k < K; k++) {
        #pragma omp parallel for
        for(int i = 0; i < node_num; i++) {
            for (int j = neighbor_ind_ptr[i]; j < neighbor_ind_ptr[i + 1]; j++) {
                // Calculate the distance between embed1[k, i, :] and embed2[k, neighbor_ids[j], :]
                for(int c = 0; c < feat_dim; c++) {
                    dst[k][j] += embed1[k][i][c] * embed2[k][neighbor_ids[j]][c];
                }
            }
        }
    }
}

void SegTakeKCorrBackwardEmbed1Impl(const Tensor<cpu, 3, float> &dst,
                                    const Tensor<cpu, 2, float> &ograd,
                                    const Tensor<cpu, 3, float> &embed2,
                                    const Tensor<cpu, 1, int> &neighbor_ids,
                                    const Tensor<cpu, 1, int> &neighbor_ind_ptr,
                                    const OpReqType req,
                                    const OpContext& ctx,
                                    Stream<cpu>* s) {
  if (req == kNullOp) return;
  int K = ograd.shape_[0];
  int node_num = neighbor_ind_ptr.shape_[0] - 1;
  int feat_dim = embed2.shape_[2];
  int neighbor_node_num = embed2.shape_[1];
  int nnz = neighbor_ids.shape_[0];
  if(req != kAddTo) {
    std::memset(dst.dptr_, 0, sizeof(float) * K * node_num * feat_dim);
  }
  for(int k = 0; k < K; k++) {
      #pragma omp parallel for
      for(int i = 0; i < node_num; i++) {
          for(int j = neighbor_ind_ptr[i]; j < neighbor_ind_ptr[i + 1]; j++) {
              for(int c = 0; c < feat_dim; c++) {
                  dst[k][i][c] += ograd[k][j] * embed2[k][neighbor_ids[j]][c];
              }
          }
      }
  }
}

void SegTakeKCorrBackwardEmbed2Impl(const Tensor<cpu, 3, float> &dst,
                                    const Tensor<cpu, 2, float> &ograd,
                                    const Tensor<cpu, 3, float> &embed1,
                                    const Tensor<cpu, 1, int> &neighbor_ids,
                                    const Tensor<cpu, 1, int> &neighbor_ind_ptr,
                                    const OpReqType req,
                                    const OpContext& ctx,
                                    Stream<cpu>* s) {
  if (req == kNullOp) return;
  int K = ograd.shape_[0];
  int node_num = embed1.shape_[1];
  int feat_dim = embed1.shape_[2];
  int neighbor_node_num = dst.shape_[1];
  int nnz = neighbor_ids.shape_[0];
  if(req != kAddTo) {
    std::memset(dst.dptr_, 0, sizeof(float) * K * neighbor_node_num * feat_dim);
  }
  std::vector<int> seg_ids(nnz);
  for (int i = 0; i < node_num; i++) {
    for (int j = neighbor_ind_ptr[i]; j < neighbor_ind_ptr[i + 1]; j++) {
      seg_ids[j] = i;
    }
  }
  #pragma omp parallel for
  for(int k = 0; k < K; k++) {
      for(int i = 0; i < nnz; i++) {
          for(int c = 0; c < feat_dim; c++) {
              dst[k][neighbor_ids[i]][c] += ograd[k][i] * embed1[k][seg_ids[i]][c];
          }
      }
  }
}

template<int pool_type>
void SegPoolImpl(const Tensor<cpu, 3, float> &dst_value,
                 const Tensor<cpu, 3, int> &pool_indices,
                 const Tensor<cpu, 3, float> &data,
                 const Tensor<cpu, 1, int> &indices,
                 const Tensor<cpu, 1, int> &indptr,
                 const OpReqType req,
                 const OpContext &ctx,
                 Stream<cpu>* s) {
  if (req == kNullOp) return;
  CHECK_NE(req, kAddTo) << "Not supported";
  int batch_num = data.shape_[0];
  int total_ind_num = data.shape_[1];
  int feat_dim = data.shape_[2];
  int seg_num = dst_value.shape_[1];
  int nnz = indices.shape_[0];
  for (int k = 0; k < batch_num; k++) {
    #pragma omp parallel for
    for (int i = 0; i < seg_num; i++) {
      for (int c = 0; c < feat_dim; c++) {
        if (pool_type == SegReduceType::kSum || pool_type == SegReduceType::kMean) {
          dst_value[k][i][c] = 0;
        } else if (pool_type == SegReduceType::kMax) {
          dst_value[k][i][c] = std::numeric_limits<float>::lowest();
          if(indptr[i + 1] == indptr[i]) {
            dst_value[k][i][c] = 0;
          }
          pool_indices[k][i][c] = -1;
        } else {
          LOG(FATAL) << "Not Implemented!";
        }
      }
      for (int j = indptr[i]; j < indptr[i + 1]; j++) {
        for (int c = 0; c < feat_dim; c++) {
          float data_val = data[k][indices[j]][c];
          if (pool_type == SegReduceType::kSum || pool_type == SegReduceType::kMean) {
            dst_value[k][i][c] += data_val;
          } else if (pool_type == SegReduceType::kMax) {
            if (data_val > dst_value[k][i][c]) {
              dst_value[k][i][c] = data_val;
              pool_indices[k][i][c] = j;
            }
          } else {
            LOG(FATAL) << "Not Implemented!";
          }
        }
      }
      if (pool_type == SegReduceType::kMean && (indptr[i + 1] - indptr[i]) > 0) {
        for (int c = 0; c < feat_dim; c++) {
          dst_value[k][i][c] /= (indptr[i + 1] - indptr[i]);
        }
      }
    }
  }
  return;
}

template<int pool_type>
void SegPoolBackwardImpl(const Tensor<cpu, 3, float> &dst,
                         const Tensor<cpu, 3, float> &ograd,
                         const Tensor<cpu, 3, int> &pool_indices,
                         const Tensor<cpu, 1, int> &indices,
                         const Tensor<cpu, 1, int> &indptr,
                         const OpReqType req,
                         const OpContext &ctx,
                         Stream<cpu>* s) {
  if (req == kNullOp) return;
  int batch_num = dst.shape_[0];
  int total_ind_num = dst.shape_[1];
  int feat_dim = dst.shape_[2];
  int seg_num = ograd.shape_[1];
  if(req != kAddTo) {
    std::memset(dst.dptr_, 0, sizeof(float) * batch_num * total_ind_num * feat_dim);
  }
  #pragma omp parallel for
  for (int k = 0; k < batch_num; k++) {
    for (int i = 0; i < seg_num; i++) {
      for (int j = indptr[i]; j < indptr[i + 1]; j++) {
        for (int c = 0; c < feat_dim; c++) {
          if (pool_type == SegReduceType::kMean) {
            dst[k][indices[j]][c] += ograd[k][i][c] / (indptr[i + 1] - indptr[i]);
          } else if(pool_type == SegReduceType::kSum) {
            dst[k][indices[j]][c] += ograd[k][i][c];
          } else {
            dst[k][indices[j]][c] += ograd[k][i][c] * (pool_indices[k][i][c] == j);
          }
        }
      }
    }
  }
}
}  // namespace seg_op

DMLC_REGISTER_PARAMETER(NNZOnlyParam);
DMLC_REGISTER_PARAMETER(SegTakeKCorrParam);
DMLC_REGISTER_PARAMETER(SegPoolParam);

NNVM_REGISTER_OP(_contrib_seg_sum)
.describe(R"code(Reduce the last dimension of the input based on the given segment indicators.

data: Shape (batch_num, nnz)
indptr: Shape (seg_num + 1,)

ret: Shape (batch_num, seg_num)


for k = 0 to batch_num - 1
    for i = 0 to seg_num - 1
        ret[k, i] = reduce(data[k, indptr[i]], ..., data[k, indptr[i + 1] - 1])

Examples::

    out = seg_sum(data=data, indptr=indptr)

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "indptr"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", SegReduceShape)
.set_attr<nnvm::FInferType>("FInferType", SegReduceSetType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", SegReduceForward<cpu, seg_op::SegReduceType::kSum>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    auto p_data = MakeNode("_contrib__backward_seg_sum", n->attrs.name + "_backward_data", {ograds[0], n->inputs[1]}, nullptr, &n);
    auto p_indptr = MakeNode("zeros_like", n->attrs.name + "_backward_indptr",
                             {n->inputs[1]}, nullptr, &n);
    std::vector<nnvm::NodeEntry> ret;
    ret.emplace_back(nnvm::NodeEntry{p_data, 0, 0});
    ret.emplace_back(nnvm::NodeEntry{p_indptr, 0, 0});
    return ret;
  })
.add_argument("data", "NDArray-or-Symbol", "The input data.")
.add_argument("indptr", "NDArray-or-Symbol", "The segment indicators.");

NNVM_REGISTER_OP(_contrib__backward_seg_sum)
.describe(R"code(
Backward of seg_softmax
inputs will be ograds, indptr
)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", SegBroadcastToForward<cpu>);

NNVM_REGISTER_OP(_contrib_seg_broadcast_add)
.describe(R"code(Broadcast rhs according to the segment indicators and add to lhs to get the result.

lhs: Shape (batch_num, nnz)
rhs: Shape (batch_num, seg_num)
indptr: Shape (seg_num + 1,)

ret: Shape (batch_num, nnz)


ret = seg_broadcast_add(lhs, rhs, indptr)

Examples::

    ret = seg_broadcast_add(lhs=lhs, rhs=rhs, indptr=indptr)

)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs", "indptr"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", SegBroadcastBinaryShape)
.set_attr<nnvm::FInferType>("FInferType", SegBroadcastBinarySetType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", SegBroadcastBinaryForward<cpu, mshadow::op::plus>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    auto p_lhs = MakeNode("identity", n->attrs.name + "_backward_lhs", {ograds[0]}, nullptr, &n);
    auto p_rhs = nnvm::Node::Create();
    p_rhs->attrs.op = nnvm::Op::Get("_contrib_seg_sum");
    p_rhs->attrs.name = n->attrs.name + "_backward_rhs";
    p_rhs->inputs.push_back(ograds[0]);
    p_rhs->inputs.push_back(n->inputs[2]);
    p_rhs->control_deps.emplace_back(n);
    auto p_indptr = MakeNode("zeros_like", n->attrs.name + "_backward_indptr",
                             {n->inputs[2]}, nullptr, &n);
    std::vector<nnvm::NodeEntry> ret;
    ret.emplace_back(nnvm::NodeEntry{p_lhs, 0, 0});
    ret.emplace_back(nnvm::NodeEntry{p_rhs, 0, 0});
    ret.emplace_back(nnvm::NodeEntry{p_indptr, 0, 0});
    return ret;
  })
.add_argument("lhs", "NDArray-or-Symbol", "The left hand side.")
.add_argument("rhs", "NDArray-or-Symbol", "The right hand side (need broadcasting).")
.add_argument("indptr", "NDArray-or-Symbol", "The segment indicators of rhs.");

NNVM_REGISTER_OP(_contrib_seg_broadcast_mul)
.describe(R"code(Broadcast rhs according to the segment indicators and multiply to lhs to get the result.

lhs: Shape (batch_num, nnz)
rhs: Shape (batch_num, seg_num)
indptr: Shape (seg_num + 1,)

ret: Shape (batch_num, nnz)


ret = seg_broadcast_mul(lhs, rhs, indptr)

Examples::

    ret = seg_broadcast_mul(lhs=lhs, rhs=rhs, indptr=indptr)

)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs", "indptr"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", SegBroadcastBinaryShape)
.set_attr<nnvm::FInferType>("FInferType", SegBroadcastBinarySetType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", SegBroadcastBinaryForward<cpu, mshadow::op::mul>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    auto p_lhs = MakeNode("_contrib_seg_broadcast_mul", n->attrs.name + "_backward_lhs", {ograds[0], n->inputs[1], n->inputs[2]}, nullptr, &n);
    auto p_rhs_mid_stage = MakeNode("elemwise_mul", n->attrs.name + "_backward_rhs_mid_stage", {ograds[0], n->inputs[0]}, nullptr, &n);
    auto p_rhs = MakeNode("_contrib_seg_sum", n->attrs.name + "_backward_rhs", {nnvm::NodeEntry{p_rhs_mid_stage, 0, 0}, n->inputs[2]}, nullptr, &n);
    auto p_indptr = MakeNode("zeros_like", n->attrs.name + "_backward_indptr",
                             {n->inputs[2]}, nullptr, &n);
    std::vector<nnvm::NodeEntry> ret;
    ret.emplace_back(nnvm::NodeEntry{p_lhs, 0, 0});
    ret.emplace_back(nnvm::NodeEntry{p_rhs, 0, 0});
    ret.emplace_back(nnvm::NodeEntry{p_indptr, 0, 0});
    return ret;
  })
.add_argument("lhs", "NDArray-or-Symbol", "The left hand side.")
.add_argument("rhs", "NDArray-or-Symbol", "The right hand side (need broadcasting).")
.add_argument("indptr", "NDArray-or-Symbol", "The segment indicators of rhs.");

NNVM_REGISTER_OP(_contrib_seg_broadcast_to)
.describe(R"code(Broadcast rhs according to the segment indicators and add to lhs to get the result.

data: Shape (batch_num, seg_num)
indptr: Shape (seg_num + 1,)
int nnz

ret: Shape (batch_num, nnz)


Examples::

    ret = seg_broadcast_to(data=data, indptr=indptr, nnz=nnz)

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NNZOnlyParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "indptr"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", SegBroadcastToShape)
.set_attr<nnvm::FInferType>("FInferType", SegBroadcastToSetType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", SegBroadcastToForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    auto p_data = nnvm::Node::Create();
    p_data->attrs.op = nnvm::Op::Get("_contrib_seg_sum");
    p_data->attrs.name = n->attrs.name + "_backward_data";
    p_data->inputs.push_back(ograds[0]);
    p_data->inputs.push_back(n->inputs[1]);
    p_data->control_deps.emplace_back(n);
    auto p_indptr = MakeNode("zeros_like", n->attrs.name + "_backward_indptr",
                             {n->inputs[1]}, nullptr, &n);
    std::vector<nnvm::NodeEntry> ret;
    ret.emplace_back(nnvm::NodeEntry{p_data, 0, 0});
    ret.emplace_back(nnvm::NodeEntry{p_indptr, 0, 0});
    return ret;
  })
.add_argument("data", "NDArray-or-Symbol", "The data to broadcast.")
.add_argument("indptr", "NDArray-or-Symbol", "The segment indicators of rhs.")
.add_arguments(NNZOnlyParam::__FIELDS__());


NNVM_REGISTER_OP(_contrib_seg_softmax)
.describe(R"code(Calculate the softmax of the the input based on the given segment indicators.

data: Shape (batch_num, nnz)
indptr: Shape (seg_num + 1,)

ret: Shape (batch_num, nnz)


for k = 0 to batch_num - 1
    for i = 0 to seg_num - 1
        ret[k, indptr[i]:indptr[i+1]] = softmax(data[k, indptr[i]:indptr[i+1]])

Examples::

    out = seg_softmax(data=data, indptr=indptr)

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "indptr"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", SegSoftmaxShape)
.set_attr<nnvm::FInferType>("FInferType", SegReduceSetType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", SegSoftmaxForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    auto p_data = MakeNode("_contrib__backward_seg_softmax", n->attrs.name + "_backward_data", {ograds[0], nnvm::NodeEntry{n, 0, 0 }, n->inputs[1]}, nullptr, &n);
    auto p_indptr = MakeNode("zeros_like", n->attrs.name + "_backward_indptr",
                             {n->inputs[1]}, nullptr, &n);
    std::vector<nnvm::NodeEntry> ret;
    ret.emplace_back(nnvm::NodeEntry{p_data, 0, 0});
    ret.emplace_back(nnvm::NodeEntry{p_indptr, 0, 0});
    return ret;
  })
.add_argument("data", "NDArray-or-Symbol", "The input data.")
.add_argument("indptr", "NDArray-or-Symbol", "The segment indicators.");

NNVM_REGISTER_OP(_contrib__backward_seg_softmax)
.describe(R"code(
Backward of seg_softmax
inputs will be ograds, val, indptr
)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", SegSoftmaxBackward<cpu>);

NNVM_REGISTER_OP(_contrib_seg_take_k_corr)
.describe(R"code(For all the nodes, computes the inner product between the node
and it's neighborhoods and add to dst.

We assume the node_ids are 0, 1, 2, ..., node_num - 1

embed1: Shape (K, node_num, feat_dim)
embed2: Shape (K, neighbor_node_num, feat_dim)
neighbor_ids: Shape (nnz, )
neighbor_indptr: Shape(node_num + 1, )

dst: Shape (K, nnz)

use mul to compute the inner-product and use squared_diff to compute the squared distance.
TODO(sxjscience): add squared distance

IMPORTANT! If you plan to deal with the case where the input is (batch_size, K, node_num, feat_dim),
you can simply set K = batch_size * K if the neigbor_ids are the same for all elements in the batch.

for k = 0 to K-1
    for i = 0  to node_num - 1
        for j = ind_ptr[i] to ind_ptr[i+1] - 1
            neighbor_id = neighbor_ids[j]
            dst[k, j] += InnerProduct(embed1[k, i], embed2[k, neighbor_id]) or ||embed1[k, i] - embed2[k, neighbor_id]||^2_2
Examples::

    out = seg_take_k_corr(embed1=embed1,
                          embed2=embed2,
                          neighbor_ids=neighbor_ids,
                          neighbor_indptr=neighbor_indptr)

)code" ADD_FILELINE)
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"embed1", "embed2", "neighbor_ids", "neighbor_indptr"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", SegTakeKCorrShape)
.set_attr<nnvm::FInferType>("FInferType", SegTakeKCorrSetType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", SegTakeKCorrForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    auto p_embed1 = MakeNode("_contrib_seg_weighted_pool", n->attrs.name + "_backward_embed1", {n->inputs[1], ograds[0], n->inputs[2], n->inputs[3]}, nullptr, &n);
    auto p_embed2 = MakeNode("_contrib__backward_seg_take_k_corr_embed2", n->attrs.name + "_backward_embed2", {ograds[0], n->inputs[0], n->inputs[2], n->inputs[3]}, nullptr, &n);
    auto p_neighbor_ids = MakeNode("zeros_like", n->attrs.name + "_backward_neighbor_ids", { n->inputs[2] }, nullptr, &n);
    auto p_neighbor_indptr = MakeNode("zeros_like", n->attrs.name + "_backward_neighbor_indptr", {n->inputs[3]}, nullptr, &n);
    std::vector<nnvm::NodeEntry> ret;
    ret.emplace_back(nnvm::NodeEntry{p_embed1, 0, 0});
    ret.emplace_back(nnvm::NodeEntry{p_embed2, 0, 0});
    ret.emplace_back(nnvm::NodeEntry{p_neighbor_ids, 0, 0});
    ret.emplace_back(nnvm::NodeEntry{p_neighbor_indptr, 0, 0});
    return ret;
  })
.add_argument("embed1", "NDArray-or-Symbol", "Embedding of the nodes.")
.add_argument("embed2", "NDArray-or-Symbol", "Embedding of the neighborhood nodes.")
.add_argument("neighbor_ids", "NDArray-or-Symbol", "The neighborhood ids.")
.add_argument("neighbor_indptr", "NDArray-or-Symbol", "The segment indicators.");

NNVM_REGISTER_OP(_contrib_seg_weighted_pool)
.describe(R"code(
Compute weighted average of values in the segments

data: Shape (batch_size, total_ind_num, feat_dim)
weights: Shape (batch_size, nnz)
indices: Shape (nnz, )
indptr: Shape (seg_num + 1,)

dst: Shape (batch_size, seg_num, feat_dim)

for k = 0 to K-1
    for i = 0  to node_num - 1
        if !add_to
            dst[k, i, :] = 0
        for j = ind_ptr[i] to ind_ptr[i+1] - 1
            dst[k, i, :] += weights[k, j] * data[k, neighbor_ids[j], :]

Examples::
  
    out = seg_weighted_pool(data=data, weights=weights, indices=indices, indptr=indptr)
)code" ADD_FILELINE)
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "weights", "indices", "indptr"};
})
.set_attr<mxnet::FInferShape>("FInferShape", SegWeightedPoolShape)
.set_attr<nnvm::FInferType>("FInferType", SegTakeKCorrSetType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", SegWeightedPoolForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    auto p_data = MakeNode("_contrib__backward_seg_take_k_corr_embed2", n->attrs.name + "_backward_data", {n->inputs[1], ograds[0], n->inputs[2], n->inputs[3]}, nullptr, &n);
    auto p_weights = MakeNode("_contrib_seg_take_k_corr", n->attrs.name + "_backward_weights", {ograds[0], n->inputs[0], n->inputs[2], n->inputs[3]}, nullptr, &n);
    auto p_indices = MakeNode("zeros_like", n->attrs.name + "_backward_indices", { n->inputs[2] }, nullptr, &n);
    auto p_indptr = MakeNode("zeros_like", n->attrs.name + "_backward_indptr", {n->inputs[3]}, nullptr, &n);
    std::vector<nnvm::NodeEntry> ret;
    ret.emplace_back(nnvm::NodeEntry{p_data, 0, 0});
    ret.emplace_back(nnvm::NodeEntry{p_weights, 0, 0});
    ret.emplace_back(nnvm::NodeEntry{p_indices, 0, 0});
    ret.emplace_back(nnvm::NodeEntry{p_indptr, 0, 0});
    return ret;
  })
.add_argument("data", "NDArray-or-Symbol", "Embedding of the nodes.")
.add_argument("weights", "NDArray-or-Symbol", "Weights of the pooling operation.")
.add_argument("indices", "NDArray-or-Symbol", "The corresponding indices in the data.")
.add_argument("indptr", "NDArray-or-Symbol", "The segment indicators.");

NNVM_REGISTER_OP(_contrib__backward_seg_take_k_corr_embed2)
.describe(R"code(
Backward of seg_take_k_corr w.r.t embed2 and backward of seg_weighted_pool w.r.t data

inputs will be ograds, embed1, neighbor_ids, neighbor_indptr

ograds: Shape (K, nnz)
embed1: Shape (K, node_num, feat_dim)
neighbor_ids: Shape (nnz, )
neighbor_indptr: Shape(node_num + 1, )

dst: Shape (K, neighbor_node_num, feat_dim)

for k = 0 to K-1
    for i = 0  to node_num - 1
        for j = ind_ptr[i] to ind_ptr[i+1] - 1
            dst[k, neighbor_ids[j], :] += g_out[k, j] * embed1[k, rev_node_ids[j], :]

TODO(sxjscience) Optimize the speed of this function
First reorganize the data in neighbor_ids, g_out, embed1, ...

for k = 0 to K-1
    for i = 0 to neighbor_node_num - 1
        for j = rev_ind_ptr[i] to rev_ind_ptr[i + 1] - 1
            dst[k, i, :] += reorder_g_out[k, j] * embed1[k, node_ids[j], :]

)code" ADD_FILELINE)
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", SegTakeKCorrBackwardEmbed2<cpu>);

NNVM_REGISTER_OP(_contrib_seg_pool)
.describe(R"code(Pooling of the values in the segments

data : Shape (batch_size, total_ind_num, feat_dim)
indices : Shape (nnz,)
indptr : Shape (seg_num + 1,)
pool_type : 'avg' or 'sum' or 'max'

dst : Shape (batch_size, seg_num, feat_dim)

Examples::

    out = seg_pool(data=data,
                   indices=indices,
                   indptr=indptr,
                   pool_type='avg')

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<SegPoolParam>)
.set_num_inputs(3)
.set_num_outputs([](const NodeAttrs& attrs) {
    const SegPoolParam& param = nnvm::get<SegPoolParam>(attrs.parsed);
    if (param.pool_type == seg_op::SegReduceType::kMax) {
      return static_cast<uint32_t>(2);
    } else {
      return static_cast<uint32_t>(1);
    }
  })
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs", [](const NodeAttrs& attrs) {
    return static_cast<uint32_t>(1);
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "indices", "indptr"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", SegPoolShape)
.set_attr<nnvm::FInferType>("FInferType", SegPoolSetType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", SegPoolForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    const SegPoolParam& param = nnvm::get<SegPoolParam>(n->attrs.parsed);
    std::vector<nnvm::NodeEntry> ret;
    nnvm::NodePtr p_data;
    if(param.pool_type == seg_op::SegReduceType::kMax) {
      p_data = MakeNode("_contrib__backward_seg_max_pool", n->attrs.name + "_backward_data", {ograds[0], nnvm::NodeEntry{n, 1, 0}, n->inputs[1], n->inputs[2] }, nullptr, &n);
    } else {
      p_data = MakeNode("_contrib__backward_seg_sum_mean_pool", n->attrs.name + "_backward_data", {ograds[0], n->inputs[1], n->inputs[2]}, &(n->attrs.dict), &n);
    }
    auto p_indices = MakeNode("zeros_like", n->attrs.name + "_backward_indices", { n->inputs[1] }, nullptr, &n);
    auto p_indptr = MakeNode("zeros_like", n->attrs.name + "_backward_indptr", {n->inputs[2]}, nullptr, &n);
    
    ret.emplace_back(nnvm::NodeEntry{p_data, 0, 0});
    ret.emplace_back(nnvm::NodeEntry{p_indices, 0, 0});
    ret.emplace_back(nnvm::NodeEntry{p_indptr, 0, 0});
    return ret;
  })
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_argument("indices", "NDArray-or-Symbol", "Indices to take the reduction.")
.add_argument("indptr", "NDArray-or-Symbol", "The neighborhood ids.")
.add_arguments(SegPoolParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib__backward_seg_sum_mean_pool)
.describe(R"code(
Backward pass of seg_pool

ograd : Shape (batch_size, seg_num, feat_dim)
indices : Shape (nnz,)
indptr : Shape (seg_num + 1,)

dst : Shape (batch_size, total_ind_num, feat_dim)

pool_type can be 'avg', 'sum'
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<SegPoolParam>)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", SegSumMeanPoolBackward<cpu>)
.add_arguments(SegPoolParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib__backward_seg_max_pool)
.describe(R"code(
Backward pass of seg_pool

ograd : Shape (batch_size, seg_num, feat_dim)
pool_indices : Shape (batch_size, seg_num, feat_dim,)
indices : Shape (nnz,)
indptr : Shape (seg_num + 1,)

dst : Shape (batch_size, total_ind_num, feat_dim)

)code" ADD_FILELINE)
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", SegMaxPoolBackward<cpu>);
}  // namespace op
}  // namespace mxnet
