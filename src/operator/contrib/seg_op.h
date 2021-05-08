#ifndef MXNET_OPERATOR_SEG_OP_H_
#define MXNET_OPERATOR_SEG_OP_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <map>
#include <limits>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <type_traits>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

namespace seg_op {
using namespace mshadow;
enum SegReduceType {kSum, kMean, kMax, kMin};
enum SegBroadcastBinaryType {};
enum SegTakeKCorrType {kInnerProduct, kEuclidean};

// SegReduce
template<int reduce_type>
void SegReduceImpl(const Tensor<cpu, 2, float> &dst,
                   const Tensor<cpu, 2, float> &data,
                   const Tensor<cpu, 1, int> &indptr,
                   const OpReqType req,
                   const OpContext& ctx,
                   Stream<cpu>* s);
template<int reduce_type>
void SegReduceImpl(const Tensor<gpu, 2, float> &dst,
                   const Tensor<gpu, 2, float> &data,
                   const Tensor<gpu, 1, int> &indptr,
                   const OpReqType req,
                   const OpContext& ctx,
                   Stream<gpu>* s);
// SegBroadcastBinary
template<typename OP>
void SegBroadcastBinaryImpl(const Tensor<cpu, 2, float> &dst,
                            const Tensor<cpu, 2, float> &lhs,
                            const Tensor<cpu, 2, float> &rhs,
                            const Tensor<cpu, 1, int> &indptr,
                            const OpReqType req,
                            const OpContext& ctx,
                            Stream<cpu>* s);
template<typename OP>
void SegBroadcastBinaryImpl(const Tensor<gpu, 2, float> &dst,
                            const Tensor<gpu, 2, float> &lhs,
                            const Tensor<gpu, 2, float> &rhs,
                            const Tensor<gpu, 1, int> &indptr,
                            const OpReqType req,
                            const OpContext& ctx,
                            Stream<gpu>* s);
// SegSoftmax
void SegSoftmaxImpl(const Tensor<cpu, 2, float> &dst,
                    const Tensor<cpu, 2, float> &data,
                    const Tensor<cpu, 1, int> &indptr,
                    const OpReqType req,
                    const OpContext& ctx,
                    Stream<cpu>* s);
void SegSoftmaxImpl(const Tensor<gpu, 2, float> &dst,
                    const Tensor<gpu, 2, float> &data,
                    const Tensor<gpu, 1, int> &indptr,
                    const OpReqType req,
                    const OpContext& ctx,
                    Stream<gpu>* s);
// SegSoftmaxBackward
void SegSoftmaxBackwardImpl(const Tensor<cpu, 2, float> &dst,
                            const Tensor<cpu, 2, float> &ograd,
                            const Tensor<cpu, 2, float> &val,
                            const Tensor<cpu, 1, int> &indptr,
                            const OpReqType req,
                            const OpContext& ctx,
                            Stream<cpu>* s);
void SegSoftmaxBackwardImpl(const Tensor<gpu, 2, float> &dst,
                            const Tensor<gpu, 2, float> &ograd,
                            const Tensor<gpu, 2, float> &val,
                            const Tensor<gpu, 1, int> &indptr,
                            const OpReqType req,
                            const OpContext& ctx,
                            Stream<gpu>* s);
// SegTakeKCorr
void SegTakeKCorrImpl(const Tensor<cpu, 2, float> &dst,
                      const Tensor<cpu, 3, float> &embed1,
                      const Tensor<cpu, 3, float> &embed2,
                      const Tensor<cpu, 1, int> &neighbor_ids,
                      const Tensor<cpu, 1, int> &neighbor_indptr,
                      const OpReqType req,
                      const OpContext& ctx,
                      Stream<cpu>* s);
void SegTakeKCorrImpl(const Tensor<gpu, 2, float> &dst,
                      const Tensor<gpu, 3, float> &embed1,
                      const Tensor<gpu, 3, float> &embed2,
                      const Tensor<gpu, 1, int> &neighbor_ids,
                      const Tensor<gpu, 1, int> &neighbor_indptr,
                      const OpReqType req,
                      const OpContext& ctx,
                      Stream<gpu>* s);
// SegTakeKCorrBackwardEmbed1
void SegTakeKCorrBackwardEmbed1Impl(const Tensor<cpu, 3, float> &dst,
                                    const Tensor<cpu, 2, float> &ograd,
                                    const Tensor<cpu, 3, float> &embed2,
                                    const Tensor<cpu, 1, int> &neighbor_ids,
                                    const Tensor<cpu, 1, int> &neighbor_indptr,
                                    const OpReqType req,
                                    const OpContext& ctx,
                                    Stream<cpu>* s);
void SegTakeKCorrBackwardEmbed1Impl(const Tensor<gpu, 3, float> &dst,
                                    const Tensor<gpu, 2, float> &ograd,
                                    const Tensor<gpu, 3, float> &embed2,
                                    const Tensor<gpu, 1, int> &neighbor_ids,
                                    const Tensor<gpu, 1, int> &neighbor_indptr,
                                    const OpReqType req,
                                    const OpContext& ctx,
                                    Stream<gpu>* s);
// SegTakeKCorrBackwardEmbed2
void SegTakeKCorrBackwardEmbed2Impl(const Tensor<cpu, 3, float> &dst,
                                    const Tensor<cpu, 2, float> &ograd,
                                    const Tensor<cpu, 3, float> &embed1,
                                    const Tensor<cpu, 1, int> &neighbor_ids,
                                    const Tensor<cpu, 1, int> &neighbor_indptr,
                                    const OpReqType req,
                                    const OpContext& ctx,
                                    Stream<cpu>* s);
void SegTakeKCorrBackwardEmbed2Impl(const Tensor<gpu, 3, float> &dst,
                                    const Tensor<gpu, 2, float> &ograd,
                                    const Tensor<gpu, 3, float> &embed1,
                                    const Tensor<gpu, 1, int> &neighbor_ids,
                                    const Tensor<gpu, 1, int> &neighbor_indptr,
                                    const OpReqType req,
                                    const OpContext& ctx,
                                    Stream<gpu>* s);
// SegPool
template<int pool_type>
void SegPoolImpl(const Tensor<cpu, 3, float> &dst_value,
                 const Tensor<cpu, 3, int> &pool_indices,
                 const Tensor<cpu, 3, float> &data,
                 const Tensor<cpu, 1, int> &indices,
                 const Tensor<cpu, 1, int> &indptr,
                 const OpReqType req,
                 const OpContext &ctx,
                 Stream<cpu>* s);
template<int pool_type>
void SegPoolImpl(const Tensor<gpu, 3, float> &dst_value,
                 const Tensor<gpu, 3, int> &pool_indices,
                 const Tensor<gpu, 3, float> &data,
                 const Tensor<gpu, 1, int> &indices,
                 const Tensor<gpu, 1, int> &indptr,
                 const OpReqType req,
                 const OpContext &ctx,
                 Stream<gpu>* s);
// SegPoolBackwardImpl
template<int pool_type>
void SegPoolBackwardImpl(const Tensor<cpu, 3, float> &dst,
                         const Tensor<cpu, 3, float> &ograd,
                         const Tensor<cpu, 3, int> &out_index,
                         const Tensor<cpu, 1, int> &indices,
                         const Tensor<cpu, 1, int> &indptr,
                         const OpReqType req,
                         const OpContext &ctx,
                         Stream<cpu>* s);
template<int pool_type>
void SegPoolBackwardImpl(const Tensor<gpu, 3, float> &dst,
                         const Tensor<gpu, 3, float> &ograd,
                         const Tensor<gpu, 3, int> &out_index,
                         const Tensor<gpu, 1, int> &indices,
                         const Tensor<gpu, 1, int> &indptr,
                         const OpReqType req,
                         const OpContext &ctx,
                         Stream<gpu>* s);
}  // namespace seg_op

struct NNZOnlyParam: public dmlc::Parameter<NNZOnlyParam> {
  int nnz;
  DMLC_DECLARE_PARAMETER(NNZOnlyParam) {
    DMLC_DECLARE_FIELD(nnz).set_lower_bound(0)
    .describe("The nnz value.");
  }
};

// TODO(sxjscience) Support different Corrtype
struct SegTakeKCorrParam: public dmlc::Parameter<SegTakeKCorrParam> {
  int corr_type;
  DMLC_DECLARE_PARAMETER(SegTakeKCorrParam) {
    DMLC_DECLARE_FIELD(corr_type)
      .add_enum("inner_product", seg_op::SegTakeKCorrType::kInnerProduct)
      .add_enum("euclidean", seg_op::SegTakeKCorrType::kEuclidean)
      .set_default(seg_op::SegTakeKCorrType::kInnerProduct)
      .describe("The nnz value.");
  }
};

struct SegPoolParam: public dmlc::Parameter<SegPoolParam> {
  int pool_type;
  DMLC_DECLARE_PARAMETER(SegPoolParam) {
    DMLC_DECLARE_FIELD(pool_type)
      .add_enum("avg", seg_op::SegReduceType::kMean)
      .add_enum("sum", seg_op::SegReduceType::kSum)
      .add_enum("max", seg_op::SegReduceType::kMax)
      .describe("The pooling type.");
  }
};

inline bool SegReduceShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector *in_attrs,
                           mxnet::ShapeVector *out_attrs) {
  using namespace mshadow;
  const TShape &data_shape = in_attrs->at(0);
  const TShape &indptr_shape = in_attrs->at(1);
  if (data_shape.ndim() !=  2) return false;
  if (indptr_shape.ndim() != 1) return false;
  int batch_size = data_shape[0];
  int seg_num = indptr_shape[0] - 1;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, Shape2(batch_size, seg_num));
  return true;
}

inline bool SegReduceSetType(const nnvm::NodeAttrs& attrs,
                             std::vector<int> *in_type,
                             std::vector<int> *out_type) {
  CHECK_EQ(in_type->size(), 2U);
  CHECK_EQ(out_type->size(), 1U);
  int data_type = (*in_type)[0];
  int indptr_type = (*in_type)[1];
  CHECK_EQ(data_type, mshadow::kFloat32) << "Only Float32 type is supported for data! Recieved " << data_type;
  CHECK_EQ(indptr_type, mshadow::kInt32) << "Only int32 type is supported for indptr! Recieved " << indptr_type;
  out_type->clear();
  out_type->push_back(data_type);
  return true;
}

template<typename xpu, int reduce_type>
void SegReduceForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 2, float> data = inputs[0].get<xpu, 2, float>();
  Tensor<xpu, 1, int> indptr = inputs[1].get<xpu, 1, int>();
  Tensor<xpu, 2, float> dst = outputs[0].get<xpu, 2, float>();
  seg_op::SegReduceImpl<reduce_type>(dst, data, indptr, req[0], ctx, s);
}

inline bool SegBroadcastBinaryShape(const nnvm::NodeAttrs& attrs,
                                    mxnet::ShapeVector *in_attrs,
                                    mxnet::ShapeVector *out_attrs) {
  using namespace mshadow;
  const TShape &lhs_shape = in_attrs->at(0);
  const TShape &rhs_shape = in_attrs->at(1);
  const TShape &indptr_shape = in_attrs->at(2);
  CHECK_EQ(lhs_shape[0], rhs_shape[0]);
  CHECK_EQ(rhs_shape[1] + 1, indptr_shape[0]);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, lhs_shape);
  return true;
}

inline bool SegBroadcastBinarySetType(const nnvm::NodeAttrs& attrs,
                                      std::vector<int> *in_type,
                                      std::vector<int> *out_type) {
  CHECK_EQ(in_type->size(), 3U);
  CHECK_EQ(out_type->size(), 1U);
  CHECK_EQ((*in_type)[0], mshadow::kFloat32);
  CHECK_EQ((*in_type)[1], mshadow::kFloat32);
  CHECK_EQ((*in_type)[2], mshadow::kInt32);
  (*out_type)[0] = mshadow::kFloat32;
  return true;
}

template<typename xpu, typename OP>
void SegBroadcastBinaryForward(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 2, float> lhs = inputs[0].get<xpu, 2, float>();
  Tensor<xpu, 2, float> rhs = inputs[1].get<xpu, 2, float>();
  Tensor<xpu, 1, int> indptr = inputs[2].get<xpu, 1, int>();
  Tensor<xpu, 2, float> dst = outputs[0].get<xpu, 2, float>();
  seg_op::SegBroadcastBinaryImpl<OP>(dst, lhs, rhs, indptr, req[0], ctx, s);
}

inline bool SegBroadcastToShape(const nnvm::NodeAttrs& attrs,
                                mxnet::ShapeVector *in_attrs,
                                mxnet::ShapeVector *out_attrs) {
  using namespace mshadow;
  const TShape &data_shape = in_attrs->at(0);
  const TShape &indptr_shape = in_attrs->at(1);
  CHECK_EQ(data_shape.ndim(), 2);  
  CHECK_EQ(data_shape[1] + 1, indptr_shape[0]);
  const NNZOnlyParam& param = nnvm::get<NNZOnlyParam>(attrs.parsed);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, Shape2(data_shape[0], param.nnz));
  return true;
}

inline bool SegBroadcastToSetType(const nnvm::NodeAttrs& attrs,
                                  std::vector<int> *in_type,
                                  std::vector<int> *out_type) {
  CHECK_EQ(in_type->size(), 2U);
  CHECK_EQ(out_type->size(), 1U);
  CHECK_EQ((*in_type)[0], mshadow::kFloat32);
  CHECK_EQ((*in_type)[1], mshadow::kInt32);
  (*out_type)[0] = mshadow::kFloat32;
  return true;
}

template<typename xpu>
void SegBroadcastToForward(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 2, float> data = inputs[0].get<xpu, 2, float>();
  Tensor<xpu, 1, int> indptr = inputs[1].get<xpu, 1, int>();
  Tensor<xpu, 2, float> dst = outputs[0].get<xpu, 2, float>();
  seg_op::SegBroadcastBinaryImpl<mshadow::op::right>(dst, dst, data, indptr, req[0], ctx, s);
}

inline bool SegSoftmaxShape(const nnvm::NodeAttrs& attrs,
                            mxnet::ShapeVector *in_attrs,
                            mxnet::ShapeVector *out_attrs) {
  using namespace mshadow;
  const TShape &data_shape = in_attrs->at(0);
  const TShape &indptr_shape = in_attrs->at(1);
  if (data_shape.ndim() != 2) return false;
  if (indptr_shape.ndim() != 1) return false;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, data_shape);
  return true;
}

template<typename xpu>
void SegSoftmaxForward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 2, float> data = inputs[0].get<xpu, 2, float>();
  Tensor<xpu, 1, int> indptr = inputs[1].get<xpu, 1, int>();
  Tensor<xpu, 2, float> dst = outputs[0].get<xpu, 2, float>();
  seg_op::SegSoftmaxImpl(dst, data, indptr, req[0], ctx, s);
}

template<typename xpu>
void SegSoftmaxBackward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 2, float> ograd = inputs[0].get<xpu, 2, float>();
  Tensor<xpu, 2, float> val = inputs[1].get<xpu, 2, float>();
  Tensor<xpu, 1, int> indptr = inputs[2].get<xpu, 1, int>();
  Tensor<xpu, 2, float> dst = outputs[0].get<xpu, 2, float>();
  seg_op::SegSoftmaxBackwardImpl(dst, ograd, val, indptr, req[0], ctx, s);
}

inline bool SegTakeKCorrShape(const nnvm::NodeAttrs& attrs,
                              mxnet::ShapeVector *in_attrs,
                              mxnet::ShapeVector *out_attrs) {
  using namespace mshadow;
  const TShape &embed1_shape = in_attrs->at(0);
  const TShape &embed2_shape = in_attrs->at(1);
  const TShape &neighbor_ids_shape = in_attrs->at(2);
  const TShape &neighbor_indptr_shape = in_attrs->at(3);
  CHECK_EQ(embed1_shape.ndim(), 3);
  CHECK_EQ(embed2_shape.ndim(), 3);
  CHECK_EQ(neighbor_ids_shape.ndim(), 1);
  CHECK_EQ(neighbor_indptr_shape.ndim(), 1);
  int K = embed1_shape[0];
  int node_num = embed1_shape[1];
  int feat_dim = embed1_shape[2];
  int nnz = neighbor_ids_shape[0];
  CHECK_EQ(embed2_shape[0], K);
  CHECK_EQ(embed2_shape[2], feat_dim);
  CHECK_EQ(neighbor_indptr_shape[0], node_num + 1);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, Shape2(K, nnz));
  return true;
}

inline bool SegTakeKCorrSetType(const nnvm::NodeAttrs& attrs,
                                std::vector<int> *in_type,
                                std::vector<int> *out_type) {
  CHECK_EQ(in_type->size(), 4U);
  CHECK_EQ(out_type->size(), 1U);
  CHECK_EQ((*in_type)[0], mshadow::kFloat32) << "Only Float32 type is supported for embed1! Recieved " << (*in_type)[0];
  CHECK_EQ((*in_type)[1], mshadow::kFloat32) << "Only Float32 type is supported for embed2! Recieved " << (*in_type)[1];
  CHECK_EQ((*in_type)[2], mshadow::kInt32) << "Only Int32 type is supported for neighbor_ids! Recieved " << (*in_type)[2];
  CHECK_EQ((*in_type)[3], mshadow::kInt32) << "Only Int32 type is supported for neighbor_indptr! Recieved " << (*in_type)[3];
  out_type->clear();
  out_type->push_back(mshadow::kFloat32);
  return true;
}

template<typename xpu>
void SegTakeKCorrForward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 4U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 3, float> embed1 = inputs[0].get<xpu, 3, float>();
  Tensor<xpu, 3, float> embed2 = inputs[1].get<xpu, 3, float>();
  Tensor<xpu, 1, int> neighbor_ids = inputs[2].get<xpu, 1, int>();
  Tensor<xpu, 1, int> neighbor_indptr = inputs[3].get<xpu, 1, int>();
  Tensor<xpu, 2, float> dst = outputs[0].get<xpu, 2, float>();
  seg_op::SegTakeKCorrImpl(dst, embed1, embed2, neighbor_ids, neighbor_indptr, req[0], ctx, s);
}

inline bool SegWeightedPoolShape(const nnvm::NodeAttrs& attrs,
                                 mxnet::ShapeVector *in_attrs,
                                 mxnet::ShapeVector *out_attrs) {
  using namespace mshadow;
  const TShape &data_shape = in_attrs->at(0);
  const TShape &weights_shape = in_attrs->at(1);
  const TShape &indices_shape = in_attrs->at(2);
  const TShape &indptr_shape = in_attrs->at(3);
  if (data_shape.ndim() !=  3) return false;
  if (weights_shape.ndim() != 2) return false;
  if (indices_shape.ndim() != 1) return false;
  if (indptr_shape.ndim() != 1) return false;
  int batch_size = data_shape[0];
  int total_ind_num = data_shape[1];
  int feat_dim = data_shape[2];
  int nnz = indices_shape[0];
  int seg_num = indptr_shape[0] - 1;
  CHECK_EQ(weights_shape[0], batch_size);
  CHECK_EQ(weights_shape[1], nnz);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, Shape3(batch_size, seg_num, feat_dim));
  return true;
}

template<typename xpu>
void SegWeightedPoolForward(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 4U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 3, float> data = inputs[0].get<xpu, 3, float>();
  Tensor<xpu, 2, float> weights = inputs[1].get<xpu, 2, float>();
  Tensor<xpu, 1, int> indices = inputs[2].get<xpu, 1, int>();
  Tensor<xpu, 1, int> indptr = inputs[3].get<xpu, 1, int>();
  Tensor<xpu, 3, float> dst = outputs[0].get<xpu, 3, float>();
  seg_op::SegTakeKCorrBackwardEmbed1Impl(dst, weights, data, indices, indptr, req[0], ctx, s);
}

template<typename xpu>
void SegTakeKCorrBackwardEmbed2(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 4U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 2, float> ograd = inputs[0].get<xpu, 2, float>();
  Tensor<xpu, 3, float> embed1 = inputs[1].get<xpu, 3, float>();
  Tensor<xpu, 1, int> neighbor_ids = inputs[2].get<xpu, 1, int>();
  Tensor<xpu, 1, int> neighbor_indptr = inputs[3].get<xpu, 1, int>();
  Tensor<xpu, 3, float> dst = outputs[0].get<xpu, 3, float>();
  seg_op::SegTakeKCorrBackwardEmbed2Impl(dst, ograd, embed1, neighbor_ids, neighbor_indptr, req[0], ctx, s);
}

inline bool SegPoolShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector *in_attrs,
                         mxnet::ShapeVector *out_attrs) {
  using namespace mshadow;
  const SegPoolParam& param = nnvm::get<SegPoolParam>(attrs.parsed);
  const TShape &data_shape = in_attrs->at(0);
  const TShape &indices_shape = in_attrs->at(1);
  const TShape &indptr_shape = in_attrs->at(2);
  if (data_shape.ndim() != 3) return false;
  if (indices_shape.ndim() != 1) return false;
  if (indptr_shape.ndim() != 1) return false;
  int batch_size = data_shape[0];
  int total_ind_num = data_shape[1];
  int feat_dim = data_shape[2];
  int nnz = indices_shape[0];
  int seg_num = indptr_shape[0] - 1;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, Shape3(batch_size, seg_num, feat_dim));
  if(param.pool_type == seg_op::SegReduceType::kMax) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, Shape3(batch_size, seg_num, feat_dim));
  }
  return true;
}

inline bool SegPoolSetType(const nnvm::NodeAttrs& attrs,
                           std::vector<int> *in_type,
                           std::vector<int> *out_type) {
  const SegPoolParam& param = nnvm::get<SegPoolParam>(attrs.parsed);
  CHECK_EQ(in_type->size(), 3U);
  if(param.pool_type == seg_op::SegReduceType::kMax) {
    CHECK_EQ(out_type->size(), 2U);
  } else {
    CHECK_EQ(out_type->size(), 1U);
  }
  
  CHECK_EQ((*in_type)[0], mshadow::kFloat32) << "Only Float32 type is supported for data! Recieved " << (*in_type)[0];
  CHECK_EQ((*in_type)[1], mshadow::kInt32) << "Only Int32 type is supported for indices! Recieved " << (*in_type)[1];
  CHECK_EQ((*in_type)[2], mshadow::kInt32) << "Only Int32 type is supported for indptr! Recieved " << (*in_type)[2];
  out_type->clear();
  out_type->push_back(mshadow::kFloat32);
  if(param.pool_type == seg_op::SegReduceType::kMax) {
    out_type->push_back(mshadow::kInt32);
  }
  return true;
}


template<typename xpu>
void SegPoolForward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  const SegPoolParam& param = nnvm::get<SegPoolParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 3U);
  if(param.pool_type == seg_op::SegReduceType::kMax) {
    CHECK_EQ(outputs.size(), 2U);
  } else {
    CHECK_EQ(outputs.size(), 1U);
  }
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 3, float> data = inputs[0].get<xpu, 3, float>();
  Tensor<xpu, 1, int> indices = inputs[1].get<xpu, 1, int>();
  Tensor<xpu, 1, int> indptr = inputs[2].get<xpu, 1, int>();
  Tensor<xpu, 3, float> dst = outputs[0].get<xpu, 3, float>();
  Tensor<xpu, 3, int> pool_indices;
  if(param.pool_type == seg_op::SegReduceType::kMax) {
    pool_indices = outputs[1].get<xpu, 3, int>();
    seg_op::SegPoolImpl<seg_op::SegReduceType::kMax>(dst, pool_indices, data, indices, indptr, req[0], ctx, s);
  } else if(param.pool_type == seg_op::SegReduceType::kMean) {
    seg_op::SegPoolImpl<seg_op::SegReduceType::kMean>(dst, pool_indices, data, indices, indptr, req[0], ctx, s);
  } else if(param.pool_type == seg_op::SegReduceType::kSum) {
    seg_op::SegPoolImpl<seg_op::SegReduceType::kSum>(dst, pool_indices, data, indices, indptr, req[0], ctx, s);
  } else {
    LOG(FATAL) << "Not supported";
  }
  
}

template<typename xpu>
void SegSumMeanPoolBackward(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  const SegPoolParam& param = nnvm::get<SegPoolParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 3, float> ograd = inputs[0].get<xpu, 3, float>();
  Tensor<xpu, 3, int> out_index;
  Tensor<xpu, 1, int> indices = inputs[1].get<xpu, 1, int>();
  Tensor<xpu, 1, int> indptr = inputs[2].get<xpu, 1, int>();
  Tensor<xpu, 3, float> dst = outputs[0].get<xpu, 3, float>();
  if(param.pool_type == seg_op::SegReduceType::kMean) {
    seg_op::SegPoolBackwardImpl<seg_op::SegReduceType::kMean>(
      dst, ograd, out_index, indices, indptr, req[0], ctx, s);
  } else if (param.pool_type == seg_op::SegReduceType::kSum) {
    seg_op::SegPoolBackwardImpl<seg_op::SegReduceType::kSum>(
      dst, ograd, out_index, indices, indptr, req[0], ctx, s);
  } else {
    LOG(FATAL) << "Only support mean and sum in the backward";
  }
}

template<typename xpu>
void SegMaxPoolBackward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 4U);
  CHECK_EQ(outputs.size(), 1U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 3, float> ograd = inputs[0].get<xpu, 3, float>();
  Tensor<xpu, 3, int> out_index = inputs[1].get<xpu, 3, int>();
  Tensor<xpu, 1, int> indices = inputs[2].get<xpu, 1, int>();
  Tensor<xpu, 1, int> indptr = inputs[3].get<xpu, 1, int>();
  Tensor<xpu, 3, float> dst = outputs[0].get<xpu, 3, float>();
  seg_op::SegPoolBackwardImpl<seg_op::SegReduceType::kMax>(
    dst, ograd, out_index, indices, indptr, req[0], ctx, s);
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SEG_OP_H_
