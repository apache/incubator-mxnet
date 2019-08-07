/*!
 * Copyright (c) 2015 by Contributors
 * \file spatial_upsampling.h
 * \brief
 * \author Bing Xu
*/
#ifndef MSHADOW_EXTENSION_SPATIAL_UPSAMPLING_NEAREST_H_
#define MSHADOW_EXTENSION_SPATIAL_UPSAMPLING_NEAREST_H_
#include "../extension.h"

namespace mshadow {
namespace expr {

/*! \brief nearest neighbor upsampling
 *         out(x, y) = in(int(x / scale_x), int(y / scale_y))
 *  \tparam SrcExp source expression
 *  \tparam DType data type
 *  \tparam srcdim source dimension
 */
template<typename SrcExp, typename DType, int srcdim>
struct UpSamplingNearestExp :
  public MakeTensorExp<UpSamplingNearestExp<SrcExp, DType, srcdim>,
                       SrcExp, srcdim, DType> {
  /*! \brief source oprand */
  const SrcExp &src_;
  /*! \brief up sampling scale */
  index_t scale_h_;
  index_t scale_w_;

  /*! \brief constructor */
  UpSamplingNearestExp(const SrcExp &src, index_t scale_h, index_t scale_w)
    : src_(src), scale_h_(scale_h), scale_w_(scale_w) {
    this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
    this->shape_[srcdim - 2] *= scale_h;
    this->shape_[srcdim - 1] *= scale_w;
  }
};


template<typename SrcExp, typename DType, int etype>
inline UpSamplingNearestExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>
upsampling_nearest(const Exp<SrcExp, DType, etype> &src, index_t scale_h, index_t scale_w) {
  TypeCheckPass<ExpInfo<SrcExp>::kDim >= 2>
    ::Error_Expression_Does_Not_Meet_Dimension_Req();
  return UpSamplingNearestExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>(src.self(), scale_h, scale_w);
}

template<typename SrcExp, typename DType, int srcdim>
struct Plan<UpSamplingNearestExp<SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const UpSamplingNearestExp<SrcExp, DType, srcdim> &e)
    : src_(MakePlan(e.src_)),
      scale_h_(e.scale_h_),
      scale_w_(e.scale_w_),
      new_height_(e.shape_[srcdim - 2]),
      //new_width_(e.shape_[srcdim - 1]),
      src_height_(static_cast<index_t>(e.shape_[srcdim - 2] / e.scale_h_)) {}
      //src_width_(static_cast<index_t>(e.shape_[srcdim - 1] / e.scale_w_)) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    const index_t x = j;
    const index_t y = i % new_height_;
    const index_t c = i / new_height_;
    const index_t h = static_cast<index_t>(y / scale_h_);
    const index_t w = static_cast<index_t>(x / scale_w_);
    return src_.Eval(c * src_height_ + h, w);
  }

 private:
  Plan<SrcExp, DType> src_;
  const index_t scale_h_;
  const index_t scale_w_;
  const index_t new_height_;
  const index_t new_width_;
  const index_t src_height_;
  const index_t src_width_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_SPATIAL_UPSAMPLING_NEAREST_H_
