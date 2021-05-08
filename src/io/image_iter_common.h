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
 *  Copyright (c) 2017 by Contributors
 * \file image_iter_common.h
 * \brief common types used by image data iterators
 */

#ifndef MXNET_IO_IMAGE_ITER_COMMON_H_
#define MXNET_IO_IMAGE_ITER_COMMON_H_

#include <mxnet/io.h>
#include <vector>
#include <unordered_map>
#include <string>

namespace mxnet {
namespace io {
/*! \brief data structure to hold labels for images */
class ImageLabelMap {
 public:
  /*!
   * \brief initialize the label list into memory
   * \param path_imglist path to the image list
   * \param label_width predefined label_width
   */
  explicit ImageLabelMap(const char *path_imglist,
                         index_t label_width,
                         bool silent) {
    this->label_width = label_width;
    image_index_.clear();
    label_.clear();
    idx2label_.clear();
    dmlc::InputSplit *fi = dmlc::InputSplit::Create
        (path_imglist, 0, 1, "text");
    dmlc::InputSplit::Blob rec;
    while (fi->NextRecord(&rec)) {
      // quick manual parsing
      char *p = reinterpret_cast<char*>(rec.dptr);
      char *end = p + rec.size;
      // skip space
      while (isspace(*p) && p != end) ++p;
      image_index_.push_back(static_cast<size_t>(atol(p)));
      for (index_t i = 0; i < label_width; ++i) {
        // skip till space
        while (!isspace(*p) && p != end) ++p;
        // skip space
        while (isspace(*p) && p != end) ++p;
        CHECK(p != end) << "Bad ImageList format";
        label_.push_back(static_cast<real_t>(atof(p)));
      }
    }
    delete fi;
    // be careful not to resize label_ afterwards
    idx2label_.reserve(image_index_.size());
    for (size_t i = 0; i < image_index_.size(); ++i) {
      idx2label_[image_index_[i]] = dmlc::BeginPtr(label_) + i * label_width;
    }
    if (!silent) {
      LOG(INFO) << "Loaded ImageList from " << path_imglist << ' '
                << image_index_.size() << " Image records";
    }
  }
  /*! \brief find a label for corresponding index */
  inline mshadow::Tensor<cpu, 1> Find(size_t imid) const {
    std::unordered_map<size_t, real_t*>::const_iterator it
        = idx2label_.find(imid);
    CHECK(it != idx2label_.end()) << "fail to find imagelabel for id " << imid;
    return mshadow::Tensor<cpu, 1>(it->second, mshadow::Shape1(label_width));
  }
  /*! \brief find a label for corresponding index, return vector as copy */
  inline std::vector<float> FindCopy(size_t imid) const {
    std::unordered_map<size_t, real_t*>::const_iterator it
        = idx2label_.find(imid);
    CHECK(it != idx2label_.end()) << "fail to find imagelabel for id " << imid;
    const real_t *ptr = it->second;
    return std::vector<float>(ptr, ptr + label_width);
  }

 private:
  // label with_
  mshadow::index_t label_width;
  // image index of each record
  std::vector<size_t> image_index_;
  // real label content
  std::vector<real_t> label_;
  // map index to label
  std::unordered_map<size_t, real_t*> idx2label_;
};

// Define image record parser parameters
struct ImageRecParserParam : public dmlc::Parameter<ImageRecParserParam> {
  /*! \brief path to image list */
  std::string path_imglist;
  /*! \brief path to image recordio */
  std::string path_imgrec;
  /*! \brief path to index file */
  std::string path_imgidx;
  /*! \brief a sequence of names of image augmenters, seperated by , */
  std::string aug_seq;
  /*! \brief label-width */
  int label_width;
  /*! \brief input shape */
  mxnet::TShape data_shape;
  /*! \brief number of threads */
  int preprocess_threads;
  /*! \brief whether to remain silent */
  bool verbose;
  /*! \brief partition the data into multiple parts */
  int num_parts;
  /*! \brief the index of the part will read */
  int part_index;
  /*! \brief device id used to create context for internal NDArray */
  int device_id;
  /*! \brief the size of a shuffle chunk */
  size_t shuffle_chunk_size;
  /*! \brief the seed for chunk shuffling */
  int shuffle_chunk_seed;
  /*! \brief random seed for augmentations */
  dmlc::optional<int> seed_aug;

  // declare parameters
  DMLC_DECLARE_PARAMETER(ImageRecParserParam) {
    DMLC_DECLARE_FIELD(path_imglist).set_default("")
        .describe("Path to the image list (.lst) file. Generally created with tools/im2rec.py. "\
                  "Format (Tab separated): "\
                  "<index of record>\t<one or more labels>\t<relative path from root folder>.");
    DMLC_DECLARE_FIELD(path_imgrec).set_default("")
        .describe("Path to the image RecordIO (.rec) file or a directory path. "\
                  "Created with tools/im2rec.py.");
    DMLC_DECLARE_FIELD(path_imgidx).set_default("")
        .describe("Path to the image RecordIO index (.idx) file. "\
                  "Created with tools/im2rec.py.");
    DMLC_DECLARE_FIELD(aug_seq).set_default("aug_default")
        .describe("The augmenter names to represent"\
                  " sequence of augmenters to be applied, seperated by comma." \
                  " Additional keyword parameters will be seen by these augmenters.");
    DMLC_DECLARE_FIELD(label_width).set_lower_bound(1).set_default(1)
        .describe("The number of labels per image.");
    DMLC_DECLARE_FIELD(data_shape)
        .set_expect_ndim(3).enforce_nonzero()
        .describe("The shape of one output image in (channels, height, width) format.");
    DMLC_DECLARE_FIELD(preprocess_threads).set_lower_bound(1).set_default(4)
        .describe("The number of threads to do preprocessing.");
    DMLC_DECLARE_FIELD(verbose).set_default(true)
        .describe("If or not output verbose information.");
    DMLC_DECLARE_FIELD(num_parts).set_default(1)
        .describe("Virtually partition the data into these many parts.");
    DMLC_DECLARE_FIELD(part_index).set_default(0)
        .describe("The *i*-th virtual partition to be read.");
    DMLC_DECLARE_FIELD(device_id).set_default(0)
        .describe("The device id used to create context for internal NDArray. "\
                  "Setting device_id to -1 will create Context::CPU(0). Setting "
                  "device_id to valid positive device id will create "
                  "Context::CPUPinned(device_id). Default is 0.");
    DMLC_DECLARE_FIELD(shuffle_chunk_size).set_default(0)
        .describe("The data shuffle buffer size in MB. Only valid if shuffle is true.");
    DMLC_DECLARE_FIELD(shuffle_chunk_seed).set_default(0)
        .describe("The random seed for shuffling");
    DMLC_DECLARE_FIELD(seed_aug).set_default(dmlc::optional<int>())
        .describe("Random seed for augmentations.");
  }
};

// Batch parameters
struct BatchParam : public dmlc::Parameter<BatchParam> {
  /*! \brief label width */
  uint32_t batch_size;
  /*! \brief use round roubin to handle overflow batch */
  bool round_batch;
  // declare parameters
  DMLC_DECLARE_PARAMETER(BatchParam) {
    DMLC_DECLARE_FIELD(batch_size)
        .describe("Batch size.");
    DMLC_DECLARE_FIELD(round_batch).set_default(true)
        .describe("Whether to use round robin to handle overflow batch or not.");
  }
};

// Batch Sampler parameters
struct BatchSamplerParam : public dmlc::Parameter<BatchSamplerParam> {
  /*! \brief Last batch behavior type */
  enum LastBatchType {
    /*! \brief Keep not fully filled last batch */
    kKeep = 0,
    /*! \brief Roll over the remaining batch to next epoch */
    kRollOver,
    /*! \brief Discard not fully filled last batch */
    kDiscard
  };  // enum LastBatchType
  /*! \brief batch size */
  uint32_t batch_size;
  /*! \brief last batch behavior */
  int last_batch;
  // declare parameters
  DMLC_DECLARE_PARAMETER(BatchSamplerParam) {
    DMLC_DECLARE_FIELD(batch_size)
        .describe("Batch size.");
    DMLC_DECLARE_FIELD(last_batch).set_default(kKeep)
        .add_enum("keep", kKeep)
        .add_enum("rollover", kRollOver)
        .add_enum("discard", kDiscard)
        .describe("Specifies how the last batch is handled if batch_size does not evenly "
                  "divide sequence length. "
                  "If 'keep', the last batch will be returned directly, but will contain "
                  "less element than `batch_size` requires. "
                  "If 'discard', the last batch will be discarded. "
                  "If 'rollover', the remaining elements will be rolled over to the next "
                  "iteration. Note: legacy batch param with round_batch will always round data "
                  "in order to always provide full batchs. Rollover behavior will instead result "
                  "in different iteration sizes for each epoch.");
  }
};

// Define image record parameters
struct ImageRecordParam: public dmlc::Parameter<ImageRecordParam> {
  /*! \brief whether to do shuffle */
  bool shuffle;
  /*! \brief random seed */
  int seed;
  /*! \brief whether to remain silent */
  bool verbose;
  // declare parameters
  DMLC_DECLARE_PARAMETER(ImageRecordParam) {
    DMLC_DECLARE_FIELD(shuffle).set_default(false)
        .describe("Whether to shuffle data randomly or not.");
    DMLC_DECLARE_FIELD(seed).set_default(0)
        .describe("The random seed.");
    DMLC_DECLARE_FIELD(verbose).set_default(true)
        .describe("Whether to output verbose information or not.");
  }
};

// normalize parameters
struct ImageNormalizeParam :  public dmlc::Parameter<ImageNormalizeParam> {
  /*! \brief random seed */
  int seed;
  /*! \brief whether to mirror the image */
  bool mirror;
  /*! \brief whether to perform rand mirror the image */
  bool rand_mirror;
  /*! \brief mean file string */
  std::string mean_img;
  /*! \brief mean value for r channel */
  float mean_r;
  /*! \brief mean value for g channel */
  float mean_g;
  /*! \brief mean value for b channel */
  float mean_b;
  /*! \brief mean value for alpha channel */
  float mean_a;
  /*! \brief standard deviation for r channel */
  float std_r;
  /*! \brief standard deviation for g channel */
  float std_g;
  /*! \brief standard deviation for b channel */
  float std_b;
  /*! \brief standard deviation for alpha channel */
  float std_a;
  /*! \brief scale on color space */
  float scale;
  /*! \brief maximum ratio of contrast variation */
  float max_random_contrast;
  /*! \brief maximum value of illumination variation */
  float max_random_illumination;
  /*! \brief silent */
  bool verbose;
  // declare parameters
  DMLC_DECLARE_PARAMETER(ImageNormalizeParam) {
    DMLC_DECLARE_FIELD(seed).set_default(0)
        .describe("The random seed.");
    DMLC_DECLARE_FIELD(mirror).set_default(false)
        .describe("Whether to mirror the image or not. If true, images are "\
                  "flipped along the horizontal axis.");
    DMLC_DECLARE_FIELD(rand_mirror).set_default(false)
        .describe("Whether to randomly mirror images or not. If true, 50% of "\
                  "the images will be randomly mirrored (flipped along the "\
                  "horizontal axis)");
    DMLC_DECLARE_FIELD(mean_img).set_default("")
        .describe("Filename of the mean image.");
    DMLC_DECLARE_FIELD(mean_r).set_default(0.0f)
        .describe("The mean value to be subtracted on the R channel");
    DMLC_DECLARE_FIELD(mean_g).set_default(0.0f)
        .describe("The mean value to be subtracted on the G channel");
    DMLC_DECLARE_FIELD(mean_b).set_default(0.0f)
        .describe("The mean value to be subtracted on the B channel");
    DMLC_DECLARE_FIELD(mean_a).set_default(0.0f)
        .describe("The mean value to be subtracted on the alpha channel");
    DMLC_DECLARE_FIELD(std_r).set_default(1.0f)
        .describe("Augmentation Param: Standard deviation on R channel.");
    DMLC_DECLARE_FIELD(std_g).set_default(1.0f)
        .describe("Augmentation Param: Standard deviation on G channel.");
    DMLC_DECLARE_FIELD(std_b).set_default(1.0f)
        .describe("Augmentation Param: Standard deviation on B channel.");
    DMLC_DECLARE_FIELD(std_a).set_default(1.0f)
        .describe("Augmentation Param: Standard deviation on Alpha channel.");
    DMLC_DECLARE_FIELD(scale).set_default(1.0f)
        .describe("Multiply the image with a scale value.");
    DMLC_DECLARE_FIELD(max_random_contrast).set_default(0.0f)
        .describe("Change the contrast with a value randomly chosen from "
                  "``[-max_random_contrast, max_random_contrast]``");
    DMLC_DECLARE_FIELD(max_random_illumination).set_default(0.0f)
        .describe("Change the illumination with a value randomly chosen from "
                  "``[-max_random_illumination, max_random_illumination]``");
    DMLC_DECLARE_FIELD(verbose).set_default(true)
        .describe("If or not output verbose information.");
  }
};

// normalize det parameters
struct ImageDetNormalizeParam :  public dmlc::Parameter<ImageDetNormalizeParam> {
  /*! \brief random seed */
  int seed;
  /*! \brief mean file string */
  std::string mean_img;
  /*! \brief mean value for r channel */
  float mean_r;
  /*! \brief mean value for g channel */
  float mean_g;
  /*! \brief mean value for b channel */
  float mean_b;
  /*! \brief mean value for alpha channel */
  float mean_a;
  /*! \brief standard deviation for r channel */
  float std_r;
  /*! \brief standard deviation for g channel */
  float std_g;
  /*! \brief standard deviation for b channel */
  float std_b;
  /*! \brief standard deviation for alpha channel */
  float std_a;
  /*! \brief scale on color space */
  float scale;
  /*! \brief silent */
  bool verbose;
  // declare parameters
  DMLC_DECLARE_PARAMETER(ImageDetNormalizeParam) {
    DMLC_DECLARE_FIELD(seed).set_default(0)
        .describe("Augmentation Param: Random Seed.");
    DMLC_DECLARE_FIELD(mean_img).set_default("")
        .describe("Augmentation Param: Mean Image to be subtracted.");
    DMLC_DECLARE_FIELD(mean_r).set_default(0.0f)
        .describe("Augmentation Param: Mean value on R channel.");
    DMLC_DECLARE_FIELD(mean_g).set_default(0.0f)
        .describe("Augmentation Param: Mean value on G channel.");
    DMLC_DECLARE_FIELD(mean_b).set_default(0.0f)
        .describe("Augmentation Param: Mean value on B channel.");
    DMLC_DECLARE_FIELD(mean_a).set_default(0.0f)
        .describe("Augmentation Param: Mean value on Alpha channel.");
    DMLC_DECLARE_FIELD(std_r).set_default(0.0f)
        .describe("Augmentation Param: Standard deviation on R channel.");
    DMLC_DECLARE_FIELD(std_g).set_default(0.0f)
        .describe("Augmentation Param: Standard deviation on G channel.");
    DMLC_DECLARE_FIELD(std_b).set_default(0.0f)
        .describe("Augmentation Param: Standard deviation on B channel.");
    DMLC_DECLARE_FIELD(std_a).set_default(0.0f)
        .describe("Augmentation Param: Standard deviation on Alpha channel.");
    DMLC_DECLARE_FIELD(scale).set_default(1.0f)
        .describe("Augmentation Param: Scale in color space.");
    DMLC_DECLARE_FIELD(verbose).set_default(true)
        .describe("Augmentation Param: Whether to print augmentor info.");
  }
};

// Define prefetcher parameters
struct PrefetcherParam : public dmlc::Parameter<PrefetcherParam> {
  enum CtxType { kGPU = 0, kCPU, kCPUPinned, kCPUShared};
  /*! \brief number of prefetched batches */
  size_t prefetch_buffer;

  /*! \brief Context data loader optimized for */
  int ctx;
  int device_id;
  /*! \brief data type */
  dmlc::optional<int> dtype;

  // declare parameters
  DMLC_DECLARE_PARAMETER(PrefetcherParam) {
    DMLC_DECLARE_FIELD(prefetch_buffer).set_default(4)
        .describe("Maximum number of batches to prefetch.");
    DMLC_DECLARE_FIELD(ctx).set_default(kGPU)
        .add_enum("cpu", kCPU)
        .add_enum("gpu", kGPU)
        .add_enum("cpu_pinned", kCPUPinned)
        .describe("Context data loader optimized for. "
                  "Note that it only indicates the optimization strategy for devices, "
                  "by no means the prefetcher will load data to GPUs. "
                  "If ctx is 'cpu_pinned' and device_id is not -1, "
                  "it will use cpu_pinned(device_id) as ctx");
    DMLC_DECLARE_FIELD(device_id).set_default(-1)
        .describe("The default device id for context. -1 indicate it's on default device");
    DMLC_DECLARE_FIELD(dtype)
      .add_enum("float32", mshadow::kFloat32)
      .add_enum("float64", mshadow::kFloat64)
      .add_enum("float16", mshadow::kFloat16)
      .add_enum("bfloat16", mshadow::kBfloat16)
      .add_enum("int64", mshadow::kInt64)
      .add_enum("int32", mshadow::kInt32)
      .add_enum("uint16", mshadow::kUint16)
      .add_enum("int16", mshadow::kInt16)
      .add_enum("uint8", mshadow::kUint8)
      .add_enum("int8", mshadow::kInt8)
      .set_default(dmlc::optional<int>())
      .describe("Output data type. ``None`` means no change.");
  }
};

}  // namespace io
}  // namespace mxnet

#endif  // MXNET_IO_IMAGE_ITER_COMMON_H_
