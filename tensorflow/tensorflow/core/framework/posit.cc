/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/posit.h"

#include "third_party/eigen3/Eigen/Core"

namespace tensorflow {

void RoundFloatToPosit160(const float* src, posit160* dst, int64 size) {
  Eigen::Map<const Eigen::ArrayXf> src_eigen(src, size);
  Eigen::Map<Eigen::Array<posit160, Eigen::Dynamic, 1>> dst_eigen(dst, size);
  dst_eigen = src_eigen.cast<posit160>();
}

void FloatToPosit160(const float* src, posit160* dst, int64 size) {
  for (; size != 0; src++, dst++, size--) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    memcpy(dst, src, sizeof(posit160));
#else
    memcpy(
        dst,
        reinterpret_cast<const char*>(src) + sizeof(float) - sizeof(posit160),
        sizeof(posit160));
#endif
  }
}

void Posit160ToFloat(const posit160* src, float* dst, int64 size) {
  Eigen::Map<const Eigen::Array<posit160, Eigen::Dynamic, 1>> src_eigen(src,
                                                                        size);
  Eigen::Map<Eigen::ArrayXf> dst_eigen(dst, size);
  dst_eigen = src_eigen.cast<float>();
}

}  // end namespace tensorflow
