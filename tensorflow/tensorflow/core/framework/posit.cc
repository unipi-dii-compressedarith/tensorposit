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

void RoundFloatToPosit16e2(const float* src, posit16e2* dst, int64 size) {
  Eigen::Map<const Eigen::ArrayXf> src_eigen(src, size);
  Eigen::Map<Eigen::Array<posit16e2, Eigen::Dynamic, 1>> dst_eigen(dst, size);
  dst_eigen = src_eigen.cast<posit16e2>();
}

void FloatToPosit16e2(const float* src, posit16e2* dst, int64 size) {
  for (; size != 0; src++, dst++, size--) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    memcpy(dst, src, sizeof(posit16e2));
#else
    memcpy(
        dst,
        reinterpret_cast<const char*>(src) + sizeof(float) - sizeof(posit16e2),
        sizeof(posit16e2));
#endif
  }
}

void Posit16e2ToFloat(const posit16e2* src, float* dst, int64 size) {
  Eigen::Map<const Eigen::Array<posit16e2, Eigen::Dynamic, 1>> src_eigen(src,
                                                                        size);
  Eigen::Map<Eigen::ArrayXf> dst_eigen(dst, size);
  dst_eigen = src_eigen.cast<float>();
}


// =======================

void RoundFloatToPosit8e2(const float* src, posit8e2* dst, int64 size) {
  Eigen::Map<const Eigen::ArrayXf> src_eigen(src, size);
  Eigen::Map<Eigen::Array<posit8e2, Eigen::Dynamic, 1>> dst_eigen(dst, size);
  dst_eigen = src_eigen.cast<posit8e2>();
}

void FloatToPosit8e2(const float* src, posit8e2* dst, int64 size) {
  for (; size != 0; src++, dst++, size--) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    memcpy(dst, src, sizeof(posit8e2));
#else
    memcpy(
        dst,
        reinterpret_cast<const char*>(src) + sizeof(float) - sizeof(posit8e2),
        sizeof(posit8e2));
#endif
  }
}

void Posit8e2ToFloat(const posit8e2* src, float* dst, int64 size) {
  Eigen::Map<const Eigen::Array<posit8e2, Eigen::Dynamic, 1>> src_eigen(src,
                                                                        size);
  Eigen::Map<Eigen::ArrayXf> dst_eigen(dst, size);
  dst_eigen = src_eigen.cast<float>();
}

// =======================


void RoundFloatToPosit32e2(const float* src, posit32e2* dst, int64 size) {
  Eigen::Map<const Eigen::ArrayXf> src_eigen(src, size);
  Eigen::Map<Eigen::Array<posit32e2, Eigen::Dynamic, 1>> dst_eigen(dst, size);
  dst_eigen = src_eigen.cast<posit32e2>();
}

void FloatToPosit32e2(const float* src, posit32e2* dst, int64 size) {
  for (; size != 0; src++, dst++, size--) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    memcpy(dst, src, sizeof(posit32e2));
#else
    memcpy(
        dst,
        reinterpret_cast<const char*>(src) + sizeof(float) - sizeof(posit32e2),
        sizeof(posit32e2));
#endif
  }
}

void Posit32e2ToFloat(const posit32e2* src, float* dst, int64 size) {
  Eigen::Map<const Eigen::Array<posit32e2, Eigen::Dynamic, 1>> src_eigen(src,
                                                                        size);
  Eigen::Map<Eigen::ArrayXf> dst_eigen(dst, size);
  dst_eigen = src_eigen.cast<float>();
}

}  // end namespace tensorflow
