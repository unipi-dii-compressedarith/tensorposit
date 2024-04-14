/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PYTHON_LIB_CORE_posit8e2_H_
#define TENSORFLOW_PYTHON_LIB_CORE_posit8e2_H_

#include <Python.h>

namespace tensorflow {

// Register the bfloat16 numpy type. Returns true on success.
bool RegisterNumpyPosit8e2();

// Returns a pointer to the bfloat16 dtype object.
PyObject* posit8e2Dtype();

// Returns the id number of the bfloat16 numpy type.
int posit8e2NumpyType();

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_LIB_CORE_posit8e2_H_