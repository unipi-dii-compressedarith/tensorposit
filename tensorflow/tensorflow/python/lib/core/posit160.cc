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

#include "tensorflow/python/lib/core/posit160.h"

#include <array>
#include <locale>
// Place `<locale>` before <Python.h> to avoid a build failure in macOS.
#include <Python.h>

#include "absl/strings/str_cat.h"
#include "third_party/cppposit_private/include/posit.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/python/lib/core/numpy.h"

namespace tensorflow {
namespace {

//using posit160 = posit::Posit<int16_t, 16 , 1, uint_fast32_t, posit::PositSpec::WithNan>;
using posit160 = tensorflow::posit160;
struct PyDecrefDeleter {
  void operator()(PyObject* p) const { Py_DECREF(p); }
};

// Safe container for an owned PyObject. On destruction, the reference count of
// the contained object will be decremented.
using Safe_PyObjectPtr = std::unique_ptr<PyObject, PyDecrefDeleter>;
Safe_PyObjectPtr make_safe(PyObject* object) {
  return Safe_PyObjectPtr(object);
}

bool PyLong_CheckNoOverflow(PyObject* object) {
  if (!PyLong_Check(object)) {
    return false;
  }
  int overflow = 0;
  PyLong_AsLongAndOverflow(object, &overflow);
  return (overflow == 0);
}

// Registered numpy type ID. Global variable populated by the registration code.
// Protected by the GIL.
int npy_posit160 = NPY_NOTYPE;

// Forward declaration.
extern PyTypeObject posit160_type;

// Pointer to the posit160 type object we are using. This is either a pointer
// to posit160_type, if we choose to register it, or to the posit160 type
// registered by another system into NumPy.
PyTypeObject* posit160_type_ptr = nullptr;

// Representation of a Python posit160 object.
struct PyPosit160 {
  PyObject_HEAD;  // Python object header
  posit160 value;
};

// Returns true if 'object' is a PyPosit160.
bool PyPosit160_Check(PyObject* object) {
  return PyObject_IsInstance(object,
                             reinterpret_cast<PyObject*>(&posit160_type));
}

// Extracts the value of a PyPosit160 object.
posit160 PyPosit160_Posit160(PyObject* object) {
  return reinterpret_cast<PyPosit160*>(object)->value;
}

// Constructs a PyPosit160 object from a posit160.
Safe_PyObjectPtr PyPosit160_FromPosit160(posit160 x) {
  Safe_PyObjectPtr ref = make_safe(posit160_type.tp_alloc(&posit160_type, 0));
  PyPosit160* p = reinterpret_cast<PyPosit160*>(ref.get());
  if (p) {
    p->value = x;
  }
  return ref;
}

// Converts a Python object to a posit160 value. Returns true on success,
// returns false and reports a Python error on failure.
bool CastToPosit160(PyObject* arg, posit160* output) {
  if (PyPosit160_Check(arg)) {
    *output = PyPosit160_Posit160(arg);
    return true;
  }
  if (PyFloat_Check(arg)) {
    double d = PyFloat_AsDouble(arg);
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(phawkins): check for overflow
    *output = posit160(d);
    return true;
  }
  if (PyLong_CheckNoOverflow(arg)) {
    long l = PyLong_AsLong(arg);  // NOLINT
    if (PyErr_Occurred()) {
      return false;
    }
    // TODO(phawkins): check for overflow
    *output = posit160(static_cast<float>(l));
    return true;
  }
  if (PyArray_IsScalar(arg, Half)) {
    Eigen::half f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = posit160(f);
    return true;
  }
  if (PyArray_IsScalar(arg, Float)) {
    float f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = posit160(f);
    return true;
  }
  if (PyArray_IsScalar(arg, Double)) {
    double f;
    PyArray_ScalarAsCtype(arg, &f);
    *output = posit160(f);
    return true;
  }
  if (PyArray_IsZeroDim(arg)) {
    Safe_PyObjectPtr ref;
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(arg);
    if (PyArray_TYPE(arr) != npy_posit160) {
      ref = make_safe(PyArray_Cast(arr, npy_posit160));
      if (PyErr_Occurred()) {
        return false;
      }
      arg = ref.get();
      arr = reinterpret_cast<PyArrayObject*>(arg);
    }
    *output = *reinterpret_cast<posit160*>(PyArray_DATA(arr));
    return true;
  }
  return false;
}

bool SafeCastToPosit160(PyObject* arg, posit160* output) {
  if (PyPosit160_Check(arg)) {
    *output = PyPosit160_Posit160(arg);
    return true;
  }
  return false;
}

// Converts a PyPosit160 into a PyFloat.
PyObject* PyPosit160_Float(PyObject* self) {
  posit160 x = PyPosit160_Posit160(self);
  return PyFloat_FromDouble(static_cast<double>(x));
}

// Converts a PyPosit160 into a PyInt.
PyObject* PyPosit160_Int(PyObject* self) {
  posit160 x = PyPosit160_Posit160(self);
  long y = static_cast<long>(x);  // NOLINT
  return PyLong_FromLong(y);
}

// Negates a PyPosit160.
PyObject* PyPosit160_Negative(PyObject* self) {
  posit160 x = PyPosit160_Posit160(self);
  return PyPosit160_FromPosit160(-x).release();
}

PyObject* PyPosit160_Add(PyObject* a, PyObject* b) {
  posit160 x, y;
  if (SafeCastToPosit160(a, &x) && SafeCastToPosit160(b, &y)) {
    return PyPosit160_FromPosit160(x + y).release();
  }
  return PyArray_Type.tp_as_number->nb_add(a, b);
}

PyObject* PyPosit160_Subtract(PyObject* a, PyObject* b) {
  posit160 x, y;
  if (SafeCastToPosit160(a, &x) && SafeCastToPosit160(b, &y)) {
    return PyPosit160_FromPosit160(x - y).release();
  }
  return PyArray_Type.tp_as_number->nb_subtract(a, b);
}

PyObject* PyPosit160_Multiply(PyObject* a, PyObject* b) {
  posit160 x, y;
  if (SafeCastToPosit160(a, &x) && SafeCastToPosit160(b, &y)) {
    return PyPosit160_FromPosit160(x * y).release();
  }
  return PyArray_Type.tp_as_number->nb_multiply(a, b);
}

PyObject* PyPosit160_TrueDivide(PyObject* a, PyObject* b) {
  posit160 x, y;
  if (SafeCastToPosit160(a, &x) && SafeCastToPosit160(b, &y)) {
    return PyPosit160_FromPosit160(x / y).release();
  }
  return PyArray_Type.tp_as_number->nb_true_divide(a, b);
}

// Python number methods for PyPosit160 objects.
PyNumberMethods PyPosit160_AsNumber = {
    PyPosit160_Add,       // nb_add
    PyPosit160_Subtract,  // nb_subtract
    PyPosit160_Multiply,  // nb_multiply
    nullptr,              // nb_remainder
    nullptr,              // nb_divmod
    nullptr,              // nb_power
    PyPosit160_Negative,  // nb_negative
    nullptr,              // nb_positive
    nullptr,              // nb_absolute
    nullptr,              // nb_nonzero
    nullptr,              // nb_invert
    nullptr,              // nb_lshift
    nullptr,              // nb_rshift
    nullptr,              // nb_and
    nullptr,              // nb_xor
    nullptr,              // nb_or
    PyPosit160_Int,       // nb_int
    nullptr,              // reserved
    PyPosit160_Float,     // nb_float

    nullptr,  // nb_inplace_add
    nullptr,  // nb_inplace_subtract
    nullptr,  // nb_inplace_multiply
    nullptr,  // nb_inplace_remainder
    nullptr,  // nb_inplace_power
    nullptr,  // nb_inplace_lshift
    nullptr,  // nb_inplace_rshift
    nullptr,  // nb_inplace_and
    nullptr,  // nb_inplace_xor
    nullptr,  // nb_inplace_or

    nullptr,                // nb_floor_divide
    PyPosit160_TrueDivide,  // nb_true_divide
    nullptr,                // nb_inplace_floor_divide
    nullptr,                // nb_inplace_true_divide
    nullptr,                // nb_index
};

// Constructs a new PyPosit160.
PyObject* PyPosit160_New(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  if (kwds && PyDict_Size(kwds)) {
    PyErr_SetString(PyExc_TypeError, "constructor takes no keyword arguments");
    return nullptr;
  }
  Py_ssize_t size = PyTuple_Size(args);
  if (size != 1) {
    PyErr_SetString(PyExc_TypeError,
                    "expected number as argument to posit160 constructor");
    return nullptr;
  }
  PyObject* arg = PyTuple_GetItem(args, 0);

  posit160 value;
  if (PyPosit160_Check(arg)) {
    Py_INCREF(arg);
    return arg;
  } else if (CastToPosit160(arg, &value)) {
    return PyPosit160_FromPosit160(value).release();
  } else if (PyArray_Check(arg)) {
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(arg);
    if (PyArray_TYPE(arr) != npy_posit160) {
      return PyArray_Cast(arr, npy_posit160);
    } else {
      Py_INCREF(arg);
      return arg;
    }
  }
  PyErr_Format(PyExc_TypeError, "expected number, got %s",
               arg->ob_type->tp_name);
  return nullptr;
}

// Comparisons on PyPosit160s.
PyObject* PyPosit160_RichCompare(PyObject* a, PyObject* b, int op) {
  posit160 x, y;
  if (!SafeCastToPosit160(a, &x) || !SafeCastToPosit160(b, &y)) {
    return PyGenericArrType_Type.tp_richcompare(a, b, op);
  }
  bool result;
  switch (op) {
    case Py_LT:
      result = x < y;
      break;
    case Py_LE:
      result = x <= y;
      break;
    case Py_EQ:
      result = x == y;
      break;
    case Py_NE:
      result = x != y;
      break;
    case Py_GT:
      result = x > y;
      break;
    case Py_GE:
      result = x >= y;
      break;
    default:
      LOG(FATAL) << "Invalid op type " << op;
  }
  return PyBool_FromLong(result);
}

// Implementation of repr() for PyPosit160.
PyObject* PyPosit160_Repr(PyObject* self) {
  posit160 x = reinterpret_cast<PyPosit160*>(self)->value;
  std::string v = absl::StrCat(static_cast<float>(x));
  return PyUnicode_FromString(v.c_str());
}

// Implementation of str() for PyPosit160.
PyObject* PyPosit160_Str(PyObject* self) {
  posit160 x = reinterpret_cast<PyPosit160*>(self)->value;
  std::string v = absl::StrCat(static_cast<float>(x));
  return PyUnicode_FromString(v.c_str());
}

// Hash function for PyPosit160. We use the identity function, which is a weak
// hash function.
Py_hash_t PyPosit160_Hash(PyObject* self) {
  return Eigen::numext::bit_cast<uint16_t>(
      reinterpret_cast<PyPosit160*>(self)->value);
}

// Python type for PyPosit160 objects.
PyTypeObject posit160_type = {
    PyVarObject_HEAD_INIT(nullptr, 0) "posit160",  // tp_name
    sizeof(PyPosit160),                            // tp_basicsize
    0,                                             // tp_itemsize
    nullptr,                                       // tp_dealloc
#if PY_VERSION_HEX < 0x03080000
    nullptr,  // tp_print
#else
    0,  // tp_vectorcall_offset
#endif
    nullptr,               // tp_getattr
    nullptr,               // tp_setattr
    nullptr,               // tp_compare / tp_reserved
    PyPosit160_Repr,       // tp_repr
    &PyPosit160_AsNumber,  // tp_as_number
    nullptr,               // tp_as_sequence
    nullptr,               // tp_as_mapping
    PyPosit160_Hash,       // tp_hash
    nullptr,               // tp_call
    PyPosit160_Str,        // tp_str
    nullptr,               // tp_getattro
    nullptr,               // tp_setattro
    nullptr,               // tp_as_buffer
                           // tp_flags
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "posit160 floating-point values",  // tp_doc
    nullptr,                           // tp_traverse
    nullptr,                           // tp_clear
    PyPosit160_RichCompare,            // tp_richcompare
    0,                                 // tp_weaklistoffset
    nullptr,                           // tp_iter
    nullptr,                           // tp_iternext
    nullptr,                           // tp_methods
    nullptr,                           // tp_members
    nullptr,                           // tp_getset
    nullptr,                           // tp_base
    nullptr,                           // tp_dict
    nullptr,                           // tp_descr_get
    nullptr,                           // tp_descr_set
    0,                                 // tp_dictoffset
    nullptr,                           // tp_init
    nullptr,                           // tp_alloc
    PyPosit160_New,                    // tp_new
    nullptr,                           // tp_free
    nullptr,                           // tp_is_gc
    nullptr,                           // tp_bases
    nullptr,                           // tp_mro
    nullptr,                           // tp_cache
    nullptr,                           // tp_subclasses
    nullptr,                           // tp_weaklist
    nullptr,                           // tp_del
    0,                                 // tp_version_tag
};

// Numpy support

PyArray_ArrFuncs NPyPosit160_ArrFuncs;

PyArray_Descr NPyPosit160_Descr = {
    PyObject_HEAD_INIT(nullptr)  //
                                 /*typeobj=*/
    (&posit160_type),
    // We must register posit160 with a kind other than "f", because numpy
    // considers two types with the same kind and size to be equal, but
    // float16 != posit160.
    // The downside of this is that NumPy scalar promotion does not work with
    // posit160 values.
    /*kind=*/'V',
    // TODO(phawkins): there doesn't seem to be a way of guaranteeing a type
    // character is unique.
    /*type=*/'E',
    /*byteorder=*/'=',
    /*flags=*/NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM,
    /*type_num=*/0,
    /*elsize=*/sizeof(posit160),
    /*alignment=*/alignof(posit160),
    /*subarray=*/nullptr,
    /*fields=*/nullptr,
    /*names=*/nullptr,
    /*f=*/&NPyPosit160_ArrFuncs,
    /*metadata=*/nullptr,
    /*c_metadata=*/nullptr,
    /*hash=*/-1,  // -1 means "not computed yet".
};

// Implementations of NumPy array methods.

PyObject* NPyPosit160_GetItem(void* data, void* arr) {
  posit160 x;
  memcpy(&x, data, sizeof(posit160));
  return PyPosit160_FromPosit160(x).release();
}

int NPyPosit160_SetItem(PyObject* item, void* data, void* arr) {
  posit160 x;
  if (!CastToPosit160(item, &x)) {
    PyErr_Format(PyExc_TypeError, "expected number, got %s",
                 item->ob_type->tp_name);
    return -1;
  }
  memcpy(data, &x, sizeof(posit160));
  return 0;
}

void ByteSwap16(void* value) {
  char* p = reinterpret_cast<char*>(value);
  std::swap(p[0], p[1]);
}

int NPyPosit160_Compare(const void* a, const void* b, void* arr) {
  posit160 x;
  memcpy(&x, a, sizeof(posit160));

  posit160 y;
  memcpy(&y, b, sizeof(posit160));

  if (x < y) {
    return -1;
  }
  if (y < x) {
    return 1;
  }
  // NaNs sort to the end.
  if (!Eigen::numext::isnan(x) && Eigen::numext::isnan(y)) {
    return -1;
  }
  if (Eigen::numext::isnan(x) && !Eigen::numext::isnan(y)) {
    return 1;
  }
  return 0;
}

void NPyPosit160_CopySwapN(void* dstv, npy_intp dstride, void* srcv,
                           npy_intp sstride, npy_intp n, int swap, void* arr) {
  char* dst = reinterpret_cast<char*>(dstv);
  char* src = reinterpret_cast<char*>(srcv);
  if (!src) {
    return;
  }
  if (swap) {
    for (npy_intp i = 0; i < n; i++) {
      char* r = dst + dstride * i;
      memcpy(r, src + sstride * i, sizeof(uint16_t));
      ByteSwap16(r);
    }
  } else if (dstride == sizeof(uint16_t) && sstride == sizeof(uint16_t)) {
    memcpy(dst, src, n * sizeof(uint16_t));
  } else {
    for (npy_intp i = 0; i < n; i++) {
      memcpy(dst + dstride * i, src + sstride * i, sizeof(uint16_t));
    }
  }
}

void NPyPosit160_CopySwap(void* dst, void* src, int swap, void* arr) {
  if (!src) {
    return;
  }
  memcpy(dst, src, sizeof(uint16_t));
  if (swap) {
    ByteSwap16(dst);
  }
}

npy_bool NPyPosit160_NonZero(void* data, void* arr) {
  posit160 x;
  memcpy(&x, data, sizeof(x));
  return x != static_cast<posit160>(0);
}

int NPyPosit160_Fill(void* buffer_raw, npy_intp length, void* ignored) {
  posit160* const buffer = reinterpret_cast<posit160*>(buffer_raw);
  const float start(buffer[0]);
  const float delta = static_cast<float>(buffer[1]) - start;
  for (npy_intp i = 2; i < length; ++i) {
    buffer[i] = static_cast<posit160>(start + i * delta);
  }
  return 0;
}

void NPyPosit160_DotFunc(void* ip1, npy_intp is1, void* ip2, npy_intp is2,
                         void* op, npy_intp n, void* arr) {
  char* c1 = reinterpret_cast<char*>(ip1);
  char* c2 = reinterpret_cast<char*>(ip2);
  float acc = 0.0f;
  for (npy_intp i = 0; i < n; ++i) {
    posit160* const b1 = reinterpret_cast<posit160*>(c1);
    posit160* const b2 = reinterpret_cast<posit160*>(c2);
    acc += static_cast<float>(*b1) * static_cast<float>(*b2);
    c1 += is1;
    c2 += is2;
  }
  posit160* out = reinterpret_cast<posit160*>(op);
  *out = static_cast<posit160>(acc);
}

int NPyPosit160_CompareFunc(const void* v1, const void* v2, void* arr) {
  posit160 b1 = *reinterpret_cast<const posit160*>(v1);
  posit160 b2 = *reinterpret_cast<const posit160*>(v2);
  if (b1 < b2) {
    return -1;
  }
  if (b1 > b2) {
    return 1;
  }
  return 0;
}

int NPyPosit160_ArgMaxFunc(void* data, npy_intp n, npy_intp* max_ind,
                           void* arr) {
  const posit160* bdata = reinterpret_cast<const posit160*>(data);
  float max_val = -std::numeric_limits<float>::infinity();
  for (npy_intp i = 0; i < n; ++i) {
    if (static_cast<float>(bdata[i]) > max_val) {
      max_val = static_cast<float>(bdata[i]);
      *max_ind = i;
    }
  }
  return 0;
}

int NPyPosit160_ArgMinFunc(void* data, npy_intp n, npy_intp* min_ind,
                           void* arr) {
  const posit160* bdata = reinterpret_cast<const posit160*>(data);
  float min_val = std::numeric_limits<float>::infinity();
  for (npy_intp i = 0; i < n; ++i) {
    if (static_cast<float>(bdata[i]) < min_val) {
      min_val = static_cast<float>(bdata[i]);
      *min_ind = i;
    }
  }
  return 0;
}

// NumPy casts

template <typename T, typename Enable = void>
struct TypeDescriptor {
  // typedef ... T;  // Representation type in memory for NumPy values of type
  // static int Dtype() { return NPY_...; }  // Numpy type number for T.
};

template <>
struct TypeDescriptor<posit160> {
  typedef posit160 T;
  static int Dtype() { return npy_posit160; }
};

template <>
struct TypeDescriptor<uint8> {
  typedef uint8 T;
  static int Dtype() { return NPY_UINT8; }
};

template <>
struct TypeDescriptor<uint16> {
  typedef uint16 T;
  static int Dtype() { return NPY_UINT16; }
};

// We register "int", "long", and "long long" types for portability across
// Linux, where "int" and "long" are the same type, and Windows, where "long"
// and "longlong" are the same type.
template <>
struct TypeDescriptor<unsigned int> {
  typedef unsigned int T;
  static int Dtype() { return NPY_UINT; }
};

template <>
struct TypeDescriptor<unsigned long> {  // NOLINT
  typedef unsigned long T;              // NOLINT
  static int Dtype() { return NPY_ULONG; }
};

template <>
struct TypeDescriptor<unsigned long long> {  // NOLINT
  typedef unsigned long long T;              // NOLINT
  static int Dtype() { return NPY_ULONGLONG; }
};

template <>
struct TypeDescriptor<int8> {
  typedef int8 T;
  static int Dtype() { return NPY_INT8; }
};

template <>
struct TypeDescriptor<int16> {
  typedef int16 T;
  static int Dtype() { return NPY_INT16; }
};

template <>
struct TypeDescriptor<int> {
  typedef int T;
  static int Dtype() { return NPY_INT; }
};

template <>
struct TypeDescriptor<long> {  // NOLINT
  typedef long T;              // NOLINT
  static int Dtype() { return NPY_LONG; }
};

template <>
struct TypeDescriptor<long long> {  // NOLINT
  typedef long long T;              // NOLINT
  static int Dtype() { return NPY_LONGLONG; }
};

template <>
struct TypeDescriptor<bool> {
  typedef int8 T;
  static int Dtype() { return NPY_BOOL; }
};

template <>
struct TypeDescriptor<Eigen::half> {
  typedef Eigen::half T;
  static int Dtype() { return NPY_HALF; }
};

template <>
struct TypeDescriptor<float> {
  typedef float T;
  static int Dtype() { return NPY_FLOAT; }
};

template <>
struct TypeDescriptor<double> {
  typedef double T;
  static int Dtype() { return NPY_DOUBLE; }
};

template <>
struct TypeDescriptor<std::complex<float>> {
  typedef std::complex<float> T;
  static int Dtype() { return NPY_COMPLEX64; }
};

template <>
struct TypeDescriptor<std::complex<double>> {
  typedef std::complex<double> T;
  static int Dtype() { return NPY_COMPLEX128; }
};

// Performs a NumPy array cast from type 'From' to 'To'.
template <typename From, typename To>
void NPyCast(void* from_void, void* to_void, npy_intp n, void* fromarr,
             void* toarr) {
  const auto* from =
      reinterpret_cast<typename TypeDescriptor<From>::T*>(from_void);
  auto* to = reinterpret_cast<typename TypeDescriptor<To>::T*>(to_void);
  for (npy_intp i = 0; i < n; ++i) {
    to[i] =
        static_cast<typename TypeDescriptor<To>::T>(static_cast<To>(from[i]));
  }
}

// Registers a cast between posit160 and type 'T'. 'numpy_type' is the NumPy
// type corresponding to 'T'.
template <typename T>
bool RegisterPosit160Cast(int numpy_type) {
  PyArray_Descr* descr = PyArray_DescrFromType(numpy_type);
  if (PyArray_RegisterCastFunc(descr, npy_posit160, NPyCast<T, posit160>) < 0) {
    return false;
  }
  if (PyArray_RegisterCastFunc(&NPyPosit160_Descr, numpy_type,
                               NPyCast<posit160, T>) < 0) {
    return false;
  }
  return true;
}

template <typename InType, typename OutType, typename Functor>
struct UnaryUFunc {
  static std::vector<int> Types() {
    return {TypeDescriptor<InType>::Dtype(), TypeDescriptor<OutType>::Dtype()};
  }
  static void Call(char** args, const npy_intp* dimensions,
                   const npy_intp* steps, void* data) {
    const char* i0 = args[0];
    char* o = args[1];
    for (npy_intp k = 0; k < *dimensions; k++) {
      auto x = *reinterpret_cast<const typename TypeDescriptor<InType>::T*>(i0);
      *reinterpret_cast<typename TypeDescriptor<OutType>::T*>(o) = Functor()(x);
      i0 += steps[0];
      o += steps[1];
    }
  }
};

template <typename InType, typename OutType, typename OutType2,
          typename Functor>
struct UnaryUFunc2 {
  static std::vector<int> Types() {
    return {TypeDescriptor<InType>::Dtype(), TypeDescriptor<OutType>::Dtype(),
            TypeDescriptor<OutType2>::Dtype()};
  }
  static void Call(char** args, const npy_intp* dimensions,
                   const npy_intp* steps, void* data) {
    const char* i0 = args[0];
    char* o0 = args[1];
    char* o1 = args[2];
    for (npy_intp k = 0; k < *dimensions; k++) {
      auto x = *reinterpret_cast<const typename TypeDescriptor<InType>::T*>(i0);
      std::tie(*reinterpret_cast<typename TypeDescriptor<OutType>::T*>(o0),
               *reinterpret_cast<typename TypeDescriptor<OutType2>::T*>(o1)) =
          Functor()(x);
      i0 += steps[0];
      o0 += steps[1];
      o1 += steps[2];
    }
  }
};

template <typename InType, typename OutType, typename Functor>
struct BinaryUFunc {
  static std::vector<int> Types() {
    return {TypeDescriptor<InType>::Dtype(), TypeDescriptor<InType>::Dtype(),
            TypeDescriptor<OutType>::Dtype()};
  }
  static void Call(char** args, const npy_intp* dimensions,
                   const npy_intp* steps, void* data) {
    const char* i0 = args[0];
    const char* i1 = args[1];
    char* o = args[2];
    for (npy_intp k = 0; k < *dimensions; k++) {
      auto x = *reinterpret_cast<const typename TypeDescriptor<InType>::T*>(i0);
      auto y = *reinterpret_cast<const typename TypeDescriptor<InType>::T*>(i1);
      *reinterpret_cast<typename TypeDescriptor<OutType>::T*>(o) =
          Functor()(x, y);
      i0 += steps[0];
      i1 += steps[1];
      o += steps[2];
    }
  }
};

template <typename InType, typename InType2, typename OutType, typename Functor>
struct BinaryUFunc2 {
  static std::vector<int> Types() {
    return {TypeDescriptor<InType>::Dtype(), TypeDescriptor<InType2>::Dtype(),
            TypeDescriptor<OutType>::Dtype()};
  }
  static void Call(char** args, const npy_intp* dimensions,
                   const npy_intp* steps, void* data) {
    const char* i0 = args[0];
    const char* i1 = args[1];
    char* o = args[2];
    for (npy_intp k = 0; k < *dimensions; k++) {
      auto x = *reinterpret_cast<const typename TypeDescriptor<InType>::T*>(i0);
      auto y =
          *reinterpret_cast<const typename TypeDescriptor<InType2>::T*>(i1);
      *reinterpret_cast<typename TypeDescriptor<OutType>::T*>(o) =
          Functor()(x, y);
      i0 += steps[0];
      i1 += steps[1];
      o += steps[2];
    }
  }
};

template <typename UFunc>
bool RegisterUFunc(PyObject* numpy, const char* name) {
  std::vector<int> types = UFunc::Types();
  PyUFuncGenericFunction fn =
      reinterpret_cast<PyUFuncGenericFunction>(UFunc::Call);
  Safe_PyObjectPtr ufunc_obj = make_safe(PyObject_GetAttrString(numpy, name));
  if (!ufunc_obj) {
    return false;
  }
  PyUFuncObject* ufunc = reinterpret_cast<PyUFuncObject*>(ufunc_obj.get());
  if (static_cast<int>(types.size()) != ufunc->nargs) {
    PyErr_Format(PyExc_AssertionError,
                 "ufunc %s takes %d arguments, loop takes %lu", name,
                 ufunc->nargs, types.size());
    return false;
  }
  if (PyUFunc_RegisterLoopForType(ufunc, npy_posit160, fn,
                                  const_cast<int*>(types.data()),
                                  nullptr) < 0) {
    return false;
  }
  return true;
}

namespace ufuncs {

struct Add {
  posit160 operator()(posit160 a, posit160 b) { return a + b; }
};
struct Subtract {
  posit160 operator()(posit160 a, posit160 b) { return a - b; }
};
struct Multiply {
  posit160 operator()(posit160 a, posit160 b) { return a * b; }
};
struct TrueDivide {
  posit160 operator()(posit160 a, posit160 b) { return a / b; }
};

std::pair<float, float> divmod(float a, float b) {
  if (b == 0.0f) {
    float nan = std::numeric_limits<float>::quiet_NaN();
    return {nan, nan};
  }
  float mod = std::fmod(a, b);
  float div = (a - mod) / b;
  if (mod != 0.0f) {
    if ((b < 0.0f) != (mod < 0.0f)) {
      mod += b;
      div -= 1.0f;
    }
  } else {
    mod = std::copysign(0.0f, b);
  }

  float floordiv;
  if (div != 0.0f) {
    floordiv = std::floor(div);
    if (div - floordiv > 0.5f) {
      floordiv += 1.0f;
    }
  } else {
    floordiv = std::copysign(0.0f, a / b);
  }
  return {floordiv, mod};
}

struct FloorDivide {
  posit160 operator()(posit160 a, posit160 b) {
    return posit160(divmod(static_cast<float>(a), static_cast<float>(b)).first);
  }
};
struct Remainder {
  posit160 operator()(posit160 a, posit160 b) {
    return posit160(
        divmod(static_cast<float>(a), static_cast<float>(b)).second);
  }
};
struct DivmodUFunc {
  static std::vector<int> Types() {
    return {npy_posit160, npy_posit160, npy_posit160, npy_posit160};
  }
  static void Call(char** args, npy_intp* dimensions, npy_intp* steps,
                   void* data) {
    const char* i0 = args[0];
    const char* i1 = args[1];
    char* o0 = args[2];
    char* o1 = args[3];
    for (npy_intp k = 0; k < *dimensions; k++) {
      posit160 x = *reinterpret_cast<const posit160*>(i0);
      posit160 y = *reinterpret_cast<const posit160*>(i1);
      float floordiv, mod;
      std::tie(floordiv, mod) =
          divmod(static_cast<float>(x), static_cast<float>(y));
      *reinterpret_cast<posit160*>(o0) = posit160(floordiv);
      *reinterpret_cast<posit160*>(o1) = posit160(mod);
      i0 += steps[0];
      i1 += steps[1];
      o0 += steps[2];
      o1 += steps[3];
    }
  }
};
struct Fmod {
  posit160 operator()(posit160 a, posit160 b) {
    return posit160(std::fmod(static_cast<float>(a), static_cast<float>(b)));
  }
};
struct Negative {
  posit160 operator()(posit160 a) { return -a; }
};
struct Positive {
  posit160 operator()(posit160 a) { return a; }
};
struct Power {
  posit160 operator()(posit160 a, posit160 b) {
    return posit160(std::pow(static_cast<float>(a), static_cast<float>(b)));
  }
};
struct Abs {
  posit160 operator()(posit160 a) {
    return posit160(std::abs(static_cast<float>(a)));
  }
};
struct Cbrt {
  posit160 operator()(posit160 a) {
    return posit160(std::cbrt(static_cast<float>(a)));
  }
};
struct Ceil {
  posit160 operator()(posit160 a) {
    return posit160(std::ceil(static_cast<float>(a)));
  }
};
struct CopySign {
  posit160 operator()(posit160 a, posit160 b) {
    return posit160(
        std::copysign(static_cast<float>(a), static_cast<float>(b)));
  }
};
struct Exp {
  posit160 operator()(posit160 a) {
    return posit160(std::exp(static_cast<float>(a)));
  }
};
struct Exp2 {
  posit160 operator()(posit160 a) {
    return posit160(std::exp2(static_cast<float>(a)));
  }
};
struct Expm1 {
  posit160 operator()(posit160 a) {
    return posit160(std::expm1(static_cast<float>(a)));
  }
};
struct Floor {
  posit160 operator()(posit160 a) {
    return posit160(std::floor(static_cast<float>(a)));
  }
};
struct Frexp {
  std::pair<posit160, int> operator()(posit160 a) {
    int exp;
    float f = std::frexp(static_cast<float>(a), &exp);
    return {posit160(f), exp};
  }
};
struct Heaviside {
  posit160 operator()(posit160 bx, posit160 h0) {
    float x = static_cast<float>(bx);
    if (Eigen::numext::isnan(x)) {
      return bx;
    }
    if (x < 0) {
      return posit160(0.0f);
    }
    if (x > 0) {
      return posit160(1.0f);
    }
    return h0;  // x == 0
  }
};
struct Conjugate {
  posit160 operator()(posit160 a) { return a; }
};
struct IsFinite {
  bool operator()(posit160 a) { return std::isfinite(static_cast<float>(a)); }
};
struct IsInf {
  bool operator()(posit160 a) { return std::isinf(static_cast<float>(a)); }
};
struct IsNan {
  bool operator()(posit160 a) {
    return Eigen::numext::isnan(static_cast<float>(a));
  }
};
struct Ldexp {
  posit160 operator()(posit160 a, int exp) {
    return posit160(std::ldexp(static_cast<float>(a), exp));
  }
};
struct Log {
  posit160 operator()(posit160 a) {
    return posit160(std::log(static_cast<float>(a)));
  }
};
struct Log2 {
  posit160 operator()(posit160 a) {
    return posit160(std::log2(static_cast<float>(a)));
  }
};
struct Log10 {
  posit160 operator()(posit160 a) {
    return posit160(std::log10(static_cast<float>(a)));
  }
};
struct Log1p {
  posit160 operator()(posit160 a) {
    return posit160(std::log1p(static_cast<float>(a)));
  }
};
struct LogAddExp {
  posit160 operator()(posit160 bx, posit160 by) {
    float x = static_cast<float>(bx);
    float y = static_cast<float>(by);
    if (x == y) {
      // Handles infinities of the same sign.
      return posit160(x + std::log(2.0f));
    }
    float out = std::numeric_limits<float>::quiet_NaN();
    if (x > y) {
      out = x + std::log1p(std::exp(y - x));
    } else if (x < y) {
      out = y + std::log1p(std::exp(x - y));
    }
    return posit160(out);
  }
};
struct LogAddExp2 {
  posit160 operator()(posit160 bx, posit160 by) {
    float x = static_cast<float>(bx);
    float y = static_cast<float>(by);
    if (x == y) {
      // Handles infinities of the same sign.
      return posit160(x + 1.0f);
    }
    float out = std::numeric_limits<float>::quiet_NaN();
    if (x > y) {
      out = x + std::log1p(std::exp2(y - x)) / std::log(2.0f);
    } else if (x < y) {
      out = y + std::log1p(std::exp2(x - y)) / std::log(2.0f);
    }
    return posit160(out);
  }
};
struct Modf {
  std::pair<posit160, posit160> operator()(posit160 a) {
    float integral;
    float f = std::modf(static_cast<float>(a), &integral);
    return {posit160(f), posit160(integral)};
  }
};

struct Reciprocal {
  posit160 operator()(posit160 a) {
    return posit160(1.f / static_cast<float>(a));
  }
};
struct Rint {
  posit160 operator()(posit160 a) {
    return posit160(std::rint(static_cast<float>(a)));
  }
};
struct Sign {
  posit160 operator()(posit160 a) {
    float f(a);
    if (f < 0) {
      return posit160(-1);
    }
    if (f > 0) {
      return posit160(1);
    }
    return a;
  }
};
struct SignBit {
  bool operator()(posit160 a) { return std::signbit(static_cast<float>(a)); }
};
struct Sqrt {
  posit160 operator()(posit160 a) {
    return posit160(std::sqrt(static_cast<float>(a)));
  }
};
struct Square {
  posit160 operator()(posit160 a) {
    float f(a);
    return posit160(f * f);
  }
};
struct Trunc {
  posit160 operator()(posit160 a) {
    return posit160(std::trunc(static_cast<float>(a)));
  }
};

// Trigonometric functions
struct Sin {
  posit160 operator()(posit160 a) {
    return posit160(std::sin(static_cast<float>(a)));
  }
};
struct Cos {
  posit160 operator()(posit160 a) {
    return posit160(std::cos(static_cast<float>(a)));
  }
};
struct Tan {
  posit160 operator()(posit160 a) {
    return posit160(std::tan(static_cast<float>(a)));
  }
};
struct Arcsin {
  posit160 operator()(posit160 a) {
    return posit160(std::asin(static_cast<float>(a)));
  }
};
struct Arccos {
  posit160 operator()(posit160 a) {
    return posit160(std::acos(static_cast<float>(a)));
  }
};
struct Arctan {
  posit160 operator()(posit160 a) {
    return posit160(std::atan(static_cast<float>(a)));
  }
};
struct Arctan2 {
  posit160 operator()(posit160 a, posit160 b) {
    return posit160(std::atan2(static_cast<float>(a), static_cast<float>(b)));
  }
};
struct Hypot {
  posit160 operator()(posit160 a, posit160 b) {
    return posit160(std::hypot(static_cast<float>(a), static_cast<float>(b)));
  }
};
struct Sinh {
  posit160 operator()(posit160 a) {
    return posit160(std::sinh(static_cast<float>(a)));
  }
};
struct Cosh {
  posit160 operator()(posit160 a) {
    return posit160(std::cosh(static_cast<float>(a)));
  }
};
struct Tanh {
  posit160 operator()(posit160 a) {
    return posit160(std::tanh(static_cast<float>(a)));
  }
};
struct Arcsinh {
  posit160 operator()(posit160 a) {
    return posit160(std::asinh(static_cast<float>(a)));
  }
};
struct Arccosh {
  posit160 operator()(posit160 a) {
    return posit160(std::acosh(static_cast<float>(a)));
  }
};
struct Arctanh {
  posit160 operator()(posit160 a) {
    return posit160(std::atanh(static_cast<float>(a)));
  }
};
struct Deg2rad {
  posit160 operator()(posit160 a) {
    static constexpr float radians_per_degree = M_PI / 180.0f;
    return posit160(static_cast<float>(a) * radians_per_degree);
  }
};
struct Rad2deg {
  posit160 operator()(posit160 a) {
    static constexpr float degrees_per_radian = 180.0f / M_PI;
    return posit160(static_cast<float>(a) * degrees_per_radian);
  }
};

struct Eq {
  npy_bool operator()(posit160 a, posit160 b) { return a == b; }
};
struct Ne {
  npy_bool operator()(posit160 a, posit160 b) { return a != b; }
};
struct Lt {
  npy_bool operator()(posit160 a, posit160 b) { return a < b; }
};
struct Gt {
  npy_bool operator()(posit160 a, posit160 b) { return a > b; }
};
struct Le {
  npy_bool operator()(posit160 a, posit160 b) { return a <= b; }
};
struct Ge {
  npy_bool operator()(posit160 a, posit160 b) { return a >= b; }
};
struct Maximum {
  posit160 operator()(posit160 a, posit160 b) {
    float fa(a), fb(b);
    return Eigen::numext::isnan(fa) || fa > fb ? a : b;
  }
};
struct Minimum {
  posit160 operator()(posit160 a, posit160 b) {
    float fa(a), fb(b);
    return Eigen::numext::isnan(fa) || fa < fb ? a : b;
  }
};
struct Fmax {
  posit160 operator()(posit160 a, posit160 b) {
    float fa(a), fb(b);
    return Eigen::numext::isnan(fb) || fa > fb ? a : b;
  }
};
struct Fmin {
  posit160 operator()(posit160 a, posit160 b) {
    float fa(a), fb(b);
    return Eigen::numext::isnan(fb) || fa < fb ? a : b;
  }
};

struct LogicalNot {
  npy_bool operator()(posit160 a) { return !a; }
};
struct LogicalAnd {
  npy_bool operator()(posit160 a, posit160 b) { return a && b; }
};
struct LogicalOr {
  npy_bool operator()(posit160 a, posit160 b) { return a || b; }
};
struct LogicalXor {
  npy_bool operator()(posit160 a, posit160 b) {
    return static_cast<bool>(a) ^ static_cast<bool>(b);
  }
};

struct NextAfter {
  posit160 operator()(posit160 from, posit160 to) {
    uint16_t from_as_int, to_as_int;
    const uint16_t sign_mask = 1 << 15;
    float from_as_float(from), to_as_float(to);
    memcpy(&from_as_int, &from, sizeof(posit160));
    memcpy(&to_as_int, &to, sizeof(posit160));
    if (Eigen::numext::isnan(from_as_float) ||
        Eigen::numext::isnan(to_as_float)) {
      return posit160(std::numeric_limits<float>::quiet_NaN());
    }
    if (from_as_int == to_as_int) {
      return to;
    }
    if (from_as_float == 0) {
      if (to_as_float == 0) {
        return to;
      } else {
        // Smallest subnormal signed like `to`.
        uint16_t out_int = (to_as_int & sign_mask) | 1;
        posit160 out;
        memcpy(&out, &out_int, sizeof(posit160));
        return out;
      }
    }
    uint16_t from_sign = from_as_int & sign_mask;
    uint16_t to_sign = to_as_int & sign_mask;
    uint16_t from_abs = from_as_int & ~sign_mask;
    uint16_t to_abs = to_as_int & ~sign_mask;
    uint16_t magnitude_adjustment =
        (from_abs > to_abs || from_sign != to_sign) ? 0xFFFF : 0x0001;
    uint16_t out_int = from_as_int + magnitude_adjustment;
    posit160 out;
    memcpy(&out, &out_int, sizeof(posit160));
    return out;
  }
};

// TODO(phawkins): implement spacing

}  // namespace ufuncs

}  // namespace

// Initializes the module.
bool InitializeP160() {
  ImportNumpy();
  import_umath1(false);

  Safe_PyObjectPtr numpy_str = make_safe(PyUnicode_FromString("numpy"));
  if (!numpy_str) {
    return false;
  }
  Safe_PyObjectPtr numpy = make_safe(PyImport_Import(numpy_str.get()));
  if (!numpy) {
    return false;
  }

  // If another module (presumably either TF or JAX) has registered a posit160
  // type, use it. We don't want two posit160 types if we can avoid it since it
  // leads to confusion if we have two different types with the same name. This
  // assumes that the other module has a sufficiently complete posit160
  // implementation. The only known NumPy posit160 extension at the time of
  // writing is this one (distributed in TF and JAX).
  // TODO(phawkins): distribute the posit160 extension as its own pip package,
  // so we can unambiguously refer to a single canonical definition of posit160.
  int typenum = PyArray_TypeNumFromName(const_cast<char*>("posit160"));
  if (typenum != NPY_NOTYPE) {
    PyArray_Descr* descr = PyArray_DescrFromType(typenum);
    // The test for an argmax function here is to verify that the
    // posit160 implementation is sufficiently new, and, say, not from
    // an older version of TF or JAX.
    if (descr && descr->f && descr->f->argmax) {
      npy_posit160 = typenum;
      posit160_type_ptr = descr->typeobj;
      return true;
    }
  }

  posit160_type.tp_base = &PyGenericArrType_Type;

  if (PyType_Ready(&posit160_type) < 0) {
    return false;
  }

  // Initializes the NumPy descriptor.
  PyArray_InitArrFuncs(&NPyPosit160_ArrFuncs);
  NPyPosit160_ArrFuncs.getitem = NPyPosit160_GetItem;
  NPyPosit160_ArrFuncs.setitem = NPyPosit160_SetItem;
  NPyPosit160_ArrFuncs.compare = NPyPosit160_Compare;
  NPyPosit160_ArrFuncs.copyswapn = NPyPosit160_CopySwapN;
  NPyPosit160_ArrFuncs.copyswap = NPyPosit160_CopySwap;
  NPyPosit160_ArrFuncs.nonzero = NPyPosit160_NonZero;
  NPyPosit160_ArrFuncs.fill = NPyPosit160_Fill;
  NPyPosit160_ArrFuncs.dotfunc = NPyPosit160_DotFunc;
  NPyPosit160_ArrFuncs.compare = NPyPosit160_CompareFunc;
  NPyPosit160_ArrFuncs.argmax = NPyPosit160_ArgMaxFunc;
  NPyPosit160_ArrFuncs.argmin = NPyPosit160_ArgMinFunc;

  Py_TYPE(&NPyPosit160_Descr) = &PyArrayDescr_Type;
  npy_posit160 = PyArray_RegisterDataType(&NPyPosit160_Descr);
  posit160_type_ptr = &posit160_type;
  if (npy_posit160 < 0) {
    return false;
  }

  Safe_PyObjectPtr typeDict_obj =
      make_safe(PyObject_GetAttrString(numpy.get(), "typeDict"));
  if (!typeDict_obj) return false;
  // Add the type object to `numpy.typeDict`: that makes
  // `numpy.dtype('posit160')` work.
  if (PyDict_SetItemString(typeDict_obj.get(), "posit160",
                           reinterpret_cast<PyObject*>(&posit160_type)) < 0) {
    return false;
  }

  // Support dtype(posit160)
  if (PyDict_SetItemString(posit160_type.tp_dict, "dtype",
                           reinterpret_cast<PyObject*>(&NPyPosit160_Descr)) <
      0) {
    return false;
  }

  // Register casts
  if (!RegisterPosit160Cast<Eigen::half>(NPY_HALF)) {
    return false;
  }

  if (!RegisterPosit160Cast<float>(NPY_FLOAT)) {
    return false;
  }
  if (!RegisterPosit160Cast<double>(NPY_DOUBLE)) {
    return false;
  }
  if (!RegisterPosit160Cast<bool>(NPY_BOOL)) {
    return false;
  }
  if (!RegisterPosit160Cast<uint8>(NPY_UINT8)) {
    return false;
  }
  if (!RegisterPosit160Cast<uint16>(NPY_UINT16)) {
    return false;
  }
  if (!RegisterPosit160Cast<unsigned int>(NPY_UINT)) {
    return false;
  }
  if (!RegisterPosit160Cast<unsigned long>(NPY_ULONG)) {  // NOLINT
    return false;
  }
  if (!RegisterPosit160Cast<unsigned long long>(NPY_ULONGLONG)) {  // NOLINT
    return false;
  }
  if (!RegisterPosit160Cast<uint64>(NPY_UINT64)) {
    return false;
  }
  if (!RegisterPosit160Cast<int8>(NPY_INT8)) {
    return false;
  }
  if (!RegisterPosit160Cast<int16>(NPY_INT16)) {
    return false;
  }
  if (!RegisterPosit160Cast<int>(NPY_INT)) {
    return false;
  }
  if (!RegisterPosit160Cast<long>(NPY_LONG)) {  // NOLINT
    return false;
  }
  if (!RegisterPosit160Cast<long long>(NPY_LONGLONG)) {  // NOLINT
    return false;
  }
  // Following the numpy convention. imag part is dropped when converting to
  // float.
  /*if (!RegisterPosit160Cast<std::complex<float>>(NPY_COMPLEX64)) {
    return false;
  }
  if (!RegisterPosit160Cast<std::complex<double>>(NPY_COMPLEX128)) {
    return false;
  }*/

  // Safe casts from posit160 to other types
  if (PyArray_RegisterCanCast(&NPyPosit160_Descr, NPY_FLOAT, NPY_NOSCALAR) <
      0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&NPyPosit160_Descr, NPY_DOUBLE, NPY_NOSCALAR) <
      0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&NPyPosit160_Descr, NPY_COMPLEX64, NPY_NOSCALAR) <
      0) {
    return false;
  }
  if (PyArray_RegisterCanCast(&NPyPosit160_Descr, NPY_COMPLEX128,
                              NPY_NOSCALAR) < 0) {
    return false;
  }

  // Safe casts to posit160 from other types
  if (PyArray_RegisterCanCast(PyArray_DescrFromType(NPY_BOOL), npy_posit160,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(PyArray_DescrFromType(NPY_UINT8), npy_posit160,
                              NPY_NOSCALAR) < 0) {
    return false;
  }
  if (PyArray_RegisterCanCast(PyArray_DescrFromType(NPY_INT8), npy_posit160,
                              NPY_NOSCALAR) < 0) {
    return false;
  }

  bool ok =
      RegisterUFunc<BinaryUFunc<posit160, posit160, ufuncs::Add>>(numpy.get(),
                                                                  "add") &&
      RegisterUFunc<BinaryUFunc<posit160, posit160, ufuncs::Subtract>>(
          numpy.get(), "subtract") &&
      RegisterUFunc<BinaryUFunc<posit160, posit160, ufuncs::Multiply>>(
          numpy.get(), "multiply") &&
      RegisterUFunc<BinaryUFunc<posit160, posit160, ufuncs::TrueDivide>>(
          numpy.get(), "divide") &&
      RegisterUFunc<BinaryUFunc<posit160, posit160, ufuncs::LogAddExp>>(
          numpy.get(), "logaddexp") &&
      RegisterUFunc<BinaryUFunc<posit160, posit160, ufuncs::LogAddExp2>>(
          numpy.get(), "logaddexp2") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Negative>>(
          numpy.get(), "negative") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Positive>>(
          numpy.get(), "positive") &&
      RegisterUFunc<BinaryUFunc<posit160, posit160, ufuncs::TrueDivide>>(
          numpy.get(), "true_divide") &&
      RegisterUFunc<BinaryUFunc<posit160, posit160, ufuncs::FloorDivide>>(
          numpy.get(), "floor_divide") &&
      RegisterUFunc<BinaryUFunc<posit160, posit160, ufuncs::Power>>(numpy.get(),
                                                                    "power") &&
      RegisterUFunc<BinaryUFunc<posit160, posit160, ufuncs::Remainder>>(
          numpy.get(), "remainder") &&
      RegisterUFunc<BinaryUFunc<posit160, posit160, ufuncs::Remainder>>(
          numpy.get(), "mod") &&
      RegisterUFunc<BinaryUFunc<posit160, posit160, ufuncs::Fmod>>(numpy.get(),
                                                                   "fmod") &&
      RegisterUFunc<ufuncs::DivmodUFunc>(numpy.get(), "divmod") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Abs>>(numpy.get(),
                                                                 "absolute") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Abs>>(numpy.get(),
                                                                 "fabs") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Rint>>(numpy.get(),
                                                                  "rint") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Sign>>(numpy.get(),
                                                                  "sign") &&
      RegisterUFunc<BinaryUFunc<posit160, posit160, ufuncs::Heaviside>>(
          numpy.get(), "heaviside") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Conjugate>>(
          numpy.get(), "conjugate") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Exp>>(numpy.get(),
                                                                 "exp") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Exp2>>(numpy.get(),
                                                                  "exp2") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Expm1>>(numpy.get(),
                                                                   "expm1") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Log>>(numpy.get(),
                                                                 "log") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Log2>>(numpy.get(),
                                                                  "log2") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Log10>>(numpy.get(),
                                                                   "log10") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Log1p>>(numpy.get(),
                                                                   "log1p") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Sqrt>>(numpy.get(),
                                                                  "sqrt") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Square>>(numpy.get(),
                                                                    "square") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Cbrt>>(numpy.get(),
                                                                  "cbrt") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Reciprocal>>(
          numpy.get(), "reciprocal") &&

      // Trigonometric functions
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Sin>>(numpy.get(),
                                                                 "sin") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Cos>>(numpy.get(),
                                                                 "cos") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Tan>>(numpy.get(),
                                                                 "tan") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Arcsin>>(numpy.get(),
                                                                    "arcsin") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Arccos>>(numpy.get(),
                                                                    "arccos") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Arctan>>(numpy.get(),
                                                                    "arctan") &&
      RegisterUFunc<BinaryUFunc<posit160, posit160, ufuncs::Arctan2>>(
          numpy.get(), "arctan2") &&
      RegisterUFunc<BinaryUFunc<posit160, posit160, ufuncs::Hypot>>(numpy.get(),
                                                                    "hypot") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Sinh>>(numpy.get(),
                                                                  "sinh") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Cosh>>(numpy.get(),
                                                                  "cosh") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Tanh>>(numpy.get(),
                                                                  "tanh") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Arcsinh>>(
          numpy.get(), "arcsinh") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Arccosh>>(
          numpy.get(), "arccosh") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Arctanh>>(
          numpy.get(), "arctanh") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Deg2rad>>(
          numpy.get(), "deg2rad") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Rad2deg>>(
          numpy.get(), "rad2deg") &&

      // Comparison functions
      RegisterUFunc<BinaryUFunc<posit160, bool, ufuncs::Eq>>(numpy.get(),
                                                             "equal") &&
      RegisterUFunc<BinaryUFunc<posit160, bool, ufuncs::Ne>>(numpy.get(),
                                                             "not_equal") &&
      RegisterUFunc<BinaryUFunc<posit160, bool, ufuncs::Lt>>(numpy.get(),
                                                             "less") &&
      RegisterUFunc<BinaryUFunc<posit160, bool, ufuncs::Gt>>(numpy.get(),
                                                             "greater") &&
      RegisterUFunc<BinaryUFunc<posit160, bool, ufuncs::Le>>(numpy.get(),
                                                             "less_equal") &&
      RegisterUFunc<BinaryUFunc<posit160, bool, ufuncs::Ge>>(numpy.get(),
                                                             "greater_equal") &&
      RegisterUFunc<BinaryUFunc<posit160, posit160, ufuncs::Maximum>>(
          numpy.get(), "maximum") &&
      RegisterUFunc<BinaryUFunc<posit160, posit160, ufuncs::Minimum>>(
          numpy.get(), "minimum") &&
      RegisterUFunc<BinaryUFunc<posit160, posit160, ufuncs::Fmax>>(numpy.get(),
                                                                   "fmax") &&
      RegisterUFunc<BinaryUFunc<posit160, posit160, ufuncs::Fmin>>(numpy.get(),
                                                                   "fmin") &&
      RegisterUFunc<BinaryUFunc<posit160, bool, ufuncs::LogicalAnd>>(
          numpy.get(), "logical_and") &&
      RegisterUFunc<BinaryUFunc<posit160, bool, ufuncs::LogicalOr>>(
          numpy.get(), "logical_or") &&
      RegisterUFunc<BinaryUFunc<posit160, bool, ufuncs::LogicalXor>>(
          numpy.get(), "logical_xor") &&
      RegisterUFunc<UnaryUFunc<posit160, bool, ufuncs::LogicalNot>>(
          numpy.get(), "logical_not") &&

      // Floating point functions
      RegisterUFunc<UnaryUFunc<posit160, bool, ufuncs::IsFinite>>(numpy.get(),
                                                                  "isfinite") &&
      RegisterUFunc<UnaryUFunc<posit160, bool, ufuncs::IsInf>>(numpy.get(),
                                                               "isinf") &&
      RegisterUFunc<UnaryUFunc<posit160, bool, ufuncs::IsNan>>(numpy.get(),
                                                               "isnan") &&
      RegisterUFunc<UnaryUFunc<posit160, bool, ufuncs::SignBit>>(numpy.get(),
                                                                 "signbit") &&
      RegisterUFunc<BinaryUFunc<posit160, posit160, ufuncs::CopySign>>(
          numpy.get(), "copysign") &&
      RegisterUFunc<UnaryUFunc2<posit160, posit160, posit160, ufuncs::Modf>>(
          numpy.get(), "modf") &&
      RegisterUFunc<BinaryUFunc2<posit160, int, posit160, ufuncs::Ldexp>>(
          numpy.get(), "ldexp") &&
      RegisterUFunc<UnaryUFunc2<posit160, posit160, int, ufuncs::Frexp>>(
          numpy.get(), "frexp") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Floor>>(numpy.get(),
                                                                   "floor") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Ceil>>(numpy.get(),
                                                                  "ceil") &&
      RegisterUFunc<UnaryUFunc<posit160, posit160, ufuncs::Trunc>>(numpy.get(),
                                                                   "trunc") &&
      RegisterUFunc<BinaryUFunc<posit160, posit160, ufuncs::NextAfter>>(
          numpy.get(), "nextafter");

  return ok;
}

bool RegisterNumpyPosit160() {
  if (npy_posit160 != NPY_NOTYPE) {
    // Already initialized.
    return true;
  }
  if (!InitializeP160()) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, "cannot load posit160 module.");
    }
    PyErr_Print();
    return false;
  }
  return true;
}

PyObject* Posit160Dtype() {
  return reinterpret_cast<PyObject*>(posit160_type_ptr);
}

int Posit160NumpyType() { return npy_posit160; }

}  // namespace tensorflow
