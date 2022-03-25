/* This file contains the base class implementation for all device arrays. The
 * base class is implemented in C so that computing typecodes for device arrays
 * can be implemented efficiently. */

#include "_pymodule.h"
#include "abstract.h"
#include "ceval.h"
#include "dlpack/dlpack.h"
#include "object.h"


/* Include _devicearray., but make sure we don't get the definitions intended
 * for consumers of the Device Array API.
 */
#define NUMBA_IN_DEVICEARRAY_CPP_
#include "_devicearray.h"

/* DeviceArray PyObject implementation. Note that adding more members here is
 * presently prohibited because mapped and managed arrays derive from both
 * DeviceArray and NumPy's ndarray, which is also a C extension class - the
 * layout of the object cannot be resolved if this class also has members beyond
 * PyObject_HEAD. */
struct DeviceArray {
    PyObject_HEAD
};

/* Trivial traversal - DeviceArray instances own nothing. */
static int
DeviceArray_traverse(DeviceArray *self, visitproc visit, void *arg)
{
    return 0;
}

/* Trivial clear of all references - DeviceArray instances own nothing. */
static int
DeviceArray_clear(DeviceArray *self)
{
    return 0;
}

/*
 * This function is called when the consuming library has finished with the
 * tensor. At this point we can safely allow GC of the Numba Device Array (if
 * there are no other references).
 */
static void
DeviceArray_managed_tensor_deleter(DLManagedTensor *managed_tensor)
{
  printf("In deleter!\n");

  // Release reference to the Numba device array
  Py_DECREF(managed_tensor->manager_ctx);

  // Free other arrays from the managed tensor
  delete[] managed_tensor->dl_tensor.shape;
  delete[] managed_tensor->dl_tensor.strides;
}

/*
 * Called when the PyCapsule containing the managed_tensor is deleted. We may
 * need to call the destructor to release the reference to the device array and
 * free our shape and strides, if the capsule was never consumed by another
 * library.
 */
static void
DeviceArray_capsule_destructor(PyObject *capsule)
{
  printf("In destructor\n");

  // An unused capsule is named "dltensor", one that has been consumed is named
  // "used_dltensor".
  DLManagedTensor *dlMTensor = (DLManagedTensor *)PyCapsule_GetPointer(capsule, "dltensor");

  if (dlMTensor) {
    // The capsule was never consumed, so let's get rid of our reference
    printf("deleting unconsumed capsule\n");

    dlMTensor->deleter(dlMTensor);
  } else {
    // Nothing to do for a consumed capsule

    printf("Capsule already consumed\n");
    // PyCapsule_GetPointer has set an error indicator
    PyErr_Clear();
  }
}

/*
 * This is a debugging aid borrowed from apps/from_numpy/numpy_dlpack.c in the
 * dlpack repo. It prints out a DLManagedTensor structure.
 */
void display(DLManagedTensor a) {
  puts("On C side:");
  int i;
  int ndim = a.dl_tensor.ndim;
  printf("data = %p\n", a.dl_tensor.data);
  printf("ctx = (device_type = %d, device_id = %d)\n",
          (int) a.dl_tensor.ctx.device_type,
          (int) a.dl_tensor.ctx.device_id);
  printf("dtype = (code = %d, bits = %d, lanes = %d)\n",
          (int) a.dl_tensor.dtype.code,
          (int) a.dl_tensor.dtype.bits,
          (int) a.dl_tensor.dtype.lanes);
  printf("ndim = %d\n",
          (int) a.dl_tensor.ndim);
  printf("shape = (");
  for (i = 0; i < ndim; ++i) {
    if (i != 0) {
      printf(", ");
    }
    printf("%d", (int) a.dl_tensor.shape[i]);
  }
  printf(")\n");
  printf("strides = (");
  for (i = 0; i < ndim; ++i) {
    if (i != 0) {
      printf(", ");
    }
    printf("%d", (int) a.dl_tensor.strides[i]);
  }
  printf(")\n");
}


static PyObject*
DeviceArray_to_dlpack(PyObject *self, PyObject *args)
{
  // Allocate arrays for shape and stride
  int ndim = PyLong_AsLong(PyObject_GetAttrString(self, "ndim"));
  int64_t *shape = new int64_t[ndim];
  int64_t *strides = new int64_t[ndim];

  // The notion of stride differs between NumPy and dlpack.
  // - NumPy:  stride in bytes
  // - DLPack: stride in elements
  //
  // So to calculate the DLPack stride, we need the itemsize.
  int itemsize = PyLong_AsLong(PyObject_GetAttrString(
        PyObject_GetAttrString(self, "dtype"),
        "itemsize"));

  // Populate the DLPack shape and strides fields
  PyObject *py_shape = PyObject_GetAttrString(self, "shape");
  PyObject *py_strides = PyObject_GetAttrString(self, "strides");

  for (auto i = 0; i < ndim; i++) {
    shape[i] = PyLong_AsLong(PyTuple_GetItem(py_shape, i));
    strides[i] = PyLong_AsLong(PyTuple_GetItem(py_strides, i)) / itemsize;
  }

  // Get the pointer to the underlying data, stored in `ptr` - mostly just
  // a case of traversing the Numba Device Array attributes.
  PyObject *gpu_data = PyObject_GetAttrString(self, "gpu_data");
  if (!gpu_data)
    return nullptr;

  PyObject *numba = PyImport_ImportModule("numba");
  PyObject *cuda = PyObject_GetAttrString(numba, "cuda");
  PyObject *cudadrv = PyObject_GetAttrString(cuda, "cudadrv");
  PyObject *driver = PyObject_GetAttrString(cudadrv, "driver");
  PyObject *MemoryPointer = PyObject_GetAttrString(driver, "MemoryPointer");

  // If we have an OwnedPointer (e.g. from Numba's memory manager) we need to
  // get the underlying memory. If we have a MemoryPointer (from RMM's EMM
  // plugin) then that's already what we need.
  PyObject *mem;
  if (PyObject_IsInstance(gpu_data, MemoryPointer))
    mem = gpu_data;
  else
    mem = PyObject_GetAttrString(gpu_data, "_mem");

  if (!mem)
    return nullptr;

  PyObject *handle = PyObject_GetAttrString(mem, "handle");
  if (!handle)
    return nullptr;

  uintptr_t ptr = PyLong_AsLongLong(PyObject_GetAttrString(handle, "value"));

  // Create and populated the DLManagedTensor that we're going to return
  DLManagedTensor *managed_tensor = new DLManagedTensor;
  managed_tensor->dl_tensor.data = (void*)ptr;
  managed_tensor->dl_tensor.byte_offset = 0;          // TODO: Need to work this out - e.g. for views?
  managed_tensor->dl_tensor.ctx.device_type = kDLGPU; // TODO: Support for pinned / mapped / managed memory
  managed_tensor->dl_tensor.ctx.device_id = 0;        // TODO: Use actual device ID, not hardcoded 0
  managed_tensor->dl_tensor.dtype.code = kDLFloat;    // TODO: Use dtype code based on actual type, not hardcoded
  managed_tensor->dl_tensor.dtype.bits = 64;          // TODO: Use bits based on actual type, not hardcoded
  managed_tensor->dl_tensor.dtype.lanes = 1;          // TODO: Support for homogeneous structured dtypes with lanes?
  managed_tensor->dl_tensor.ndim = ndim;
  managed_tensor->dl_tensor.shape = shape;
  managed_tensor->dl_tensor.strides = strides;
  managed_tensor->deleter = DeviceArray_managed_tensor_deleter;
  managed_tensor->manager_ctx = self;

  // Debug print
  display(*managed_tensor);

  // We don't want the Device Array to go out of scope prior to a consumer
  // finishing with it, so increment the reference count here. The corresponding
  // decref is in the deleter, which will be called by the consuming library
  // once it has finished with the tensor, or the capsule destructor for an
  // unconsumed tensor.
  Py_INCREF(self);

  // DLPack tensors must be named "dltensor"
  return PyCapsule_New((void *)managed_tensor, "dltensor", DeviceArray_capsule_destructor);
}

static PyMethodDef DeviceArray_methods[] = {
    { "to_dlpack", (PyCFunction)DeviceArray_to_dlpack, METH_NOARGS, NULL },
    { NULL },
};


/* The _devicearray.DeviceArray type */
PyTypeObject DeviceArrayType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_devicearray.DeviceArray",                  /* tp_name */
    sizeof(DeviceArray),                         /* tp_basicsize */
    0,                                           /* tp_itemsize */
    0,                                           /* tp_dealloc */
    0,                                           /* tp_print */
    0,                                           /* tp_getattr */
    0,                                           /* tp_setattr */
    0,                                           /* tp_compare */
    0,                                           /* tp_repr */
    0,                                           /* tp_as_number */
    0,                                           /* tp_as_sequence */
    0,                                           /* tp_as_mapping */
    0,                                           /* tp_hash */
    0,                                           /* tp_call*/
    0,                                           /* tp_str*/
    0,                                           /* tp_getattro*/
    0,                                           /* tp_setattro*/
    0,                                           /* tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
                                                 /* tp_flags*/
    "DeviceArray object",                        /* tp_doc */
    (traverseproc) DeviceArray_traverse,         /* tp_traverse */
    (inquiry) DeviceArray_clear,                 /* tp_clear */
    0,                                           /* tp_richcompare */
    0,                                           /* tp_weaklistoffset */
    0,                                           /* tp_iter */
    0,                                           /* tp_iternext */
    DeviceArray_methods,                         /* tp_methods */
    0,                                           /* tp_members */
    0,                                           /* tp_getset */
    0,                                           /* tp_base */
    0,                                           /* tp_dict */
    0,                                           /* tp_descr_get */
    0,                                           /* tp_descr_set */
    0,                                           /* tp_dictoffset */
    0,                                           /* tp_init */
    0,                                           /* tp_alloc */
    0,                                           /* tp_new */
    0,                                           /* tp_free */
    0,                                           /* tp_is_gc */
    0,                                           /* tp_bases */
    0,                                           /* tp_mro */
    0,                                           /* tp_cache */
    0,                                           /* tp_subclasses */
    0,                                           /* tp_weaklist */
    0,                                           /* tp_del */
    0,                                           /* tp_version_tag */
    0,                                           /* tp_finalize */
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION == 8
    0,                                           /* tp_vectorcall */
    0,                                           /* tp_print */
#endif
};

/* CUDA device array C API */
static void *_DeviceArray_API[1] = {
    (void*)&DeviceArrayType
};

MOD_INIT(_devicearray) {
    PyObject *m = nullptr;
    PyObject *d = nullptr;
    PyObject *c_api = nullptr;
    int error = 0;

    MOD_DEF(m, "_devicearray", "No docs", NULL)
    if (m == NULL)
        goto error_occurred;

    c_api = PyCapsule_New((void *)_DeviceArray_API, "numba._devicearray._DEVICEARRAY_API", NULL);
    if (c_api == NULL)
        goto error_occurred;

    DeviceArrayType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&DeviceArrayType) < 0)
        goto error_occurred;

    Py_INCREF(&DeviceArrayType);
    error = PyModule_AddObject(m, "DeviceArray", (PyObject*)(&DeviceArrayType));
    if (error)
        goto error_occurred;

    d = PyModule_GetDict(m);
    if (d == NULL)
        goto error_occurred;

    error = PyDict_SetItemString(d, "_DEVICEARRAY_API", c_api);
    Py_DECREF(c_api);

    if (error)
        goto error_occurred;

    return MOD_SUCCESS_VAL(m);

error_occurred:
    Py_XDECREF(m);
    Py_XDECREF(c_api);
    Py_XDECREF((PyObject*)&DeviceArrayType);

    return MOD_ERROR_VAL;
}
