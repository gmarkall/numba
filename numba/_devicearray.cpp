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

struct DeviceArray {
    PyObject_HEAD
};

static int
DeviceArray_traverse(DeviceArray *self, visitproc visit, void *arg)
{
    return 0;
}

static int
DeviceArray_clear(DeviceArray *self)
{
    return 0;
}

static void
DeviceArray_managed_tensor_deleter(DLManagedTensor *managed_tensor)
{
  printf("In deleter!\n");
  Py_DECREF(managed_tensor->manager_ctx);
  delete[] managed_tensor->dl_tensor.shape;
  delete[] managed_tensor->dl_tensor.strides;
}

static void
DeviceArray_capsule_destructor(PyObject *capsule)
{
  printf("In destructor\n");
  DLManagedTensor *dlMTensor = (DLManagedTensor *)PyCapsule_GetPointer(capsule, "dltensor");
  if (dlMTensor) {
    printf("deleting unconsumed capsule\n");
    dlMTensor->deleter(dlMTensor);
  } else {
    printf("Capsule already consumed\n");
    // PyCapsule_GetPointer has set an error indicator
    PyErr_Clear();
  }
}

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
  int ndim = PyLong_AsLong(PyObject_GetAttrString(self, "ndim"));
  int64_t *shape = new int64_t[ndim];
  int64_t *strides = new int64_t[ndim];

  PyObject *py_shape = PyObject_GetAttrString(self, "shape");
  PyObject *py_strides = PyObject_GetAttrString(self, "strides");
  int itemsize = PyLong_AsLong(PyObject_GetAttrString(
        PyObject_GetAttrString(self, "dtype"),
        "itemsize"));

  for (auto i = 0; i < ndim; i++) {
    shape[i] = PyLong_AsLong(PyTuple_GetItem(py_shape, i));
    strides[i] = PyLong_AsLong(PyTuple_GetItem(py_strides, i)) / itemsize;
  }

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

  DLManagedTensor *managed_tensor = new DLManagedTensor;
  managed_tensor->dl_tensor.data = (void*)ptr;
  managed_tensor->dl_tensor.byte_offset = 0;
  managed_tensor->dl_tensor.ctx.device_type = kDLGPU;
  managed_tensor->dl_tensor.ctx.device_id = 0;
  managed_tensor->dl_tensor.dtype.code = kDLFloat;
  managed_tensor->dl_tensor.dtype.bits = 64;
  managed_tensor->dl_tensor.dtype.lanes = 1;
  managed_tensor->dl_tensor.ndim = ndim;
  managed_tensor->dl_tensor.shape = shape;
  managed_tensor->dl_tensor.strides = strides;
  managed_tensor->deleter = DeviceArray_managed_tensor_deleter;
  managed_tensor->manager_ctx = self;

  display(*managed_tensor);

  Py_INCREF(self);

  return PyCapsule_New((void *)managed_tensor, "dltensor", DeviceArray_capsule_destructor);
}

static PyMethodDef DeviceArray_methods[] = {
    { "to_dlpack", (PyCFunction)DeviceArray_to_dlpack, METH_NOARGS, NULL },
    { NULL },
};


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
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION > 7
    0,                                           /* tp_vectorcall */
    0,                                           /* tp_print */
#endif
};

/* CUDA device array API */
static void *_DeviceArray_API[1] = {
    (void*)&DeviceArrayType
};

MOD_INIT(_devicearray) {
    PyObject *m, *d;
    MOD_DEF(m, "_devicearray", "No docs", NULL)
    if (m == NULL)
        return MOD_ERROR_VAL;

    PyObject *c_api;
    c_api = PyCapsule_New((void *)_DeviceArray_API, NULL, NULL);
    if (c_api == NULL) {
        return MOD_ERROR_VAL;
    }

    DeviceArrayType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&DeviceArrayType) < 0) {
        return MOD_ERROR_VAL;
    }
    Py_INCREF(&DeviceArrayType);
    PyModule_AddObject(m, "DeviceArray", (PyObject*)(&DeviceArrayType));

    d = PyModule_GetDict(m);
    if (d == NULL) {
        return MOD_ERROR_VAL;
    }

    PyDict_SetItemString(d, "_DEVICEARRAY_API", c_api);
    Py_DECREF(c_api);
    if (PyErr_Occurred()) {
        return MOD_ERROR_VAL;
    }

    return MOD_SUCCESS_VAL(m);
}
