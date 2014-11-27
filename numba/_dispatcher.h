#ifndef NUMBA_DISPATCHER_H_
#define NUMBA_DISPATCHER_H_

#include "_pymodule.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

/* Hack whilst debugging. If you see this in a pull request please berate me. */
#ifdef __cplusplus
#define NO_IMPORT_ARRAY
#endif

#include <numpy/ndarrayobject.h>


#ifdef __cplusplus
    extern "C" {
#endif

typedef struct _opaque_dispatcher dispatcher_t;

dispatcher_t *
dispatcher_new(void *tm, int argct);

void
dispatcher_del(dispatcher_t *obj);

void
dispatcher_add_defn(dispatcher_t *obj, int tys[], void* callable);

void*
dispatcher_resolve(dispatcher_t *obj, int sig[], int *matches,
                   int allow_unsafe);

int
dispatcher_count(dispatcher_t *obj);

int dispatcher_get_ndarray_typecode(int ndim, int layout, PyArray_Descr* descr);

void dispatcher_insert_ndarray_typecode(int ndim, int layout,
                                        PyArray_Descr* descr, int typecode);

int dispatcher_get_arrayscalar_typecode(PyArray_Descr* descr);

void dispatcher_insert_arrayscalar_typecode(PyArray_Descr* descr, int typecode);

#ifdef __cplusplus
    }
#endif

#endif  /* NUMBA_DISPATCHER_H_ */
