#include <stdlib.h>
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "vsl_random.h"

static VSLRandom mr;

static PyObject* randn(PyObject *self, PyObject *args, PyObject *kw)
{
    PyArrayObject* arr = NULL;
    PyObject *shape = NULL;
    static char *keywords[] = {"shape", NULL};
    NpyIter *arr_iter;
    NpyIter_IterNextFunc *arr_iternext;
    double **arr_dataptr;
    npy_intp *arr_dims;
    double *out;
    int tuple_size = 0;
    PyObject *tuple_item = NULL;
    long tuple_item_long = 0;
    long random_length = 1;
    int i;
    if (!PyArg_ParseTupleAndKeywords((PyObject*)args, kw, "O", keywords, &shape)) {
        return NULL;
    }
    if (PyInt_Check(shape)) {
        random_length = PyInt_AsLong(shape);
        arr_dims = (npy_intp*)malloc(sizeof(npy_intp));
        arr_dims[0] = random_length;
        tuple_size = 1;
    } else {
        tuple_size = PyTuple_Size(shape);
        arr_dims = (npy_intp*)malloc(tuple_size*sizeof(npy_intp));
        for (i = 0; i < tuple_size; ++i) {
			tuple_item = PyTuple_GetItem(shape, i);
            if (PyInt_Check(tuple_item) || PyLong_Check(tuple_item)) {
                tuple_item_long = PyInt_AsLong(tuple_item);
                arr_dims[i] = tuple_item_long;
                random_length *= tuple_item_long;
            } else {
                return NULL;
            }
        }
    }
    arr = (PyArrayObject*)PyArray_SimpleNew(tuple_size, arr_dims, NPY_DOUBLE);
    out = new double[random_length];
    mr.randn(random_length, out);
    arr_iter = NpyIter_New(arr, NPY_ITER_READWRITE, NPY_FORTRANORDER,
                            NPY_NO_CASTING, NULL);
    if (arr_iter == NULL) {
        return NULL;
    }
    arr_iternext = NpyIter_GetIterNext(arr_iter, NULL);
    if (arr_iternext == NULL) {
        NpyIter_Deallocate(arr_iter);
        return NULL;
    }
    arr_dataptr = (double **) NpyIter_GetDataPtrArray(arr_iter);
    i = 0;
    do {
        **arr_dataptr = out[i++];
    } while(arr_iternext(arr_iter));
    delete[] out;
    NpyIter_Deallocate(arr_iter);
    //Py_INCREF(arr);
    free(arr_dims);
    return (PyObject*)arr;
}

static PyObject* rand(PyObject *self, PyObject *args, PyObject *kw)
{
    PyArrayObject* arr = NULL;
    PyObject *shape = NULL;
    static char *keywords[] = {"shape", NULL};
    NpyIter *arr_iter;
    NpyIter_IterNextFunc *arr_iternext;
    double **arr_dataptr;
    npy_intp *arr_dims;
    double *out;
    int tuple_size = 0;
    PyObject *tuple_item = NULL;
    long tuple_item_long = 0;
    long random_length = 1;
    int i;
    if (!PyArg_ParseTupleAndKeywords((PyObject*)args, kw, "O", keywords, &shape)) {
        return NULL;
    }
    if (PyInt_Check(shape)) {
        random_length = PyInt_AsLong(shape);
        arr_dims = (npy_intp*)malloc(sizeof(npy_intp));
        arr_dims[0] = random_length;
        tuple_size = 1;
    } else {
        tuple_size = PyTuple_Size(shape);
        arr_dims = (npy_intp*)malloc(tuple_size*sizeof(npy_intp));
        for (i = 0; i < tuple_size; ++i) {
			tuple_item = PyTuple_GetItem(shape, i);
            if (PyInt_Check(tuple_item) || PyLong_Check(tuple_item)) {
                tuple_item_long = PyInt_AsLong(tuple_item);
                arr_dims[i] = tuple_item_long;
                random_length *= tuple_item_long;
            } else {
                return NULL;
            }
        }
    }
    arr = (PyArrayObject*)PyArray_SimpleNew(tuple_size, arr_dims, NPY_DOUBLE);
    out = new double[random_length];
    mr.rand(random_length, out);
    arr_iter = NpyIter_New(arr, NPY_ITER_READWRITE, NPY_FORTRANORDER,
                            NPY_NO_CASTING, NULL);
    if (arr_iter == NULL) {
        return NULL;
    }
    arr_iternext = NpyIter_GetIterNext(arr_iter, NULL);
    if (arr_iternext == NULL) {
        NpyIter_Deallocate(arr_iter);
        return NULL;
    }
    arr_dataptr = (double **) NpyIter_GetDataPtrArray(arr_iter);
    i = 0;
    do {
        **arr_dataptr = out[i++];
    } while(arr_iternext(arr_iter));
    delete[] out;
    NpyIter_Deallocate(arr_iter);
    //Py_INCREF(arr);
    free(arr_dims);
    return (PyObject*)arr;
}

static PyObject* rng(PyObject *self, PyObject *args, PyObject *kw)
{
    int seed;
    static char *keywords[] = {"seed", NULL};
    if (!PyArg_ParseTupleAndKeywords((PyObject*)args, kw, "i", keywords, &seed)) {
        return NULL;
    }
    mr.rng(seed);
    return Py_BuildValue("");
}

static PyMethodDef Methods[] =
{
     {"randn", (PyCFunction)randn, METH_VARARGS | METH_KEYWORDS,
         "return ndarray of normally distributed variables using parallel VSL & OpenMP algorithms"},
     {"rand", (PyCFunction)randn, METH_VARARGS | METH_KEYWORDS,
         "return ndarray of normally distributed variables using parallel VSL & OpenMP algorithms"},
     {"rng", (PyCFunction)rng, METH_VARARGS | METH_KEYWORDS,
         "seed the random number generator"},
     {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initvsl_random(void)
{
     (void) Py_InitModule("vsl_random", Methods);
     /* IMPORTANT: this must be called */
     import_array();
}
