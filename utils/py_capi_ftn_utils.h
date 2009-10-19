
#include <rpnmacros.h>
#include <Python.h>
#include <numpy/arrayobject.h>

#if !defined(__PY_CAPI_FTN_UTILS__)
#define __PY_CAPI_FTN_UTILS__

#define FTN_Style_Array NPY_FARRAY

#define PYCAPIFTN_ERR -1
#define PYCAPIFTN_OK 0

#define RPN_DT_ANY -1
#define RPN_DT_SHORT 6
#define RPN_DT_INT 8
#define RPN_DT_LONG 8
#define RPN_DT_FLOAT 138
#define RPN_DT_DOUBLE 9

void getPyFtnArrayDims(int dims[4],PyArrayObject *array);
void getPyFtnArrayDataTypeAndLen(int *datyp, int *dtl,PyArrayObject *array);
int isPyFtnArrayValid(PyArrayObject *array,int requestedDataType);

#endif
