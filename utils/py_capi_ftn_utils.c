
#include "py_capi_ftn_utils.h"

void getPyFtnArrayDims(int dims[4],PyArrayObject *array){
    int i;
    for (i=0; i < 4; i++) dims[i] = 1;
    for (i=0; i < ((array->nd <= 4) ? array->nd : 4); i++)
        dims[i] = (array->dimensions[i] > 0) ? array->dimensions[i] : 1;
}

void getPyFtnArrayDataTypeAndLen(int *dataType, int *dataLen,PyArrayObject *array) {
    dataType[0] = -1;
    dataLen[0]   = -1;
    switch (array->descr->type_num) {
        case NPY_INT:
            dataType[0]=4;
            dataLen[0]=4;
            break;
        case NPY_LONG:
            if (sizeof(long)!=4)
                fprintf(stderr,"WARNING: Sizeof(long)=%d\n",(int)sizeof(long));
            dataType[0]=4;
            dataLen[0]=4;
            break;
        case NPY_SHORT:
            dataType[0]=4;
            dataLen[0]=2;
            break;
        case NPY_FLOAT:
            dataType[0]=134;
            dataLen[0]=4;
            break;
        case NPY_DOUBLE:
            dataType[0]=1;
            dataLen[0]=8;
            break;
        default:
            fprintf(stderr,"ERROR: Unsupported data type :%c\n",array->descr->type);
            break;
    }
}

int isPyFtnArrayValid(PyArrayObject *array,int requestedDataType){
    int istat = 0,dataType,dataLen;
    if (!((PyArray_ISCONTIGUOUS(array) || (array->flags & NPY_FARRAY)) && array->nd > 0 && array->dimensions[0] > 0)) {
        fprintf(stderr,"ERROR: Array is not CONTIGUOUS in memory\n");
        istat = -1;
    } else {
        getPyFtnArrayDataTypeAndLen(&dataType,&dataLen,array);
        if (dataType<0 || (requestedDataType>0 && requestedDataType!=(dataType+dataLen))) istat = -1;
    }
    return istat;
}
