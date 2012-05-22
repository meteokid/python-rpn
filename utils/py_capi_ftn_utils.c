
#include "py_capi_ftn_utils.h"

void getPyFtnArrayDims(int dims[4],PyArrayObject *array){
    int i;
    for (i=0; i < 4; i++) dims[i] = 1;
    for (i=0; i < ((array->nd <= 4) ? array->nd : 4); i++)
        dims[i] = (array->dimensions[i] > 0) ? array->dimensions[i] : 1;
}

void getPyFtnArrayDataTypeAndLen(int *dataType, int *dataLen,PyArrayObject *array) {
  fprintf(stderr,"INFO: datyp in=%d\n",dataType[0]);

    //dataType[0] = -1;
    dataLen[0] = -1;
    switch (array->descr->type_num) {
        case NPY_INT:
          if (!(dataType[0] == 0 || dataType[0] == 2 || dataType[0] == 4 || dataType[0] == 130 || dataType[0] == 132)) dataType[0]=4;
            dataLen[0]=4;
            break;
        case NPY_LONG:
            if (sizeof(long)!=4)
                fprintf(stderr,"WARNING: Sizeof(long)=%d\n",(int)sizeof(long));
            if (!(dataType[0] == 0 || dataType[0] == 2 || dataType[0] == 4 || dataType[0] == 130 || dataType[0] == 132)) dataType[0]=4;
            dataLen[0]=4;
            break;
        case NPY_SHORT:
            dataType[0]=4;
            dataLen[0]=2;
            break;
        case NPY_FLOAT:
  //TODO: Do not force dataType[0] if already set/consistent with array type, see Fstdc_fstprm

          if (!(dataType[0] == 1 || dataType[0] == 5 || dataType[0] == 6 || dataType[0] == 134 || dataType[0] == 133)) dataType[0]=134;
            dataLen[0]=4;
            break;
        case NPY_DOUBLE:
            if (!(dataType[0] == 1 || dataType[0] == 5 || dataType[0] == 6 || dataType[0] == 134 || dataType[0] == 133)) dataType[0]=1;
            dataLen[0]=8;
            break;
            //TODO: case NPY_CHAR, dataType[0] == 3;dataLen[0]=?
        default:
            fprintf(stderr,"ERROR: Unsupported data type :%c\n",array->descr->type);
            break;
    }
  fprintf(stderr,"INFO: datyp out=%d\n",dataType[0]);

}

int isPyFtnArrayValid(PyArrayObject *array,int requestedDataType){
    int istat = 0,dataType=-1,dataLen;
    if (!((PyArray_ISCONTIGUOUS(array) || (array->flags & NPY_FARRAY)) && array->nd > 0 && array->dimensions[0] > 0)) {
        fprintf(stderr,"ERROR: Array is not CONTIGUOUS in memory\n");
        istat = -1;
    } else {
        getPyFtnArrayDataTypeAndLen(&dataType,&dataLen,array);
        if (dataType<0 || (requestedDataType>0 && requestedDataType!=(dataType+dataLen))) istat = -1;
    }
    return istat;
}
