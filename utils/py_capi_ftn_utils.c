
#include "py_capi_ftn_utils.h"

void getPyFtnArrayDims(int dims[4],PyArrayObject *array){
    int i;
    for (i=0; i < 4; i++) dims[i] = 1;
    for (i=0; i < ((array->nd <= 4) ? array->nd : 4); i++)
        dims[i] = (array->dimensions[i] > 0) ? array->dimensions[i] : 1;
}

void getPyFtnArrayDataTypeAndLen(int *dataType, int *dataLen,PyArrayObject *array,int *nbits) {
   /* The standard format is a bit peculiar about how it handles various data lengths, for
      floating point data:

         1) Double-precision (uncompressed) data is supported, but it requires setting
            nbits = 64 upon writing.  nbits <= 32 will write the data out as single-
            precision, which will treat the array in an incorrect manner.
         2) For nbits=32, the 'real' format (1/134) does not compress the data, whereas
            the 'ieee' format (5/133) does.
         3) For nbits < 32, the 'ieee' format drops precision by simply truncating the
            mantissa; this is a serious loss of accuracy (nbits=8 corresponds to not
            even a full order of magnitude; nbits=16 gives three digits of accuracy).
            Instead, it's important that the 'real' format (which is smarter here) be
            used for nbits < 32.

      Consequently, the dataType depends both on the type of the underlying object and
      the number of bits of output, at least for floating point data types.  This
      function, however, is called both from the fstecr routine (to get the dtype
      parameter) and from pyFtnArrayValid, which merely checks to see if the array -could-
      be written with suitable parameters.  Checking up on nbits is appropriate in the
      first place (with modifications possible, since many of the errors above are fixable),
      but there's no guarantee that nbits is set appropriately from ArrayValid.  Instead,
      that function will pass in a null pointer, and checks here will first check for its
      validity.  This gives the best of both worlds. */
    //fprintf(stderr,"INFO: datyp in=%d\n",dataType[0]);
    dataLen[0]  = -1;
    switch (array->descr->type_num) {
        case NPY_INT:
            if (!(dataType[0] == 0 || dataType[0] == 2 || 
                  dataType[0] == 4 || dataType[0] == 130 || 
                  dataType[0] == 132)) {
                dataType[0]=4;
            }
            dataLen[0]=4;
            break;
        case NPY_LONG:
            if (sizeof(long)!=4)
                fprintf(stderr,"WARNING: Sizeof(long)=%d\n",(int)sizeof(long));
            if (!(dataType[0] == 0 || dataType[0] == 2 || 
                  dataType[0] == 4 || dataType[0] == 130 || 
                  dataType[0] == 132)) {
                dataType[0]=4;
            }
            dataLen[0]=4;
            break;
        case NPY_SHORT:
            dataType[0]=4;
            dataLen[0]=2;
            break;
        case NPY_FLOAT:
            if (!(dataType[0] == 1 || dataType[0] == 5 || 
                  dataType[0] == 6 || dataType[0] == 134 || 
                  dataType[0] == 133)) {
                /* Logic for data-type assignment, if one is not already given:
                 * If nbits > 32 and we're given an NPY_FLOAT, then something
                 is inconsistent; output a warning and continue as if nbits
                 was 32. */
                if (nbits && *nbits > 32) {
                    fprintf(stderr,"WARNING: Trying to use nbits > 32 on single-precision data\n");
                    *nbits = 32;
                }
                /* If nbits == 32, then use the IEEE-format output (E32/e32) */
                if (nbits && *nbits == 32) {
                    /* 2D arrays larger than 16x16 can be compressed */
                    if (array->nd >= 2 &&
                            array->dimensions[0] >= 16 &&
                            array->dimensions[1] >= 16) {
                        /* e32 format on voir */
                        dataType[0] = 133;
                    } else {
                        /* E32 format on voir */
                        dataType[0] = 5;
                    }
                } else {
                    /* If nbits < 32, then reduced-precision otuput is desired.  This
                       is semantically valid for e# format, but it truncates a great
                       deal of precision.  Instead, use 'real'-type output */
                    /* 2D arrays larger than 16 x 16 can be compressed (f[#])*/
                    if (array->nd >= 2 &&
                            array->dimensions[0] >= 16 &&
                            array->dimensions[1] >= 16) {
                        dataType[0] = 134;
                    } else {
                        /* Otherwise, use R[#] format */
                        dataType[0] = 1;
                    }
                }
            }
            dataLen[0]=4;
            break;
        case NPY_DOUBLE:
            /* This format only makes sense with IEEE floats.  Ordinarily, this
               should check for array dimensions/compressibility.  Attempting
               to write out with format 1333 does give a file with a seemingly-
               valid header, but the fst utilities all segfault when reading that
               data.  Therefore, write out strictly uncompressed 64-bit data. */
            /* For double-precision output, nbits must be >= 32 */
            if (nbits && *nbits <= 32) {
                fprintf(stderr,"ERROR: Trying to use nbits <= 32 on double-precision data\n");
                *nbits = 64;
            }
            /* E64 format on voir */
            dataType[0] = 5;
            dataLen[0]=8;
            break;
        //TODO: case NPY_CHAR, dataType[0] == 3;dataLen[0]=?
        default:
            fprintf(stderr,"ERROR: Unsupported data type :%c\n",array->descr->type);
            // dataType[0] should now be set to -1 as an error condition, see ArrayValid
            dataType[0]=-1;
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
        getPyFtnArrayDataTypeAndLen(&dataType,&dataLen,array,0); // call with null pointer for nbits
        if (dataType<0 || (requestedDataType>0 && requestedDataType!=(dataType+dataLen))) istat = -1;
    }
    return istat;
}
