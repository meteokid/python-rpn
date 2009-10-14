/*
Module Fstdc contains the classes used to access RPN Standard Files (rev 2000)
@author: Mario Lepine <mario.lepine@ec.gc.ca>
@author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
@date: 2009-09
*/
/* #define DEBUG On */
#include <rpnmacros.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "rpn_version.h"

static char version[] = VERSION;
static char lastmodified[] = LASTUPDATE;

int c_fst_data_length(int length_type);

static void imprime_ca(char *varname, float *array, int nb);

/* # include "numpy/oldnumeric.h" */

#define FTN_Style_Array NPY_FARRAY
//#define FTN_Style_Array 1
static PyObject *ErrorObject;

static int datyps[32]={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
                       -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
static int lentab[32]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

/* initialize the table giving the length of the array element types */
static void init_lentab() {
	lentab[NPY_CHAR]=sizeof(char);
	lentab[NPY_UBYTE]=sizeof(unsigned char);
	lentab[NPY_BYTE]=sizeof(signed char);
	lentab[NPY_SHORT]=sizeof(short);
	lentab[NPY_INT]=sizeof(int);
	lentab[NPY_LONG]=sizeof(long);
	lentab[NPY_FLOAT]=sizeof(float);
	lentab[NPY_DOUBLE]=sizeof(double);
	lentab[NPY_CFLOAT]=2*sizeof(float);
	lentab[NPY_CDOUBLE]=2*sizeof(double);
	lentab[NPY_OBJECT]=sizeof(PyObject *);

	datyps[NPY_CHAR]=-1;
	datyps[NPY_UBYTE]=-1;
	datyps[NPY_BYTE]=-1;
	datyps[NPY_SHORT]=-1;
	datyps[NPY_INT]=2;
	datyps[NPY_LONG]=2;
	datyps[NPY_FLOAT]=1;
	datyps[NPY_DOUBLE]=-1;
	datyps[NPY_CFLOAT]=-1;
	datyps[NPY_CDOUBLE]=-1;
	datyps[NPY_OBJECT]=-1;
}
/* ----------------------------------------------------- */
#if defined (mips) || defined (__mips)
void __dshiftr4() {
    fprintf(stderr,"ERROR: __dshiftr4 called\n");
    exit(1);
}
void __mask4() {
    fprintf(stderr,"ERROR: __mask4 called\n");
    exit(1);
}
void __dshiftl4() {
    fprintf(stderr,"ERROR: __dshiftl4 called\n");
    exit(1);
}
#endif

void imprime_ca(char *varname, float *tabl, int nb) {
  int i;

  for (i=0; i < nb; i++)
     printf("%s[%d]=%f\n",varname,i,tabl[i]);
}


static PyObject *
c2py_fstprm(int handle) {
    int ni=0, nj=0, nk=0, ip1=0, ip2=0, ip3=0;
    char TYPVAR[3]={' ',' ','\0'};
    char NOMVAR[5]={' ',' ',' ',' ','\0'};
    char ETIKET[13]={' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','\0'};
    char GRTYP[2]={' ','\0'};
    int dateo=0, deet=0, npas=0, nbits=0, datyp=0, ig1=0, ig2=0, ig3=0, ig4=0;
    int swa=0, lng=0, dltf=0, ubc=0, extra1=0, extra2=0, extra3=0;
    int junk;

    if(handle >= 0) {
        junk =  c_fstprm(handle,&dateo,&deet,&npas,&ni,&nj,&nk,
                &nbits,&datyp,&ip1,&ip2,&ip3,
                TYPVAR,NOMVAR,ETIKET,GRTYP,&ig1,&ig2,&ig3,&ig4,
                &swa,&lng,&dltf,&ubc,&extra1,&extra2,&extra3);
    }
    if(junk < 0 || handle < 0) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    return Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:s,s:s,s:s,s:s,s:i,s:i,s:i,s:i}",
            "handle",handle,"ni",ni,"nj",nj,"nk",nk,"dateo",dateo,"ip1",ip1,"ip2",ip2,"ip3",ip3,
            "deet",deet,"npas",npas,"datyp",datyp,"nbits",nbits,
            "type",TYPVAR,"nom",NOMVAR,"etiket",ETIKET,"grtyp",GRTYP,
            "ig1",ig1,"ig1",ig2,"ig3",ig3,"ig4",ig4,"datev",extra1);
}


void getPyArrayDims(int dims[4],PyArrayObject *array){
    int i;
    for (i=0; i < 4; i++) dims[i] = 1;
    for (i=0; i < ((array->nd <= 4) ? array->nd : 4); i++)
        dims[i] = (array->dimensions[i] > 0) ? array->dimensions[i] : 1;
}


void getPyArrayDatyp(int *datyp, int *dtl,PyArrayObject *array) {
    datyp[0] = -1;
    dtl[0]   = -1;
    switch (array->descr->type_num) {
        case NPY_INT:
            datyp[0]=4;
            dtl[0]=4;
            break;
        case NPY_LONG:
            if (sizeof(long)!=4)
                fprintf(stderr,"WARNING: Fstdc - sizeof(long)=%d\n",sizeof(long));
            datyp[0]=4;
            dtl[0]=4;
            break;
        case NPY_SHORT:
            datyp[0]=4;
            dtl[0]=2;
            break;
        case NPY_FLOAT:
            datyp[0]=134;
            dtl[0]=4;
            break;
        case NPY_DOUBLE:
            datyp[0]=1;
            dtl[0]=8;
            break;
        default:
            fprintf(stderr,"ERROR: Fstdc - unsupported data type :%c\n",array->descr->type);
            break;
    }
}


int isPyArrayValid(PyArrayObject *array){
    int istat = 0,datyp,dtl;
    if (!((PyArray_ISCONTIGUOUS(array) || (array->flags & NPY_FARRAY)) && array->nd > 0 && array->dimensions[0] > 0)) {
        fprintf(stderr,"ERROR: Fstdc - array is not CONTIGUOUS in memory\n");
        istat = -1;
    } else {
        getPyArrayDatyp(&datyp,&dtl,array);
        if (datyp<0) istat = -1;
    }
    return istat;
}


int isGridTypeValid(char *grtyp) {
    int istat = 0;
    switch (grtyp[0]) {
        case 'A':
        case 'B':
        case 'E':
        case 'G':
        case 'L':
        case 'N':
        case 'S':
            break;
        default:
            fprintf(stderr,"ERROR: Fstdc - unsupported grid type: %c\n",grtyp[0]);
            istat = -1;
            break;
        }
    return istat;
}


int isGridValid(int ni,int nj,char *grtyp,char *grref,int ig1,int ig2,int ig3,int ig4,int i0, int j0,PyArrayObject *xs,PyArrayObject *ys) {
    int istat = 0,xdims[4],ydims[4];
    if (ni>0 && nj>0) {
        switch (grtyp[0]) {
            case '#':
                if ((PyObject *)xs==Py_None || (PyObject *)ys==Py_None || isPyArrayValid(xs)<0 || isPyArrayValid(ys)<0) {
                    fprintf(stderr,"ERROR: Fstdc - #-grtyp need valid axis\n");
                    istat = -1;
                } else {
                    getPyArrayDims(xdims,xs);
                    getPyArrayDims(ydims,ys);
                    if (i0<1 || j0<1 || (i0+ni-1)>(xdims[0]) || (j0+nj-1)>(ydims[1]) || xdims[1]>1 || ydims[0]>1) {
                        fprintf(stderr,"ERROR: Fstdc - #-grtyp need valid consistant dims: (i0,j0) = (%d,%d); (xdims0,xdims1) = (%d,%d); (ydims0,ydims1) = (%d,%d); (ni,nj) = (%d,%d)\n",i0,j0,xdims[0],xdims[1],ydims[0],ydims[1],ni,nj);
                        istat = -1;
                    }
                }
                if (istat>=0) istat = isGridTypeValid(grref);
                break;
            case 'Y':
                if ((PyObject *)xs==Py_None || (PyObject *)ys==Py_None || isPyArrayValid(xs)<0 || isPyArrayValid(ys)<0) istat = -1;
                else {
                    getPyArrayDims(xdims,xs);
                    getPyArrayDims(ydims,ys);
                    if (xdims[0]!=ni || xdims[1]!=nj || ydims[0]!=ni || ydims[1]!=nj) istat = -1;
                }
                if (istat>=0) istat = isGridTypeValid(grref);
                break;
            case 'Z':
                if ((PyObject *)xs==Py_None || (PyObject *)ys==Py_None || isPyArrayValid(xs)<0 || isPyArrayValid(ys)<0)  {
                    fprintf(stderr,"ERROR: Fstdc - Z-grtyp need valid axis\n");
                    istat = -1;
                } else {
                    getPyArrayDims(xdims,xs);
                    getPyArrayDims(ydims,ys);
                    if (xdims[0]!=ni || ydims[1]!=nj || xdims[1]>1 || ydims[0]>1)  {
                        fprintf(stderr,"ERROR: Fstdc - Z-grtyp need valid consistant dims: (xdims0,xdims1) = (%d,%d); (ydims0,ydims1) = (%d,%d); (ni,nj) = (%d,%d)\n",xdims[0],xdims[1],ydims[0],ydims[1],ni,nj);
                        istat = -1;
                    }
                    if (istat>=0) istat = isGridTypeValid(grref);
                }
                break;
            case 'A': //TODO: validate ig14 for these grids
            case 'B':
            case 'E':
            case 'G':
            case 'L':
            case 'N':
            case 'S':
                break;
            default:
                fprintf(stderr,"ERROR: Fstdc - unsupported grid type: %c %c\n",grref[0],grtyp[0]);
                istat = -1;
                break;
        }
    }
    if (istat<0) fprintf(stderr,"ERROR: Fstdc - invalid Grid parameters for grid: %c %c\n",grref[0],grtyp[0]);
    return istat;
}


int getGridHandle(int ni,int nj,char *grtyp,char *grref,int ig1,int ig2,int ig3,int ig4,int i0, int j0,PyArrayObject *xs,PyArrayObject *ys) {
    int gdid = -1,i0b=0,j0b=0;
    char *grtypZ = "Z";
    float *xsd,*ysd;
    if (isGridValid(ni,nj,grtyp,grref,ig1,ig2,ig3,ig4,i0,j0,xs,ys)>=0) {
        switch (grtyp[0]) {
            case '#':
                i0b = i0-1;
                j0b = j0-1;
            case 'Z':
                xsd = (float *)xs->data;
                ysd = (float *)ys->data;
                gdid = c_ezgdef_fmem(ni,nj,grtypZ,grref,ig1,ig2,ig3,ig4,&xsd[i0b],&ysd[j0b]);
                break;
            case 'Y': //TODO: support for Y grids
                fprintf(stderr,"ERROR: Fstdc - grtyp=Y not Yet Supported\n");
                break;
            default:
                gdid = c_ezqkdef(ni,nj,grtyp,ig1,ig2,ig3,ig4,0);
                break;
        }
    }
    return gdid;
}


static char Fstdc_fstouv__doc__[] =
"Interface to fstouv and fnom to open a RPN 2000 Standard File\niunit = Fstdc_fstouv(iunit,filename,options)\n@param iunit unit number of the file handle, 0 for a new one (int)\n@param filename (string)\n@param option type of file and R/W options(string)\n@return File unit number (int), None on error";

static PyObject *
Fstdc_fstouv(PyObject *self, PyObject *args) {
    int iun=0;
    char *filename="None";
    char *options="RND";
    if (PyArg_ParseTuple(args, "iss",&iun,&filename,&options)) {
        if (c_fnom(&iun,filename,options,0) >= 0) {
            if (c_fstouv(iun,filename,options) >= 0)
                return Py_BuildValue("i",iun);
        }
    }
    fprintf(stderr,"ERROR: Fstdc_fstouv  - failed to open file %s\n",filename);
    Py_INCREF(Py_None);
    return Py_None;
}


static char Fstdc_fstvoi__doc__[] =
"Print a list view a RPN Standard File rec (Interface to fstvoi)\nFstdc_fstvoi(iunit,option)\n@param iunit file unit number handle returned by Fstdc_fstouv (int)\n@param option 'OLDSTYLE' or 'NEWSTYLE' (sting)\n@return None";

static PyObject *
Fstdc_fstvoi(PyObject *self, PyObject *args) {
    char *options="NEWSTYLE";
    int iun;
    if (PyArg_ParseTuple(args, "is",&iun,&options))
        c_fstvoi(iun,options);
    Py_INCREF(Py_None);
    return Py_None;
}


static char Fstdc_fstfrm__doc__[] =
"Interface to fclos to close a RPN 2000 Standard File\nFstdc_fstfrm(iunit)\n@param iunit file unit number handle returned by Fstdc_fstouv (int)\n@return None";

static PyObject *
Fstdc_fstfrm(PyObject *self, PyObject *args) {
    int iun=0;
    if (PyArg_ParseTuple(args, "i",&iun)) {
        c_fstfrm(iun);
        c_fclos(iun);
    }
    Py_INCREF(Py_None);
    return Py_None;
}


static char Fstdc_fstsui__doc__[] =
"Interface to fstsui,\nrecParamDict = Fstdc_fstsui(iunit)\n@param iunit file unit number handle returned by Fstdc_fstouv (int)\n@returns python dict with record handle + record params keys/values";

static PyObject *
Fstdc_fstsui(PyObject *self, PyObject *args) {
    int iun, ni=0, nj=0, nk=0, handle;
    if (!PyArg_ParseTuple(args, "i",&iun)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    handle = c_fstsui(iun,&ni,&nj,&nk);
    return c2py_fstprm(handle);
}


static char Fstdc_fstinf__doc__[] =
"Find a record matching provided criterias (Interface to fstinf, dsfsui, fstinfx)\nrecParamDict = Fstdc_fstinf(iunit,nomvar,typvar,etiket,ip1,ip2,ip3,datev,inhandle)\nparam iunit file unit number handle returned by Fstdc_fstouv (int)\n@param nomvar seclect according to var name, blank==wildcard (string)\n@param typvar seclect according to var type, blank==wildcard (string)\n@param etiket seclect according to etiket, blank==wildcard (string)\n@param ip1 seclect according to ip1, -1==wildcard  (int)\n@param ip2 seclect according to ip2, -1==wildcard (int)\n@param ip3  seclect according to ip3, -1==wildcard (int)\n@param datev seclect according to date of validity, -1==wildcard (int)\n@param inhandle selcation criterion; inhandle=-2:search with criterion from start of file; inhandle=-1==fstsui, use previously provided criterion to find the next matching one; inhandle>=0 search with criterion from provided rec-handle (int)\n@returns python dict with record handle + record params keys/values";

static PyObject *
Fstdc_fstinf(PyObject *self, PyObject *args) {
    int iun, inhandle=-2, ni=0, nj=0, nk=0, datev=0, ip1=0, ip2=0, ip3=0;
    char *typvar, *nomvar, *etiket;
    int handle;

    if (!PyArg_ParseTuple(args, "isssiiiii",&iun,&nomvar,&typvar,&etiket,&ip1,&ip2,&ip3,&datev,&inhandle)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    if(inhandle == -2) {
        handle=c_fstinf(iun,&ni,&nj,&nk,datev,etiket,ip1,ip2,ip3,typvar,nomvar);
    } else if(inhandle == -1) {
        handle=c_fstsui(iun,&ni,&nj,&nk);
    } else if(inhandle >= 0) {
        handle=c_fstinfx(inhandle,iun,&ni,&nj,&nk,datev,etiket,ip1,ip2,ip3,typvar,nomvar);
    } else {
        Py_INCREF(Py_None); return Py_None;
    }
    return c2py_fstprm(handle);
}


static char Fstdc_fstinl__doc__[] =
"Find all records matching provided criterias (Interface to fstinl)\nWarning: list is limited to the first 10000 records in a file.\nrecList = Fstdc_fstinl(iunit,nomvar,typvar,etiket,ip1,ip2,ip3,datev)\nparam iunit file unit number handle returned by Fstdc_fstouv (int)\n@param nomvar seclect according to var name, blank==wildcard (string)\n@param typvar seclect according to var type, blank==wildcard (string)\n@param etiket seclect according to etiket, blank==wildcard (string)\n@param ip1 seclect according to ip1, -1==wildcard  (int)\n@param ip2 seclect according to ip2, -1==wildcard (int)\n@param ip3  seclect according to ip3, -1==wildcard (int)\n@param datev seclect according to date of validity, -1==wildcard (int)\n@returns python dict with handles+params of all matching records";

static PyObject *
Fstdc_fstinl(PyObject *self, PyObject *args) {
    int i,iun, ier, ni=0, nj=0, nk=0, datev=0, ip1=0, ip2=0, ip3=0;
    char *typvar, *nomvar, *etiket;
    PyObject *handle_list,*ihandle_obj;
    int nliste=0,nmax=50000,liste[50000];

    if (!PyArg_ParseTuple(args, "isssiiii",&iun,&nomvar,&typvar,&etiket,&ip1,&ip2,&ip3,&datev)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    handle_list = PyList_New(0);
    Py_INCREF(handle_list);

    ier = c_fstinl(iun, &ni, &nj, &nk,datev,etiket,ip1,ip2,ip3,typvar,nomvar,liste,&nliste,nmax);
    if (nliste==0) {Py_INCREF(Py_None); return Py_None;}
    for (i=0; i < nliste; i++) {
        ihandle_obj = c2py_fstprm(liste[i]);
        if (ihandle_obj == Py_None) {
            return Py_None;
        }
        Py_INCREF(ihandle_obj);
        PyList_Append(handle_list,ihandle_obj);
    }
    return handle_list;
}


static char Fstdc_fsteff__doc__[] =
"Erase a record (Interface to fsteff)\nistatus = Fstdc_fsteff(ihandle)\n@param ihandle handle of the record to erase (int)\n@return status (int)";

static PyObject *
Fstdc_fsteff(PyObject *self, PyObject *args) {
    int handle=0,status=0;
    if (!PyArg_ParseTuple(args, "i",&handle)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    status=c_fsteff(handle);
    return Py_BuildValue("i",status);
}


static char Fstdc_fstecr__doc__[] =
"Wrtie record to file (Interface to fstecr), always happend\nFstdc_fstecr(array,iunit,nomvar,typvar,etiket,ip1,ip2,ip3,dateo,grtyp,ig1,ig2,ig3,ig4,deet,npas,nbits)\n@param array data to be written to file (numpy.ndarray)\n@param iunit file unit number handle returned by Fstdc_fstouv (int)\n@param ... \n@return None";

static PyObject *
Fstdc_fstecr(PyObject *self, PyObject *args) {
    int iun, ip1=0, ip2=0, ip3=0, istat;
    char *typvar, *nomvar, *etiket, *grtyp;
    int dateo=0, deet=0, npas=0, nbits=0, ig1=0, ig2=0, ig3=0, ig4=0;
    int ni=0,nj=0,nk=0,datyp=0,rewrit=0;
    int dtl=4;
    int dims[4];
    PyArrayObject *array;

    if (!PyArg_ParseTuple(args, "Oisssiiiisiiiiiii",
        &array,&iun,&nomvar,&typvar,&etiket,&ip1,&ip2,&ip3,&dateo,&grtyp,&ig1,&ig2,&ig3,&ig4,&deet,&npas,&nbits)) {
        Py_INCREF(Py_None);
        return Py_None;
    }

#if defined(DEBUG)
    printf("datyp=%d dtl=%d\n",datyp,dtl);
    printf("writing array, nd=%d,ni=%d,nj=%d,nk=%d,datyp=%d,element length=%d,fstdtyp=%d\n",
        array->nd,ni,nj,nk,datyp,lentab[array->descr->type_num],datyps[array->descr->type_num]);
    printf("writing iun=%d,nomvar=:%s:,typvar=:%s:,etiket=:%s:,ip1=%d,ip2=%d,ip3=%d,\
        dateo=%d,grtyp=:%s:,ig1=%d,ig2=%d,ig3=%d,ig4=%d,deet=%d,npas=%d,nbits=%d\n",
        iun,nomvar,typvar,etiket,ip1,ip2,ip3,dateo,grtyp,ig1,ig2,ig3,ig4,deet,npas,nbits);
    /* printf("%x %x %x\n",array->data[0],array->data[4],array->data[8]); */
#endif

    if (isPyArrayValid(array)<0 || nbits<8 || nbits>64) {
        fprintf(stderr,"ERROR: Fstdc.fstecr - invalid input data/meta (nbits=%d)\n",nbits);
        Py_INCREF(Py_None);
        return Py_None;
    }
    getPyArrayDatyp(&datyp,&dtl,array);
    getPyArrayDims(dims,array);
    ni = dims[0];
    nj = dims[1];
    nk = dims[2];
    istat = c_fstecr(array->data,array->data,-nbits,iun,dateo,deet,npas,ni,nj,nk,ip1,ip2,ip3,
                typvar,nomvar,etiket,grtyp,ig1,ig2,ig3,ig4,datyp+c_fst_data_length(dtl),rewrit);
    return Py_BuildValue("i",istat);
}


static char Fstdc_fstluk__doc__[] =
"Read record on file (Interface to fstluk)\nmyRecDataDict = Fstdc_fstluk(ihandle)\n@param ihandle record handle (int) \n@return python dict with record params keys/values and data (numpy.ndarray)";

static PyObject *
Fstdc_fstluk(PyObject *self, PyObject *args) {
    PyArrayObject *newarray;
    int dimensions[3]={1,1,1}, ndimensions=3;
    int type_num=NPY_FLOAT;
    int ni=0, nj=0, nk=0, ip1=0, ip2=0, ip3=0;
    char TYPVAR[3]={' ',' ','\0'};
    char NOMVAR[5]={' ',' ',' ',' ','\0'};
    char ETIKET[13]={' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','\0'};
    char GRTYP[2]={' ','\0'};
    int handle, junk;
    int dateo=0, deet=0, npas=0, nbits=0, datyp=0, ig1=0, ig2=0, ig3=0, ig4=0;
    int swa=0, lng=0, dltf=0, ubc=0, extra1=0, extra2=0, extra3=0;

    if (!PyArg_ParseTuple(args, "i",&handle)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    if(handle >= 0) {
        junk =  c_fstprm(handle,&dateo,&deet,&npas,&ni,&nj,&nk,&nbits,&datyp,&ip1,&ip2,&ip3,
                TYPVAR,NOMVAR,ETIKET,GRTYP,&ig1,&ig2,&ig3,&ig4,
                &swa,&lng,&dltf,&ubc,&extra1,&extra2,&extra3);
    }
    if(junk < 0 || handle < 0) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    if (datyp == 0 || datyp == 2 || datyp == 4 || datyp == 130)
        type_num=NPY_INT;
    else if (datyp == 1 || datyp == 5 || datyp == 6 || datyp == 134)
        type_num=NPY_FLOAT;
    else if (datyp == 3 )
        type_num=NPY_CHAR;
    else {
        fprintf(stderr,"ERROR: PyFstluk - unrecognized data type: %d\n",datyp);
        Py_INCREF(Py_None);
        return Py_None;
    }

    dimensions[0] = (ni>1) ? ni : 1  ;
    dimensions[1] = (nj>1) ? nj : 1 ;
    dimensions[2] = (nk>1) ? nk : 1 ;
    if(nk>1) ndimensions=3;
    else if(nj>1) ndimensions=2;
    else ndimensions=1;

    newarray = PyArray_NewFromDescr(&PyArray_Type,
                                    PyArray_DescrFromType(type_num),
                                    ndimensions,dimensions,
                                    NULL, NULL, FTN_Style_Array,
                                    NULL);
    junk = c_fstluk(newarray->data,handle,&ni,&nj,&nk);
    if( junk >= 0 )
        return Py_BuildValue("O",newarray);
    else {
        Py_DECREF(newarray);
        Py_INCREF(Py_None);
        return Py_None;
    }
}


static char Fstdc_fst_edit_dir__doc__[] =
"Rewrite the parameters of an rec on file, data part is unchanged (Interface to fst_edit_dir)\nstatus = Fstdc_fst_edit_dir(ihandle,date,deet,npas,ni,nj,nk,ip1,ip2,ip3,typvar,nomvar,etiket,grtyp,ig1,ig2,ig3,ig4,datyp)\n@param ihandle record handle (int)\n@param ... \n@return status (int)";

static PyObject *
Fstdc_fst_edit_dir(PyObject *self, PyObject *args) {
    int handle=0;
    int status=0;
    int ip1=-1, ip2=-1, ip3=-1;
    char *typvar, *nomvar, *etiket, *grtyp;
    int date=-1, deet=-1, npas=-1, ig1=-1, ig2=-1, ig3=-1, ig4=-1;
    int ni=-1,nj=-1,nk=-1,datyp=-1;

    if (!PyArg_ParseTuple(args, "iiiiiiiiiissssiiiii",
            &handle,&date,&deet,&npas,&ni,&nj,&nk,&ip1,&ip2,&ip3,&typvar,&nomvar,&etiket,&grtyp,&ig1,&ig2,&ig3,&ig4,&datyp)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    status = c_fst_edit_dir(handle,date,deet,npas,ni,nj,nk,ip1,ip2,ip3,typvar,nomvar,etiket,grtyp,ig1,ig2,ig3,ig4,datyp);
    return Py_BuildValue("i",status);
}


static char Fstdc_newdate__doc__[] =
"Convert data to/from printable format and CMC stamp (Interface to newdate)\n(fdat1,fdat2,fdat3) = Fstdc_newdate(date1,date2,date3,mode)\n@param ...see newdate doc... \n@return tuple with converted date values ...see newdate doc...";

static PyObject *
Fstdc_newdate(PyObject *self, PyObject *args) {
    int date1=0,date2=0,date3=0,mode=0;
    int status=0;
    wordint fdat1,fdat2,fdat3,fmode;
    if (!PyArg_ParseTuple(args, "iiii",&date1,&date2,&date3,&mode)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    fdat1 = date1; fdat2 = date2; fdat3=date3;
    fmode = mode;
    status=f77name(newdate)(&fdat1,&fdat2,&fdat3,&fmode);
    return Py_BuildValue("(iii)",fdat1,fdat2,fdat3);
}


static char Fstdc_difdatr__doc__[] =
"Compute differenc between 2 CMC datatime stamps (Interface to difdatr)\nnhours = Fstdc_difdatr(date1,date2)\n@param date1 CMC datatime stamp (int)\n@param date2 CMC datatime stamp (int)\n@return number of hours = date2-date1 (float)";

static PyObject *
Fstdc_difdatr(PyObject *self, PyObject *args) {
    int date1=0,date2=0;
    int status=0;
    double nhours;
    wordint fdat1,fdat2;
    if (!PyArg_ParseTuple(args, "ii",&date1,&date2)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    fdat1 = date1; fdat2 = date2;
    status=f77name(difdatr)(&fdat1,&fdat2,&nhours);
    return Py_BuildValue("d",nhours);
}


static char Fstdc_incdatr__doc__[] =
"Increase CMC datetime stamp by a N hours (Interface to incdatr)\ndate2 = Fstdc_incdatr(date1,nhours)\n@param date1 original CMC datetime stamp(int)\n@param nhours number of hours to increase the date (float)\n@return Increase CMC datetime stamp (int)"
;

static PyObject *
Fstdc_incdatr(PyObject *self, PyObject *args) {
    int date1=0,date2=0;
    int status=0;
    double nhours;
    wordint fdat1,fdat2;
    if (!PyArg_ParseTuple(args, "id",&date2,&nhours)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    fdat2 = date2;
    status=f77name(incdatr)(&fdat1,&fdat2,&nhours);
    date1 = fdat1;
    return Py_BuildValue("i",date1);
}


static char Fstdc_datematch__doc__[] =
"Determine if date stamp match search crieterias\ndoesmatch = Fstdc_datematch(indate,dateRangeStart,dateRangeEnd,delta)\n@param indate Date to be check against, CMC datetime stamp (int)\n@param dateRangeStart, CMC datetime stamp (int) \n@param dateRangeEnd, CMC datetime stamp (int)\n@param delta (float)\n@return 1:if date match; 0 otherwise";

static PyObject *
Fstdc_datematch(PyObject *self, PyObject *args) {
    int datelu, debut, fin;
    float delta;
    double nhours,modulo,ddelta=delta;
    float toler=.00023;            /* tolerance d'erreur de 5 sec */

    if (!PyArg_ParseTuple(args, "iiif",&datelu,&debut,&fin,&delta)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    if ((fin != -1) && (datelu > fin)) return Py_BuildValue("i",0);
    if (debut != -1) {
        if (datelu < debut) return Py_BuildValue("i",0);
        f77name(difdatr)(&datelu,&debut,&nhours);
    }
    else {
        if (fin == -1) return Py_BuildValue("i",1);
        f77name(difdatr)(&fin,&datelu,&nhours);
    }
    modulo = fmod(nhours,ddelta);
    if ((modulo < toler) || ((delta - modulo) < toler))
        return Py_BuildValue("i",1);
    else
        return Py_BuildValue("i",0);
}


static char Fstdc_level_to_ip1__doc__[] =
"Encode level value to ip1 (Interface to convip)\nmyip1List = Fstdc_level_to_ip1(level_list,kind) \n @param level_list list of level values (list of float)\n @param kind type of level (int)\n @return [(ip1new,ip1old),...] (list of tuple of int)";

static PyObject *
Fstdc_level_to_ip1(PyObject *self, PyObject *args) {
    int i,kind, nelm, status;
    float level;
    long ipnew, ipold;
    wordint fipnew, fipold, fmode, flag=0, fkind;
    char strg[30];
    PyObject *level_list, *ip1_list=Py_None, *item, *ipnewold_obj;
    int convip();

    if (!PyArg_ParseTuple(args, "Oi",&level_list,&kind)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    fkind = (wordint) kind;
    nelm = PyList_Size(level_list);
    ip1_list = PyList_New(0);
    Py_INCREF(ip1_list);
    for (i=0; i < nelm; i++) {
        item = PyList_GetItem(level_list,i);
        level = (float) PyFloat_AsDouble(item);
        fmode = 2;
        status=f77name(convip)(&fipnew,&level,&fkind,&fmode,&strg,&flag,30);
        fmode = 3;
        status=f77name(convip)(&fipold,&level,&fkind,&fmode,&strg,&flag,30);
        ipnew = (long) fipnew;
        ipold = (long) fipold;
        ipnewold_obj = Py_BuildValue("(l,l)",ipnew,ipold);
        Py_INCREF(ipnewold_obj);
        PyList_Append(ip1_list,ipnewold_obj);
    }
    return (ip1_list);
}

static char Fstdc_ip1_to_level__doc__[] =
"Decode ip1 to level type,value (Interface to convip)\nmyLevelList = Fstdc_ip1_to_level(ip1_list)\n@parma tuple/list of ip1 values to decode\n@return list of tuple (level,kind)";

static PyObject *
Fstdc_ip1_to_level(PyObject *self, PyObject *args) {
    int i,kind, nelm, status;
    float level;
    wordint fip1, fmode, flag=0, fkind;
    char strg[30];
    PyObject *ip1_list=Py_None, *level_list, *item, *level_kind_obj;
    int convip();

    if (!PyArg_ParseTuple(args, "O",&ip1_list)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    fmode = -1;
    nelm = PyList_Size(ip1_list);
    level_list = PyList_New(0);
    Py_INCREF(level_list);
    for (i=0; i < nelm; i++) {
        item  = PyList_GetItem(ip1_list,i);
        fip1  = (wordint) PyLong_AsLong(item);
        status=f77name(convip)(&fip1,&level,&fkind,&fmode,&strg,&flag,30);
        kind  = (int) fkind;
        level_kind_obj = Py_BuildValue("(f,i)",level,kind);
        Py_INCREF(level_kind_obj);
        PyList_Append(level_list,level_kind_obj);
    }
    return (level_list);
}


static char Fstdc_mapdscrpt__doc__[] =
"Interface to get map descriptors for use with PyNGL\nmyMapDescDict = Fstdc_mapdscrpt(x1,y1,x2,y2,ni,nj,cgrtyp,ig1,ig2,ig3,ig4)\n@param ...TODO... \n@return python dict with keys/values";

static PyObject *
Fstdc_mapdscrpt(PyObject *self, PyObject *args) {
	int ig1, ig2, ig3, ig4, one=1, ni, nj, proj;
        char *cgrtyp;
        float x1,y1, x2,y2, polat,polong, rot, lat1,lon1, lat2,lon2;

	if (!PyArg_ParseTuple(args, "ffffiisiiii",&x1,&y1,&x2,&y2,&ni,&nj,&cgrtyp,&ig1,&ig2,&ig3,&ig4)) {
	  Py_INCREF(Py_None);
	  return Py_None;
        }
#if defined(DEBUG)
        printf("Debug apres parse tuple cgrtyp[0]=%c\n",cgrtyp[0]);
        printf("Debug appel Mapdesc_PyNGL\n");
#endif
        f77name(mapdesc_pyngl)(cgrtyp,&one,&ig1,&ig2,&ig3,&ig4,&x1,&y1,&x2,&y2,&ni,&nj,\
                               &proj,&polat,&polong,&rot,&lat1,&lon1,&lat2,&lon2,1);
#if defined(DEBUG)
	printf("Fstdc_mapdscrpt ig1=%d ig2=%d ig3=%d ig4=%d\n",ig1,ig2,ig3,ig4);
	printf("Fstdc_mapdscrpt polat=%f polong=%f rot=%f, lat1=%f lon1=%f lat2=%f, lon2=%f\n",polat,polong,rot,lat1,lon1,lat2,lon2);
#endif
        return Py_BuildValue("{s:f,s:f,s:f,s:f,s:f,s:f,s:f}","polat",polat,"polong",polong,\
                             "rot",rot,"lat1",lat1,"lon1",lon1,"lat2",lat2,"lon2",lon2);
/*        return Py_BuildValue("f",rot); */
}



static char Fstdc_ezinterp__doc__[] =
"Interpolate from one grid to another\nnewArray = Fstdc_ezinterp(arrayin,arrayin2,(niS,njS),grtypS,(grrefS,ig1S,ig2S,ig3S,ig4S),(xsS,ysS),hasSrcAxis,(i0S,j0S),(niD,njD),grtypD,(grrefD,ig1D,ig2D,ig3D,ig4D),(xsD,ysD),hasDstAxis,(i0D,j0D),isVect)\n@param ...TODO...\n@return interpolated data (numpy.ndarray)";

static PyObject *
Fstdc_ezinterp(PyObject *self, PyObject *args) {
    int ig1S, ig2S, ig3S, ig4S, niS, njS, i0S,j0S, gdid_src;
    int ig1D, ig2D, ig3D, ig4D, niD, njD, i0D,j0D, gdid_dst;
    int hasSrcAxis,hasDstAxis,isVect,ier;
    char *grtypS,*grtypD,*grrefS,*grrefD;
    int dimensions[3]={1,1,1}, ndimensions=3;
    int type_num=NPY_FLOAT;
    PyArrayObject *arrayin,*arrayin2,*newarray,*newarray2,*xsS,*ysS,*xsD,*ysD;

    if (!PyArg_ParseTuple(args, "OO(ii)s(siiii)(OO)i(ii)(ii)s(siiii)(OO)i(ii)i",
            &arrayin,&arrayin2,
            &niS,&njS,&grtypS,&grrefS,&ig1S,&ig2S,&ig3S,&ig4S,&xsS,&ysS,&hasSrcAxis,&i0S,&j0S,
            &niD,&njD,&grtypD,&grrefD,&ig1D,&ig2D,&ig3D,&ig4D,&xsD,&ysD,&hasDstAxis,&i0D,&j0D,
            &isVect)) {
            fprintf(stderr,"ERROR: Fstdc_ezinterp() - wrong arg type\n");
            Py_INCREF(Py_None);
            return Py_None;
    }

#if defined(DEBUG)
    printf("Debug Fstdc_ezinterp grtypS[0]=%c grrefS[0]=%c\n",grtypS[0],grrefS[0]);
    printf("Debug Fstdc_ezinterp ig1S=%d ig2S=%d ig3S=%d ig4S=%d\n",ig1S,ig2S,ig3S,ig4S);
    printf("Debug Fstdc_ezinterp niS=%d njS=%d axis=%d\n",niS,njS,hasSrcAxis);
    if (hasSrcAxis) {
        printf("Debug Fstdc_ezinterp xaxisS[0],[ni]=%f, %f\n",(float)xsS->data[0],(float)xsS->data[niS-1]);
        printf("Debug Fstdc_ezinterp yaxisS[0],[nj]=%f, %f\n",(float)ysS->data[0],(float)ysS->data[njS-1]);
        imprime_ca("xs",(float *)xsS->data,niS);
        imprime_ca("ys",(float *)ysS->data,njS);
    }
    printf("Debug Fstdc_ezinterp grtypD[0]=%c grrefD[0]=%c\n",grtypD[0],grrefD[0]);
    printf("Debug Fstdc_ezinterp ig1D=%d ig2D=%d ig3D=%d ig4D=%d\n",ig1D,ig2D,ig3D,ig4D);
    printf("Debug Fstdc_ezinterp niD=%d njD=%d axis=%d\n",niD,njD,hasDstAxis);
    if (hasDstAxis) {
        printf("Debug Fstdc_ezinterp xaxisD[0],[ni]=%f, %f\n",(float)xsD->data[0],(float)xsD->data[niD-1]);
        printf("Debug Fstdc_ezinterp yaxisD[0],[nj]=%f, %f\n",(float)ysD->data[0],(float)ysD->data[njD-1]);
        imprime_ca("xs",(float *)xsD->data,niD);
        imprime_ca("ys",(float *)ysD->data,njD);
    }
    //	imprime_ca("arrayin",(float *)arrayin->data,10);
#endif

    ier = 0;
    if (isPyArrayValid(arrayin)<0 || arrayin->descr->type_num != type_num) ier=-1;
    if (isVect && (isPyArrayValid(arrayin2)<0 || arrayin2->descr->type_num != type_num)) ier=-1;
    if (ier<0) {
        fprintf(stderr,"ERROR: Fstdc_ezinterp() - Input arrays should be Fortran/Continuous of type float32\n");
        Py_INCREF(Py_None);
        return Py_None;
    }

    gdid_src = getGridHandle(niS,njS,grtypS,grrefS,ig1S,ig2S,ig3S,ig4S,i0S,j0S,xsS,ysS);
    gdid_dst = getGridHandle(niD,njD,grtypD,grrefD,ig1D,ig2D,ig3D,ig4D,i0D,j0D,xsD,ysD);
    if (gdid_src<0 || gdid_dst<0) {
        fprintf(stderr,"ERROR: Fstdc_ezinterp() - invalid Grid Desc\n");
        Py_INCREF(Py_None);
        return Py_None;
    }

    ier = c_ezdefset(gdid_dst,gdid_src);
    if (ier<0) {
        fprintf(stderr,"ERROR: Fstdc_ezinterp() - problem defining a grid interpolation set\n");
        Py_INCREF(Py_None);
        return Py_None;
    }

    dimensions[0] = (niD>1) ? niD : 1;
    dimensions[1] = (njD>1) ? njD : 1;
    dimensions[2] = 1 ;
    if(njD>1) ndimensions=2;
    else ndimensions=1;

    newarray = PyArray_NewFromDescr(&PyArray_Type,
                                    PyArray_DescrFromType(type_num),
                                    ndimensions,dimensions,
                                    NULL, NULL, FTN_Style_Array,
                                    NULL);
    if (isVect)
        newarray2 = PyArray_NewFromDescr(&PyArray_Type,
                                        PyArray_DescrFromType(type_num),
                                        ndimensions,dimensions,
                                        NULL, NULL, FTN_Style_Array,
                                        NULL);

    if (isVect) {
        ier = c_ezuvint(newarray->data,newarray2->data,arrayin->data,arrayin2->data);
        if (ier>=0) return Py_BuildValue("OO",newarray,newarray2);
    } else {
        ier = c_ezsint(newarray->data,arrayin->data);
        if (ier>=0) return Py_BuildValue("O",newarray);
    }
    fprintf(stderr,"ERROR: Fstdc_ezinterp() - interpolation problem in ezscint\n");
    Py_DECREF(newarray);
    if (isVect) {
        Py_DECREF(newarray2);
    }
    Py_INCREF(Py_None);
    return Py_None;
}


static char Fstdc_cxgaig__doc__[] =
"Encode grid descriptors (Interface to cxgaig)\n(ig1,ig2,ig3,ig4) = Fstdc_cxgaig(grtyp,xg1,xg2,xg3,xg4) \n@param ...TODO...\n@return (ig1,ig2,ig3,ig4)";

static PyObject *
Fstdc_cxgaig(PyObject *self, PyObject *args) {
	int ig1,ig2,ig3,ig4;
        float xg1,xg2,xg3,xg4;
        char *grtyp;
        int status;
	if (!PyArg_ParseTuple(args, "sffff",&grtyp,&xg1,&xg2,&xg3,&xg4)) {
	   Py_INCREF(Py_None);
           return Py_None;
	}
	status=f77name(cxgaig)(grtyp,&ig1,&ig2,&ig3,&ig4,&xg1,&xg2,&xg3,&xg4);
	return Py_BuildValue("(iiii)",ig1,ig2,ig3,ig4);
}


static char Fstdc_cigaxg__doc__[] =
"Decode grid descriptors (Interface to cigaxg)\n(xg1,xg2,xg3,xg4) = Fstdc_cigaxg(grtyp,ig1,ig2,ig3,ig4)\n@param ...TODO...\n@return (xg1,xg2,xg3,xg4)"
;

static PyObject *
Fstdc_cigaxg(PyObject *self, PyObject *args) {
	int ig1,ig2,ig3,ig4;
        float xg1,xg2,xg3,xg4;
        char *grtyp;
        int status;
	if (!PyArg_ParseTuple(args, "siiii",&grtyp,&ig1,&ig2,&ig3,&ig4)) {
	   Py_INCREF(Py_None);
           return Py_None;
	}
	status=f77name(cigaxg)(grtyp,&xg1,&xg2,&xg3,&xg4,&ig1,&ig2,&ig3,&ig4);
	return Py_BuildValue("(ffff)",xg1,xg2,xg3,xg4);
}


static char Fstdc_ezgetlalo__doc__[] =
"Get Lat-Lon of grid points centers and corners\n(lat,lon,clat,clon) = Fstdc_ezgetlalo((niS,njS),grtypS,(grrefS,ig1S,ig2S,ig3S,ig4S),(xsS,ysS),hasSrcAxis,(i0S,j0S),doCorners)\n@param ...TODO...\n@return tuple of (numpy.ndarray) with center lat/lon (lat,lon) and optionally corners lat/lon (clat,clon)";

static PyObject *
Fstdc_ezgetlalo(PyObject *self, PyObject *args) {
    int ig1S, ig2S, ig3S, ig4S, niS, njS, i0S,j0S, gdid_src;
    int hasSrcAxis,doCorners,ier,n,nbcorners=4;
    char *grtypS,*grrefS;
    int dimensions[3]={1,1,1}, ndimensions=3;
    int type_num=NPY_FLOAT;
    PyArrayObject *lat,*lon,*clat,*clon,*xsS,*ysS,*x,*y,*xc,*yc;

    if (!PyArg_ParseTuple(args, "(ii)s(siiii)(OO)i(ii)i",
            &niS,&njS,&grtypS,&grrefS,&ig1S,&ig2S,&ig3S,&ig4S,
            &xsS,&ysS,&hasSrcAxis,&i0S,&j0S,&doCorners)) {
        fprintf(stderr,"ERROR: Fstdc_ezgetlalo() - wrong arg type\n");
        Py_INCREF(Py_None);
        return Py_None;
    }

#if defined(DEBUG)
    printf("Debug Fstdc_ezinterp grtypS[0]=%c grrefS[0]=%c\n",grtypS[0],grrefS[0]);
    printf("Debug Fstdc_ezinterp ig1S=%d ig2S=%d ig3S=%d ig4S=%d\n",ig1S,ig2S,ig3S,ig4S);
    printf("Debug Fstdc_ezinterp niS=%d njS=%d axis=%d\n",niS,njS,hasSrcAxis);
    if (hasSrcAxis) {
        printf("Debug Fstdc_ezinterp (%d,%d) xaxisS[0],[ni]=%f, %f\n",xsS->dimensions[0],xsS->dimensions[1],((float*)xsS->data)[0],((float*)xsS->data)[niS-1]);
        printf("Debug Fstdc_ezinterp (%d,%d) yaxisS[0],[nj]=%f, %f\n",ysS->dimensions[0],ysS->dimensions[1],((float*)ysS->data)[0],((float*)ysS->data)[njS-1]);
        imprime_ca("xs",(float *)xsS->data,niS);
        imprime_ca("ys",(float *)ysS->data,njS);
    }
#endif

    gdid_src = getGridHandle(niS,njS,grtypS,grrefS,ig1S,ig2S,ig3S,ig4S,i0S,j0S,xsS,ysS);
    if (gdid_src<0) {
        fprintf(stderr,"ERROR: Fstdc_ezgetlalo() - invalid Grid Desc\n");
        Py_INCREF(Py_None);
        return Py_None;
    }

    //Define output array dims
    niS = (niS>1) ? niS : 1 ;
    njS = (njS>1) ? njS : 1 ;
    dimensions[0] = niS;
    dimensions[1] = njS;
    dimensions[2] = 1 ;
    if(njS>1) ndimensions=2;
    else ndimensions=1;

    //Compute centers Lat-Lon values
    lat = PyArray_NewFromDescr(&PyArray_Type,
                                PyArray_DescrFromType(type_num),
                                ndimensions,dimensions,
                                NULL, NULL, FTN_Style_Array,
                                NULL);
    lon = PyArray_NewFromDescr(&PyArray_Type,
                                PyArray_DescrFromType(type_num),
                                ndimensions,dimensions,
                                NULL, NULL, FTN_Style_Array,
                                NULL);

    ier = c_gdll(gdid_src, (float *)lat->data, (float *)lon->data);
    if (ier<0) {
        fprintf(stderr,"ERROR: Fstdc_ezgetlalo() - problem computing lat,lon in ezscint\n");
        Py_DECREF(lat);
        Py_DECREF(lon);
        Py_INCREF(Py_None);
        return Py_None;
    }

    //Compute corners Lat-Lon values
    if (doCorners) {
        x = PyArray_NewFromDescr(&PyArray_Type,
                                    PyArray_DescrFromType(type_num),
                                    ndimensions,dimensions,
                                    NULL, NULL, FTN_Style_Array,
                                    NULL);
        y = PyArray_NewFromDescr(&PyArray_Type,
                                    PyArray_DescrFromType(type_num),
                                    ndimensions,dimensions,
                                    NULL, NULL, FTN_Style_Array,
                                    NULL);

        dimensions[0] = nbcorners;
        dimensions[1] = niS;
        dimensions[2] = njS;
        if(njS>1) ndimensions=3;
        else ndimensions=2;
        clat = PyArray_NewFromDescr(&PyArray_Type,
                                    PyArray_DescrFromType(type_num),
                                    ndimensions,dimensions,
                                    NULL, NULL, FTN_Style_Array,
                                    NULL);
        clon = PyArray_NewFromDescr(&PyArray_Type,
                                    PyArray_DescrFromType(type_num),
                                    ndimensions,dimensions,
                                    NULL, NULL, FTN_Style_Array,
                                    NULL);
        xc = PyArray_NewFromDescr(&PyArray_Type,
                                    PyArray_DescrFromType(type_num),
                                    ndimensions,dimensions,
                                    NULL, NULL, FTN_Style_Array,
                                    NULL);
        yc = PyArray_NewFromDescr(&PyArray_Type,
                                    PyArray_DescrFromType(type_num),
                                    ndimensions,dimensions,
                                    NULL, NULL, FTN_Style_Array,
                                    NULL);

        n = niS*njS;
        ier = c_gdxyfll(gdid_src,x->data,y->data,lat->data,lon->data,n);
        if (ier>=0) ier = f77name(get_corners_xy)(xc->data,yc->data,x->data,y->data,&niS,&njS);
        n = niS*njS*nbcorners;
        if (ier>=0) ier = c_gdllfxy(gdid_src,clat->data,clon->data,xc->data,yc->data,n);

        Py_DECREF(x);
        Py_DECREF(y);
        Py_DECREF(xc);
        Py_DECREF(yc);

        if (ier<0) {
            fprintf(stderr,"ERROR: Fstdc_ezgetlalo() - problem computing clat,clon in ezscint\n");
            Py_DECREF(clat);
            Py_DECREF(clon);
            Py_INCREF(Py_None);
            return Py_None;
        }

        return Py_BuildValue("OOOO",lat,lon,clat,clon);
    } else {
        return Py_BuildValue("OO",lat,lon);
    }

}

/* List of methods defined in the module */

static struct PyMethodDef Fstdc_methods[] = {
    {"fstouv",	(PyCFunction)Fstdc_fstouv,	METH_VARARGS,	Fstdc_fstouv__doc__},
    {"fstvoi",	(PyCFunction)Fstdc_fstvoi,	METH_VARARGS,	Fstdc_fstvoi__doc__},
    {"fstfrm",	(PyCFunction)Fstdc_fstfrm,	METH_VARARGS,	Fstdc_fstfrm__doc__},
    {"fstsui",	(PyCFunction)Fstdc_fstsui,	METH_VARARGS,	Fstdc_fstsui__doc__},
    {"fstinf",	(PyCFunction)Fstdc_fstinf,	METH_VARARGS,	Fstdc_fstinf__doc__},
    {"fstinl",	(PyCFunction)Fstdc_fstinl,	METH_VARARGS,	Fstdc_fstinl__doc__},
    {"fstecr",	(PyCFunction)Fstdc_fstecr,	METH_VARARGS,	Fstdc_fstecr__doc__},
    {"fsteff",	(PyCFunction)Fstdc_fsteff,	METH_VARARGS,	Fstdc_fsteff__doc__},
    {"fstluk",	(PyCFunction)Fstdc_fstluk,	METH_VARARGS,	Fstdc_fstluk__doc__},
    {"fst_edit_dir",(PyCFunction)Fstdc_fst_edit_dir,METH_VARARGS,	Fstdc_fst_edit_dir__doc__},
    {"newdate",	(PyCFunction)Fstdc_newdate,	METH_VARARGS,	Fstdc_newdate__doc__},
    {"difdatr",	(PyCFunction)Fstdc_difdatr,	METH_VARARGS,	Fstdc_difdatr__doc__},
    {"incdatr",	(PyCFunction)Fstdc_incdatr,	METH_VARARGS,	Fstdc_incdatr__doc__},
    {"datematch",	(PyCFunction)Fstdc_datematch,	METH_VARARGS,	Fstdc_datematch__doc__},
    {"level_to_ip1",(PyCFunction)Fstdc_level_to_ip1,METH_VARARGS,	Fstdc_level_to_ip1__doc__},
    {"ip1_to_level",(PyCFunction)Fstdc_ip1_to_level,METH_VARARGS,	Fstdc_ip1_to_level__doc__},
    {"mapdscrpt",	(PyCFunction)Fstdc_mapdscrpt,	METH_VARARGS,	Fstdc_mapdscrpt__doc__},
    {"ezinterp",	(PyCFunction)Fstdc_ezinterp,	METH_VARARGS,	Fstdc_ezinterp__doc__},
    {"cxgaig",	(PyCFunction)Fstdc_cxgaig,	METH_VARARGS,	Fstdc_cxgaig__doc__},
    {"cigaxg",	(PyCFunction)Fstdc_cigaxg,	METH_VARARGS,	Fstdc_cigaxg__doc__},
    {"ezgetlalo",	(PyCFunction)Fstdc_ezgetlalo,	METH_VARARGS,	Fstdc_ezgetlalo__doc__},

    {NULL,	 (PyCFunction)NULL, 0, NULL}		/* sentinel */
};


/* Initialization function for the module (*must* be called initFstdc) */

static char Fstdc_module_documentation[] =
"Module Fstdc contains the classes used to access RPN Standard Files (rev 2000)\n@author: Mario Lepine <mario.lepine@ec.gc.ca>\n@author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>";

void initFstdc() {
    PyObject *m, *d;
    int istat;
    char *msglvl="MSGLVL";
    char *tolrnc="TOLRNC";

    /* Create the module and add the functions */
    m = Py_InitModule4("Fstdc", Fstdc_methods,
            Fstdc_module_documentation,
            (PyObject*)NULL,PYTHON_API_VERSION);

    /* Import the array object */
    import_array();

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);
    ErrorObject = PyString_FromString("Fstdc.error");
    PyDict_SetItemString(d, "error", ErrorObject);

    /* XXXX Add constants here */

    /* Check for errors */
    if (PyErr_Occurred())
            Py_FatalError("can't initialize module Fstdc");
    //printf("RPN (2000) Standard File module V-%s (%s) initialized\n",version,lastmodified);
    init_lentab();

    istat = c_fstopi(msglvl,8,0); //8 - print fatal error messages and up;10 - print system (internal) error messages only
    istat = c_fstopi(tolrnc,6,0); //6 - tolerate warning level and lower;8 - tolerate error level and lower
}

// kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
