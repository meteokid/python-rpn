/*
Module Fstdc contains the functions used to access RPN Standard Files (rev 2000)
@author: Michel Valin <michel.valin@ec.gc.ca>
@author: Mario Lepine <mario.lepine@ec.gc.ca>
@author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
@date: 2009-09
*/
/* #define DEBUG On */
#include <Python.h>
#include <numpy/arrayobject.h>

#include <rpnmacros.h>
#include "armnlib.h"

#include <strings.h>

#include "utils/py_capi_ftn_utils.h"
#include "utils/get_corners_xy.h"
#include "rpn_version.h"

#define FSTDC_FILE_RW "RND+R/W"
#define FSTDC_FILE_RW_OLD "RND+R/W+OLD"
#define FSTDC_FILE_RO "RND+R/O"

#define LEVEL_KIND_MSL 0
#define LEVEL_KIND_SIG 1
#define LEVEL_KIND_PMB 2
#define LEVEL_KIND_ANY 3
#define LEVEL_KIND_MGL 4
#define LEVEL_KIND_HYB 5
#define LEVEL_KIND_TH 6

static const int withFortranOrder = 1;

//TODO: add more Error distinctions to catch more specific things
static PyObject *FstdcError;
static PyObject *FstdcTooManyRecError;

static PyObject *Fstdc_version(PyObject *self, PyObject *args);
static PyObject *Fstdc_fstouv(PyObject *self, PyObject *args);
static PyObject *Fstdc_fstvoi(PyObject *self, PyObject *args);
static PyObject *Fstdc_fstrwd(PyObject *self, PyObject *args);
static PyObject *Fstdc_fstinf(PyObject *self, PyObject *args);
static PyObject *c2py_fstprm(int handle);
static PyObject *Fstdc_fstinl(PyObject *self, PyObject *args);
static PyObject *Fstdc_fstsui(PyObject *self, PyObject *args);
static PyObject *Fstdc_fstluk(PyObject *self, PyObject *args);
static PyObject *Fstdc_fst_edit_dir(PyObject *self, PyObject *args);
static PyObject *Fstdc_fstecr(PyObject *self, PyObject *args);
static PyObject *Fstdc_fsteff(PyObject *self, PyObject *args);
static PyObject *Fstdc_fstfrm(PyObject *self, PyObject *args);
static PyObject *Fstdc_cxgaig(PyObject *self, PyObject *args);
static PyObject *Fstdc_cigaxg(PyObject *self, PyObject *args);
static PyObject *Fstdc_level_to_ip1(PyObject *self, PyObject *args);
static PyObject *Fstdc_ip1_to_level(PyObject *self, PyObject *args);
static PyObject *Fstdc_newdate(PyObject *self, PyObject *args);
static PyObject *Fstdc_difdatr(PyObject *self, PyObject *args);
static PyObject *Fstdc_incdatr(PyObject *self, PyObject *args);
static PyObject *Fstdc_datematch(PyObject *self, PyObject *args);
static PyObject *Fstdc_ezgetlalo(PyObject *self, PyObject *args);
// Add option-setting for ezscint
static PyObject *Fstdc_ezgetopt(PyObject *self, PyObject *args);
static PyObject *Fstdc_ezsetopt(PyObject *self, PyObject *args);
static PyObject *Fstdc_ezsetval(PyObject *self, PyObject *args);
static PyObject *Fstdc_ezgetval(PyObject *self, PyObject *args);

static int getGridHandle(int ni,int nj,char *grtyp,char *grref,
    int ig1,int ig2,int ig3,int ig4,int i0, int j0,
    PyArrayObject *xs,PyArrayObject *ys);
static int isGridValid(int ni,int nj,char *grtyp,char *grref,
    int ig1,int ig2,int ig3,int ig4,int i0, int j0,
    PyArrayObject *xs,PyArrayObject *ys);
static int isGridTypeValid(char *grtyp);
static PyObject *Fstdc_ezinterp(PyObject *self, PyObject *args);


static char Fstdc_version__doc__[] = "(version,lastUpdate) = Fstdc.version()";

static PyObject *Fstdc_version(PyObject *self, PyObject *args) {
    return Py_BuildValue("(ss)",VERSION,LASTUPDATE);
}


static char Fstdc_fstouv__doc__[] =
        "Interface to fstouv and fnom to open a RPN 2000 Standard File\n\
        iunit = Fstdc.fstouv(iunit,filename,options)\n\
        @param iunit unit number of the file handle, 0 for a new one (int)\n\
        @param filename (string)\n\
        @param option type of file and R/W options (string)\n\
        @return File unit number (int), NULL on error\n\
        @exception TypeError\n\
        @exception Fstdc.error";

static PyObject *Fstdc_fstouv(PyObject *self, PyObject *args) {
    int iun=0,errorCode;
    char *filename="None";
    char *options="RND";
    if (!PyArg_ParseTuple(args, "iss",&iun,&filename,&options)) {
        return NULL;
    }
    errorCode = c_fnom(&iun,filename,options,0);
    if (errorCode >= 0)
        errorCode = c_fstouv(iun,filename,options);
    if (errorCode >= 0) {
        return Py_BuildValue("i",iun);
    } else {
        PyErr_SetString(FstdcError,"Failed to open file");
        return NULL;
    }
}


static char Fstdc_fstvoi__doc__[] =
        "Print a list view a RPN Standard File rec (Interface to fstvoi)\n\
        Fstdc.fstvoi(iunit,option)\n\
        @param iunit file unit number handle returned by Fstdc_fstouv (int)\n\
        @param option 'OLDSTYLE' or 'NEWSTYLE' (sting)\n\
        @return None\n\
        @exception TypeError";

static PyObject *Fstdc_fstvoi(PyObject *self, PyObject *args) {
    char *options="NEWSTYLE";
    int iun;
    if (!PyArg_ParseTuple(args, "is",&iun,&options)) {
        return NULL;
    }
    c_fstvoi(iun,options);
    Py_INCREF(Py_None);
    return Py_None;
}


static char Fstdc_fstrwd__doc__[] =
        "Interface to fstrwd to rewind a RPN 2000 Standard File\n\
        Fstdc.fstrwd(iunit)\n\
        @param iunit unit number of the file handle, 0 for a new one (int)\n\
        @exception TypeError\n\
        @exception Fstdc.error";

static PyObject *Fstdc_fstrwd(PyObject *self, PyObject *args) {
    int iun=0,errorCode;
    if (!PyArg_ParseTuple(args, "i",&iun)) {
        return NULL;
    }
	 errorCode = c_fstrwd(iun);
    if (errorCode >= 0) {
		  Py_INCREF(Py_None);
		  return Py_None;
    } else {
        PyErr_SetString(FstdcError,"Failed to rewind file");
        return NULL;
    }
}


static char Fstdc_fstinf__doc__[] =
        "Find a record matching provided criterias (Interface to fstinf, dsfsui, fstinfx)\n\
        recParamDict = Fstdc.fstinf(iunit,nomvar,typvar,etiket,ip1,ip2,ip3,datev,inhandle)\n\
        @param iunit file unit number handle returned by Fstdc_fstouv (int)\n\
        @param nomvar seclect according to var name, blank==wildcard (string)\n\
        @param typvar seclect according to var type, blank==wildcard (string)\n\
        @param etiket seclect according to etiket, blank==wildcard (string)\n\
        @param ip1 seclect according to ip1, -1==wildcard  (int)\n\
        @param ip2 seclect according to ip2, -1==wildcard (int)\n\
        @param ip3  seclect according to ip3, -1==wildcard (int)\n\
        @param datev seclect according to date of validity, -1==wildcard (int)\n\
        @param inhandle selcation criterion; inhandle=-2:search with criterion from start of file; inhandle=-1==fstsui, use previously provided criterion to find the next matching one; inhandle>=0 search with criterion from provided rec-handle (int)\n\
        @returns python dict with record handle + record params keys/values\n\
        @exception TypeError\n\
        @exception Fstdc.error";

static PyObject *Fstdc_fstinf(PyObject *self, PyObject *args) {
    int iun, inhandle=-2, ni=0, nj=0, nk=0, datev=0, ip1=0, ip2=0, ip3=0,handle=0;
    char *typvar, *nomvar, *etiket;

    if (!PyArg_ParseTuple(args, "isssiiiii",&iun,&nomvar,&typvar,&etiket,&ip1,&ip2,&ip3,&datev,&inhandle)) {
        return NULL;
    }
    if (inhandle < -1) {
        handle = c_fstinf(iun,&ni,&nj,&nk,datev,etiket,ip1,ip2,ip3,typvar,nomvar);
    } else if (inhandle == -1) {
        handle = c_fstsui(iun,&ni,&nj,&nk);
    } else if (inhandle >= 0) {
        handle = c_fstinfx(inhandle,iun,&ni,&nj,&nk,datev,etiket,ip1,ip2,ip3,typvar,nomvar);
    }
    return c2py_fstprm(handle);
}


static PyObject *c2py_fstprm(int handle) {
    int ni=0, nj=0, nk=0, ip1=0, ip2=0, ip3=0;
    char TYPVAR[3]={' ',' ','\0'};
    char NOMVAR[5]={' ',' ',' ',' ','\0'};
    char ETIKET[13]={' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','\0'};
    char GRTYP[2]={' ','\0'};
    int dateo=0, deet=0, npas=0, nbits=0, datyp=0, ig1=0, ig2=0, ig3=0, ig4=0;
    int swa=0, lng=0, dltf=0, ubc=0, extra1=0, extra2=0, extra3=0;
    int errorCode;

    errorCode = 0;
    if (handle >= 0) {
        errorCode =  c_fstprm(handle,&dateo,&deet,&npas,&ni,&nj,&nk,&nbits,&datyp,
            &ip1,&ip2,&ip3,TYPVAR,NOMVAR,ETIKET,GRTYP,&ig1,&ig2,&ig3,&ig4,
            &swa,&lng,&dltf,&ubc,&extra1,&extra2,&extra3);
    }
    if (errorCode < 0 || handle < 0) {
        PyErr_SetString(FstdcError,"Problem getting record parameters");
        return NULL;
    }
    return Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:s,s:s,s:s,s:s,s:i,s:i,s:i,s:i}",
            "handle",handle,"ni",ni,"nj",nj,"nk",nk,"dateo",dateo,"ip1",ip1,"ip2",ip2,"ip3",ip3,
            "deet",deet,"npas",npas,"datyp",datyp,"nbits",nbits,
            "type",TYPVAR,"nom",NOMVAR,"etiket",ETIKET,"grtyp",GRTYP,
            "ig1",ig1,"ig2",ig2,"ig3",ig3,"ig4",ig4,"datev",extra1);
}


static char Fstdc_fstinl__doc__[] =
        "Find all records matching provided criterias (Interface to fstinl)\n\
        Warning: list is limited to the first 50000 records in a file, subsequent matches raises Fstdc.tooManyRecError and are ignored.\n\
        recList = Fstdc.fstinl(iunit,nomvar,typvar,etiket,ip1,ip2,ip3,datev)\n\
        param iunit file unit number handle returned by Fstdc_fstouv (int)\n\
        @param nomvar seclect according to var name, blank==wildcard (string)\n\
        @param typvar seclect according to var type, blank==wildcard (string)\n\
        @param etiket seclect according to etiket, blank==wildcard (string)\n\
        @param ip1 seclect according to ip1, -1==wildcard  (int)\n\
        @param ip2 seclect according to ip2, -1==wildcard (int)\n\
        @param ip3  seclect according to ip3, -1==wildcard (int)\n\
        @param datev seclect according to date of validity, -1==wildcard (int)\n\
        @returns python dict with handles+params of all matching records\n\
        @exception TypeError\n\
        @exception Fstdc.error\n\
        @exception Fstdc.tooManyRecError";

static PyObject *Fstdc_fstinl(PyObject *self, PyObject *args) {
    int i,iun, ier, ni=0, nj=0, nk=0, datev=0, ip1=0, ip2=0, ip3=0;
    char *typvar, *nomvar, *etiket;
    int nMatch=0,maxMatch=50000,recHandleMatchList[50000];
    PyObject *recParamList,*recParam;

    if (!PyArg_ParseTuple(args, "isssiiii",&iun,&nomvar,&typvar,&etiket,&ip1,&ip2,&ip3,&datev)) {
        return NULL;
    }
    recParamList = PyList_New(0);
    Py_INCREF(recParamList);
    ier = c_fstinl(iun, &ni, &nj, &nk,datev,etiket,ip1,ip2,ip3,typvar,nomvar,recHandleMatchList,&nMatch,maxMatch);
    if (ier<0) {
        PyErr_SetString(FstdcError,"Problem getting record list");
        return NULL;
    }
    if (nMatch == maxMatch)
        PyErr_SetString(FstdcTooManyRecError,"Reached max number of rec/match");
    for (i=0; i < nMatch; i++) {
        recParam = c2py_fstprm(recHandleMatchList[i]);
        if (recParam == NULL)
            return NULL;
        PyList_Append(recParamList,recParam);
    }
    return recParamList;
}


static char Fstdc_fstsui__doc__[] =
        "Find next record matching criterions (Interface to fstsui)\n\
        recParamDict = Fstdc.fstsui(iunit)\n\
        @param iunit file unit number handle returned by Fstdc_fstouv (int)\n\
        @returns python dict with record handle + record params keys/values\n\
        @exception TypeError\n\
        @exception Fstdc.error";

static PyObject *Fstdc_fstsui(PyObject *self, PyObject *args) {
    int iun, ni=0, nj=0, nk=0, handle;
    if (!PyArg_ParseTuple(args, "i",&iun)) {
        return NULL;
    }
    handle = c_fstsui(iun,&ni,&nj,&nk);
    return c2py_fstprm(handle);
}


static char Fstdc_fstluk__doc__[] =
        "Read record data on file (Interface to fstluk)\n\
        myRecDataDict = Fstdc.fstluk(ihandle)\n\
        @param ihandle record handle (int) \n\
        @return record data (numpy.ndarray)\n\
        @exception TypeError\n\
        @exception Fstdc.error";

static PyObject *Fstdc_fstluk(PyObject *self, PyObject *args) {
    PyArrayObject *newarray;
    int ndims=3;
    long dims[3]={1,1,1};
    int type_num=NPY_FLOAT;
    int ni=0, nj=0, nk=0, ip1=0, ip2=0, ip3=0;
    char TYPVAR[3]={' ',' ','\0'};
    char NOMVAR[5]={' ',' ',' ',' ','\0'};
    char ETIKET[13]={' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','\0'};
    char GRTYP[2]={' ','\0'};
    int handle, errorCode=-1;
    int dateo=0, deet=0, npas=0, nbits=0, datyp=0, ig1=0, ig2=0, ig3=0, ig4=0;
    int swa=0, lng=0, dltf=0, ubc=0, extra1=0, extra2=0, extra3=0;

    if (!PyArg_ParseTuple(args, "i",&handle)) {
        return NULL;
    }
    if (handle >= 0) {
        errorCode =  c_fstprm(handle,&dateo,&deet,&npas,&ni,&nj,&nk,&nbits,&datyp,
            &ip1,&ip2,&ip3,TYPVAR,NOMVAR,ETIKET,GRTYP,&ig1,&ig2,&ig3,&ig4,
            &swa,&lng,&dltf,&ubc,&extra1,&extra2,&extra3);
    }
    if (errorCode < 0 || handle < 0) {
        PyErr_SetString(FstdcError,"Problem getting record parameters");
        return NULL;
    }

    if (datyp == 0 || datyp == 2 || datyp == 4 || datyp == 130 || datyp == 132)
        type_num=NPY_INT;
    else if (datyp == 1 || datyp == 5 || datyp == 6 || datyp == 134 || datyp == 133) {
        /* csubich -- if nbits > 32, then we're dealing with double-precision
         *            data; treating it as a float will cause a segfault. */
        if (nbits > 32) {
            type_num=NPY_DOUBLE;
        } else {
            type_num=NPY_FLOAT;
        }
    }
    else if (datyp == 3 )
        type_num=NPY_CHAR;
    else {
        PyErr_SetString(FstdcError,"Unrecognized data type");
        return NULL;
    }
    dims[0] = (ni>1) ? ni : 1  ;
    dims[1] = (nj>1) ? nj : 1 ;
    dims[2] = (nk>1) ? nk : 1 ;
    if (nk>1) ndims=3;
    else if (nj>1) ndims=2;
    else ndims=1;
    newarray = (PyArrayObject*)PyArray_EMPTY(ndims, (npy_intp*)dims, type_num,
                                             withFortranOrder);
    if (newarray == NULL) {
        PyErr_SetString(FstdcError,"Problem allocating mem");
        return NULL;
    }
    errorCode = c_fstluk((void*)(newarray->data),handle,&ni,&nj,&nk);
    if (errorCode >= 0)
        return (PyObject *)newarray;
    else {
        Py_DECREF(newarray);
        PyErr_SetString(FstdcError,"Problem reading rec data");
        return NULL;
    }
}


static char Fstdc_fst_edit_dir__doc__[] =
        "Rewrite the parameters of an rec on file, data part is unchanged (Interface to fst_edit_dir)\n\
        Fstdc.fst_edit_dir(ihandle,date,deet,npas,ni,nj,nk,ip1,ip2,ip3,typvar,nomvar,etiket,grtyp,ig1,ig2,ig3,ig4,datyp)\n\
        @param ihandle record handle (int)\n\
        @param ... \n\
        @exception TypeError\n\
        @exception Fstdc.error";

static PyObject *Fstdc_fst_edit_dir(PyObject *self, PyObject *args) {
    int handle=0;
    int errorCode=0;
    int ip1=-1, ip2=-1, ip3=-1;
    char *typvar, *nomvar, *etiket, *grtyp;
    int date=-1, deet=-1, npas=-1, ig1=-1, ig2=-1, ig3=-1, ig4=-1;
    int ni=-1,nj=-1,nk=-1,datyp=-1;

    if (!PyArg_ParseTuple(args, "iiiiiiiiiissssiiiii",
            &handle,&date,&deet,&npas,&ni,&nj,&nk,&ip1,&ip2,&ip3,&typvar,&nomvar,&etiket,&grtyp,&ig1,&ig2,&ig3,&ig4,&datyp)) {
        return NULL;
    }
    errorCode = c_fst_edit_dir(handle,date,deet,npas,ni,nj,nk,ip1,ip2,ip3,typvar,nomvar,etiket,grtyp,ig1,ig2,ig3,ig4,datyp);
    if (errorCode >= 0 ) {
        Py_INCREF(Py_None);
        return Py_None;
    } else {
        PyErr_SetString(FstdcError,"Problem setting rec param/meta");
        return NULL;
    }
}


static char Fstdc_fstecr__doc__[] =
        "Write record data & meta(params) to file (Interface to fstecr), always append (no overwrite)\n\
        Fstdc.fstecr(array,iunit,nomvar,typvar,etiket,ip1,ip2,ip3,dateo,grtyp,ig1,ig2,ig3,ig4,deet,npas,nbits)\n\
        @param array data to be written to file (numpy.ndarray)\n\
        @param iunit file unit number handle returned by Fstdc_fstouv (int)\n\
        @param ... \n\
        @exception TypeError\n\
        @exception Fstdc.error";

static PyObject *Fstdc_fstecr(PyObject *self, PyObject *args) {
    int iun, ip1=0, ip2=0, ip3=0, istat;
    char *typvar, *nomvar, *etiket, *grtyp;
    int dateo=0, deet=0, npas=0, nbits=0, ig1=0, ig2=0, ig3=0, ig4=0;
    int ni=0,nj=0,nk=0,datyp=-1,rewrit=0;
    int dtl=4;
    int dims[4];
    PyArrayObject *array;
    /* See http://web-mrb.cmc.ec.gc.ca/mrb/si/eng/si/index.html --
       fst_data_length is documented to always return 0. */
    /*extern int c_fst_data_length(int);*/

    if (!PyArg_ParseTuple(args, "Oisssiiiisiiiiiiii",
                          &array,&iun,&nomvar,&typvar,&etiket,&ip1,&ip2,&ip3,&dateo,&grtyp,&ig1,&ig2,&ig3,&ig4,&deet,&npas,&nbits,&datyp)) {
        return NULL;
    }
    if (isPyFtnArrayValid(array,RPN_DT_ANY)<0 || nbits<8 || nbits>64) {
        PyErr_SetString(FstdcError,"Invalid input data/meta");
        return NULL;
    }
    getPyFtnArrayDataTypeAndLen(&datyp,&dtl,array,&nbits);
    getPyFtnArrayDims(dims,array);
    ni = dims[0];
    nj = dims[1];
    nk = dims[2];
    istat = c_fstecr((void*)(array->data),(void*)(array->data),-nbits,iun,dateo,deet,npas,
        ni,nj,nk,ip1,ip2,ip3,typvar,nomvar,etiket,grtyp,ig1,ig2,ig3,ig4,
        datyp,rewrit);
    if (istat >= 0 ) {
        Py_INCREF(Py_None);
        return Py_None;
    } else {
        PyErr_SetString(FstdcError,"Problem writing rec param/meta");
        return NULL;
    }
}


static char Fstdc_fsteff__doc__[] =
        "Erase a record (Interface to fsteff)\n\
        Fstdc.fsteff(ihandle)\n\
        @param ihandle handle of the record to erase (int)\n\
        @exception TypeError\n\
        @exception Fstdc.error";

static PyObject *Fstdc_fsteff(PyObject *self, PyObject *args) {
    int handle=0,istat=0;
    if (!PyArg_ParseTuple(args, "i",&handle)) {
        return NULL;
    }
    istat = c_fsteff(handle);
    if (istat >= 0 ) {
        Py_INCREF(Py_None);
        return Py_None;
    } else {
        PyErr_SetString(FstdcError,"Problem erasing rec");
        return NULL;
    }
}


static char Fstdc_fstfrm__doc__[] =
        "Close a RPN 2000 Standard File (Interface to fclos to close) \n\
        Fstdc_fstfrm(iunit)\n\
        @param iunit file unit number handle returned by Fstdc_fstouv (int)\n\
        @exception TypeError\n\
        @exception Fstdc.error";

static PyObject *Fstdc_fstfrm(PyObject *self, PyObject *args) {
    int iun=0,istat1,istat2;
    if (!PyArg_ParseTuple(args, "i",&iun)) {
        return NULL;
    }
    istat1 = c_fstfrm(iun);
    istat2 = c_fclos(iun);
    if (istat1 >= 0 && istat2 >= 0) {
        Py_INCREF(Py_None);
        return Py_None;
    } else {
        PyErr_SetString(FstdcError,"Problem closing file");
        return NULL;
    }
}


static char Fstdc_cxgaig__doc__[] =
        "Encode grid descriptors (Interface to cxgaig)\n\
        (ig1,ig2,ig3,ig4) = Fstdc.cxgaig(grtyp,xg1,xg2,xg3,xg4) \n\
        @param ...TODO...\n\
        @return (ig1,ig2,ig3,ig4)\n\
        @exception TypeError\n\
        @exception Fstdc.error";

static PyObject *Fstdc_cxgaig(PyObject *self, PyObject *args) {
    F77_INTEGER fig1,fig2,fig3,fig4;
    F77_REAL fxg1,fxg2,fxg3,fxg4;
    float xg1,xg2,xg3,xg4;
    char *grtyp;
    if (!PyArg_ParseTuple(args, "sffff",&grtyp,&xg1,&xg2,&xg3,&xg4)) {
        return NULL;
    }
    fxg1 = (F77_REAL)xg1;
    fxg2 = (F77_REAL)xg2;
    fxg3 = (F77_REAL)xg3;
    fxg4 = (F77_REAL)xg4;
    f77name(cxgaig)(grtyp,&fig1,&fig2,&fig3,&fig4,&fxg1,&fxg2,&fxg3,&fxg4);
    return Py_BuildValue("(iiii)",(int)fig1,(int)fig2,(int)fig3,(int)fig4);
}


static char Fstdc_cigaxg__doc__[] =
        "Decode grid descriptors (Interface to cigaxg)\n\
        (xg1,xg2,xg3,xg4) = Fstdc.cigaxg(grtyp,ig1,ig2,ig3,ig4)\n\
        @param ...TODO...\n\
        @return (xg1,xg2,xg3,xg4)\n\
        @exception TypeError\n\
        @exception Fstdc.error";

static PyObject *Fstdc_cigaxg(PyObject *self, PyObject *args) {
    int ig1,ig2,ig3,ig4;
    F77_INTEGER fig1,fig2,fig3,fig4;
    F77_REAL fxg1=0,fxg2=0,fxg3=0,fxg4=0;
    char *grtyp;
    if (!PyArg_ParseTuple(args, "siiii",&grtyp,&ig1,&ig2,&ig3,&ig4)) {
        return NULL;
    }
    fig1 = (F77_INTEGER)ig1;
    fig2 = (F77_INTEGER)ig2;
    fig3 = (F77_INTEGER)ig3;
    fig4 = (F77_INTEGER)ig4;
    f77name(cigaxg)(grtyp,&fxg1,&fxg2,&fxg3,&fxg4,&fig1,&fig2,&fig3,&fig4);
    return Py_BuildValue("(ffff)",(float)fxg1,(float)fxg2,(float)fxg3,(float)fxg4);
}



static char Fstdc_level_to_ip1__doc__[] =
        "Encode level value to ip1 (Interface to convip)\n\
        myip1List = Fstdc.level_to_ip1(level_list,kind) \n\
        @param level_list list of level values (list of float)\n\
        @param kind type of level (int)\n\
        @return [(ip1new,ip1old),...] (list of tuple of int)\n\
        @exception TypeError\n\
        @exception Fstdc.error";

static PyObject *Fstdc_level_to_ip1(PyObject *self, PyObject *args) {
    const int strglen = 30;
    int i,kind, nelm;
    F77_INTEGER fipnew, fipold, fmode, flag=0, fkind;
    F77_REAL flevel;
    char strg[strglen];
    PyObject *level_list, *ip1_list=Py_None, *item, *ipnewold_obj;

    if (!PyArg_ParseTuple(args, "Oi",&level_list,&kind)) {
        return NULL;
    }
    fkind = (F77_INTEGER)kind;
    nelm = PyList_Size(level_list);
    ip1_list = PyList_New(0);
    Py_INCREF(ip1_list);
    for (i=0; i < nelm; i++) {
        item = PyList_GetItem(level_list,i);
        flevel = (F77_REAL)PyFloat_AsDouble(item);
        fmode = 2;
        f77name(convip)(&fipnew,&flevel,&fkind,&fmode,strg,&flag,(F77_INTEGER)strglen);
        fmode = 3;
        f77name(convip)(&fipold,&flevel,&fkind,&fmode,strg,&flag,(F77_INTEGER)strglen);
        ipnewold_obj = Py_BuildValue("(l,l)",(long)fipnew,(long)fipold);
        PyList_Append(ip1_list,ipnewold_obj);
    }
    return (ip1_list);
}


static char Fstdc_ip1_to_level__doc__[] =
        "Decode ip1 to level type,value (Interface to convip)\n\
        myLevelList = Fstdc.ip1_to_level(ip1_list)\n\
        @param tuple/list of ip1 values to decode\n\
        @return list of tuple (level,kind)\n\
        @exception TypeError\n\
        @exception Fstdc.error";

static PyObject *Fstdc_ip1_to_level(PyObject *self, PyObject *args) {
    const int strglen = 30;
    int i,nelm;
    F77_REAL flevel;
    F77_INTEGER fip1, fmode, flag=0, fkind;
    char strg[strglen];
    PyObject *ip1_list=Py_None, *level_list, *item, *level_kind_obj;

    if (!PyArg_ParseTuple(args, "O",&ip1_list)) {
        return NULL;
    }
    fmode = -1;
    nelm = PyList_Size(ip1_list);
    level_list = PyList_New(0);
    Py_INCREF(level_list);
    for (i=0; i < nelm; i++) {
        item  = PyList_GetItem(ip1_list,i);
        fip1  = (F77_INTEGER)PyLong_AsLong(item);
        f77name(convip)(&fip1,&flevel,&fkind,&fmode,strg,&flag,(F77_INTEGER)strglen);
        level_kind_obj = Py_BuildValue("(f,i)",(float)flevel,(int)fkind);
        PyList_Append(level_list,level_kind_obj);
    }
    return (level_list);
}


static char Fstdc_newdate__doc__[] =
        "Convert data to/from printable format and CMC stamp (Interface to newdate)\n\
        (fdat1,fdat2,fdat3) = Fstdc.newdate(date1,date2,date3,mode)\n\
        @param ...see newdate doc... \n\
        @return tuple with converted date values ...see newdate doc...\n\
        @exception TypeError\n\
        @exception Fstdc.error";

static PyObject *Fstdc_newdate(PyObject *self, PyObject *args) {
    int date1,date2,date3,mode,istat;
    F77_INTEGER fdat1,fdat2,fdat3,fmode;
    if (!PyArg_ParseTuple(args, "iiii",&date1,&date2,&date3,&mode)) {
        return NULL;
    }
    fdat1 = (F77_INTEGER)date1;
    fdat2 = (F77_INTEGER)date2;
    fdat3 = (F77_INTEGER)date3;
    fmode = (F77_INTEGER)mode;
    istat = f77name(newdate)(&fdat1,&fdat2,&fdat3,&fmode);
    if (istat < 0) {
        PyErr_SetString(FstdcError,"Problem in newdate");
        return NULL;
    }
    return Py_BuildValue("(iii)",(int)fdat1,(int)fdat2,(int)fdat3);
}


static char Fstdc_difdatr__doc__[] =
        "Compute differenc between 2 CMC datatime stamps (Interface to difdatr)\n\
        nhours = Fstdc.difdatr(date1,date2)\n\
        @param date1 CMC datatime stamp (int)\n\
        @param date2 CMC datatime stamp (int)\n\
        @return number of hours = date2-date1 (float)\n\
        @exception TypeError";

static PyObject *Fstdc_difdatr(PyObject *self, PyObject *args) {
    int date1,date2;
    F77_REAL8 fnhours=0;
    F77_INTEGER fdat1,fdat2;
    if (!PyArg_ParseTuple(args, "ii",&date1,&date2)) {
        return NULL;
    }
    fdat1 = (F77_INTEGER)date1;
    fdat2 = (F77_INTEGER)date2;
    f77name(difdatr)(&fdat1,&fdat2,&fnhours);
    return Py_BuildValue("d",(double)fnhours);
}


static char Fstdc_incdatr__doc__[] =
"Increase CMC datetime stamp by a N hours (Interface to incdatr)\n\
        date2 = Fstdc.incdatr(date1,nhours)\n\
        @param date1 original CMC datetime stamp(int)\n\
        @param nhours number of hours to increase the date (double)\n\
        @return Increase CMC datetime stamp (int)\n\
        @exception TypeError\n\
        @exception Fstdc.error";

static PyObject *Fstdc_incdatr(PyObject *self, PyObject *args) {
    int date2,istat;
    double nhours;
    F77_INTEGER fdat1,fdat2;
    F77_REAL8 fnhours;
    if (!PyArg_ParseTuple(args, "id",&date2,&nhours)) {
        return NULL;
    }
    fdat2 = (F77_INTEGER)date2;
    fnhours = (F77_REAL8)nhours;
    istat = f77name(incdatr)(&fdat1,&fdat2,&fnhours);
    if (istat < 0) {
        PyErr_SetString(FstdcError,"Problem computing date increase");
        return NULL;
    }
    return Py_BuildValue("i",(int)fdat1);
}


static char Fstdc_datematch__doc__[] =
"Determine if date stamp match search crieterias\n\
        doesmatch = Fstdc.datematch(indate,dateRangeStart,dateRangeEnd,delta)\n\
        @param indate Date to be check against, CMC datetime stamp (int)\n\
        @param dateRangeStart, CMC datetime stamp (int) \n\
        @param dateRangeEnd, CMC datetime stamp (int)\n\
        @param delta (float)\n\
        @return 1:if date match; 0 otherwise\n\
        @exception TypeError";

static PyObject *Fstdc_datematch(PyObject *self, PyObject *args) {
    int datelu=0, debut=0, fin=0;
    float delta=0.;
    double modulo,ddelta;
    float toler=.00023; //tolerance d'erreur de 5 sec
    F77_REAL8 fnhours=0;
    F77_INTEGER fdat1,fdat2;

    if (!PyArg_ParseTuple(args, "iiif",&datelu,&debut,&fin,&delta)) {
        return NULL;
    }
	 ddelta = (double)delta;
    if (fin != -1 && datelu > fin)
        return Py_BuildValue("i",0);
    if (debut != -1) {
        if (datelu < debut)
            return Py_BuildValue("i",0);
        fdat1 = (F77_INTEGER)datelu;
        fdat2 = (F77_INTEGER)debut;
        f77name(difdatr)(&fdat1,&fdat2,&fnhours);
    } else {
        if (fin == -1)
            return Py_BuildValue("i",1);
        fdat1 = (F77_INTEGER)fin;
        fdat2 = (F77_INTEGER)datelu;
        f77name(difdatr)(&fdat1,&fdat2,&fnhours);
    }
    modulo = fmod((double)fnhours,ddelta);
    if (modulo < toler || (delta - modulo) < toler)
        return Py_BuildValue("i",1);
    else
        return Py_BuildValue("i",0);
}


static char Fstdc_ezgetlalo__doc__[] =
        "Get Lat-Lon of grid points centers and corners\n\
        (lat,lon,clat,clon) = Fstdc.ezgetlalo((niS,njS),grtypS,(grrefS,ig1S,ig2S,ig3S,ig4S),(xsS,ysS),hasSrcAxis,(i0S,j0S),doCorners)\n\
        @param ...TODO...\n\
        @return tuple of (numpy.ndarray) with center lat/lon (lat,lon) and optionally corners lat/lon (clat,clon)\n\
        @exception TypeError\n\
        @exception Fstdc.error";

static PyObject *Fstdc_ezgetlalo(PyObject *self, PyObject *args) {
    int ig1S, ig2S, ig3S, ig4S, niS, njS, i0S,j0S, gdid_src;
    int hasSrcAxis,doCorners,ier,n,nbcorners=4;
    char *grtypS,*grrefS;
    long dims[3]={1,1,1};
    int ndims=3;
    int type_num=NPY_FLOAT;
    PyArrayObject *lat,*lon,*clat,*clon,*xsS,*ysS,*x,*y,*xc,*yc;
    F77_INTEGER fni,fnj;
    if (!PyArg_ParseTuple(args, "(ii)s(siiii)(OO)i(ii)i",
            &niS,&njS,&grtypS,&grrefS,&ig1S,&ig2S,&ig3S,&ig4S,
            &xsS,&ysS,&hasSrcAxis,&i0S,&j0S,&doCorners)) {
        return NULL;
    }

    gdid_src = getGridHandle(niS,njS,grtypS,grrefS,ig1S,ig2S,ig3S,ig4S,i0S,j0S,xsS,ysS);
    if (gdid_src<0) {
        PyErr_SetString(FstdcError,"Invalid Grid Desc");
        return NULL;
    }
    niS = (niS>1) ? niS : 1 ;
    njS = (njS>1) ? njS : 1 ;
    dims[0] = niS;
    dims[1] = njS;
    dims[2] = 1 ;
    if (njS>1) ndims=2;
    else ndims=1;

    lat = (PyArrayObject*)PyArray_EMPTY(ndims, (npy_intp*)dims, type_num,
                                        withFortranOrder);
    lon = (PyArrayObject*)PyArray_EMPTY(ndims, (npy_intp*)dims, type_num,
                                        withFortranOrder);

    ier = c_gdll(gdid_src, (float *)lat->data, (float *)lon->data);
    if (ier<0) {
        Py_XDECREF(lat);
        Py_XDECREF(lon);
        PyErr_SetString(FstdcError,"Problem computing lat,lon in ezscint");
        return NULL;
    }

    //Compute corners Lat-Lon values
    if (doCorners) {
        x = (PyArrayObject*)PyArray_EMPTY(ndims, (npy_intp*)dims, type_num,
                                          withFortranOrder);
        y = (PyArrayObject*)PyArray_EMPTY(ndims, (npy_intp*)dims, type_num,
                                          withFortranOrder);
        dims[0] = nbcorners;
        dims[1] = niS;
        dims[2] = njS;
        if (njS>1) ndims=3;
        else ndims=2;

        clat = (PyArrayObject*)PyArray_EMPTY(ndims, (npy_intp*)dims, type_num,
                                             withFortranOrder);
        clon = (PyArrayObject*)PyArray_EMPTY(ndims, (npy_intp*)dims, type_num,
                                             withFortranOrder);
        xc   = (PyArrayObject*)PyArray_EMPTY(ndims, (npy_intp*)dims, type_num,
                                             withFortranOrder);
        yc   = (PyArrayObject*)PyArray_EMPTY(ndims, (npy_intp*)dims, type_num,
                                             withFortranOrder);


        n = niS*njS;
        ier = c_gdxyfll(gdid_src,(float *)x->data,(float *)y->data,(float *)lat->data,(float *)lon->data,n);
        if (ier>=0) {
            fni = (F77_INTEGER)niS;
            fnj = (F77_INTEGER)njS;
            ier = f77name(get_corners_xy)((F77_REAL *)xc->data,(F77_REAL *)yc->data,(F77_REAL *)x->data,(F77_REAL *)y->data,&fni,&fnj);
        }
        if (ier>=0) {
            n = niS*njS*nbcorners;
            ier = c_gdllfxy(gdid_src,(float *)clat->data,(float *)clon->data,(float *)xc->data,(float *)yc->data,n);
        }
        Py_XDECREF(x);
        Py_XDECREF(y);
        Py_XDECREF(xc);
        Py_XDECREF(yc);
        if (ier<0) {
            Py_XDECREF(lat);
            Py_XDECREF(lon);
            Py_XDECREF(clat);
            Py_XDECREF(clon);
            PyErr_SetString(FstdcError,"Problem computing corners lat,lon");
            return NULL;
        }

        return Py_BuildValue("OOOO",lat,lon,clat,clon);
    } else {
        return Py_BuildValue("OO",lat,lon);
    }

}


static int getGridHandle(int ni,int nj,char *grtyp,char *grref,int ig1,int ig2,int ig3,int ig4,int i0, int j0,PyArrayObject *xs,PyArrayObject *ys) {
    int gdid = -1,i0b=0,j0b=0;
    char *grtypZ = "Z";
    char *grtypY = "Y";
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
                xsd = (float *)xs->data;
                ysd = (float *)ys->data;
                gdid = c_ezgdef_fmem(ni,nj,grtypY,grref,ig1,ig2,ig3,ig4,&xsd[i0b],&ysd[j0b]);
                break;
            default:
                gdid = c_ezqkdef(ni,nj,grtyp,ig1,ig2,ig3,ig4,0);
                break;
        }
    }
    return gdid;
}


static int isGridValid(int ni,int nj,char *grtyp,char *grref,int ig1,int ig2,int ig3,int ig4,int i0, int j0,PyArrayObject *xs,PyArrayObject *ys) {
    int istat = 0,xdims[4],ydims[4];
    if (ni>0 && nj>0) {
        switch (grtyp[0]) {
            case '#':
                if ((PyObject *)xs==Py_None || (PyObject *)ys==Py_None || isPyFtnArrayValid(xs,RPN_DT_ANY)<0 || isPyFtnArrayValid(ys,RPN_DT_ANY)<0) {
                    fprintf(stderr,"ERROR: Fstdc - #-grtyp need valid axis\n");
                    istat = -1;
                } else {
                    getPyFtnArrayDims(xdims,xs);
                    getPyFtnArrayDims(ydims,ys);
                    if (i0<1 || j0<1 || (i0+ni-1)>(xdims[0]) || (j0+nj-1)>(ydims[1]) || xdims[1]>1 || ydims[0]>1) {
                        fprintf(stderr,"ERROR: Fstdc - #-grtyp need valid consistant dims: (i0,j0) = (%d,%d); (xdims0,xdims1) = (%d,%d); (ydims0,ydims1) = (%d,%d); (ni,nj) = (%d,%d)\n",i0,j0,xdims[0],xdims[1],ydims[0],ydims[1],ni,nj);
                        istat = -1;
                    }
                }
                if (istat>=0) istat = isGridTypeValid(grref);
                break;
            case 'Y':
                if ((PyObject *)xs==Py_None || (PyObject *)ys==Py_None || isPyFtnArrayValid(xs,RPN_DT_ANY)<0 || isPyFtnArrayValid(ys,RPN_DT_ANY)<0) istat = -1;
                else {
                    getPyFtnArrayDims(xdims,xs);
                    getPyFtnArrayDims(ydims,ys);
                    if (xdims[0]!=ni || xdims[1]!=nj || ydims[0]!=ni || ydims[1]!=nj) istat = -1;
                }
                if (istat>=0) istat = isGridTypeValid(grref);
                break;
            case 'Z':
                if ((PyObject *)xs==Py_None || (PyObject *)ys==Py_None || isPyFtnArrayValid(xs,RPN_DT_ANY)<0 || isPyFtnArrayValid(ys,RPN_DT_ANY)<0)  {
                    fprintf(stderr,"ERROR: Fstdc - Z-grtyp need valid axis\n");
                    istat = -1;
                } else {
                    getPyFtnArrayDims(xdims,xs);
                    getPyFtnArrayDims(ydims,ys);
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


static int isGridTypeValid(char *grtyp) {
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


static char Fstdc_ezinterp__doc__[] =
        "Interpolate from one grid to another\n\
        newArray = Fstdc.ezinterp(arrayin,arrayin2,\n  \
        (niS,njS),grtypS,(grrefS,ig1S,ig2S,ig3S,ig4S),(xsS,ysS),hasSrcAxis,(i0S,j0S),\n  \
        (niD,njD),grtypD,(grrefD,ig1D,ig2D,ig3D,ig4D),(xsD,ysD),hasDstAxis,(i0D,j0D),\n  \
        isVect)\n\
        @param ...TODO...\n\
        @return interpolated data (numpy.ndarray)\n\
        @exception TypeError\n\
        @exception Fstdc.error";

static PyObject *Fstdc_ezinterp(PyObject *self, PyObject *args) {
    int ig1S, ig2S, ig3S, ig4S, niS, njS, i0S,j0S, gdid_src;
    int ig1D, ig2D, ig3D, ig4D, niD, njD, i0D,j0D, gdid_dst;
    int hasSrcAxis,hasDstAxis,isVect,ier;
    char *grtypS,*grtypD,*grrefS,*grrefD;
    long dims[3]={1,1,1};
    int ndims=3;
    int type_num=NPY_FLOAT;
    PyArrayObject *arrayin,*arrayin2,*newarray,*newarray2,*xsS,*ysS,*xsD,*ysD;

    newarray2=NULL;                     // shut up the compiler

    if (!PyArg_ParseTuple(args, "OO(ii)s(siiii)(OO)i(ii)(ii)s(siiii)(OO)i(ii)i",
        &arrayin,&arrayin2,
        &niS,&njS,&grtypS,&grrefS,&ig1S,&ig2S,&ig3S,&ig4S,&xsS,&ysS,&hasSrcAxis,&i0S,&j0S,
        &niD,&njD,&grtypD,&grrefD,&ig1D,&ig2D,&ig3D,&ig4D,&xsD,&ysD,&hasDstAxis,&i0D,&j0D,
        &isVect)) {
        return NULL;
    }
    ier = 0;
    if (isPyFtnArrayValid(arrayin,RPN_DT_ANY)<0 || arrayin->descr->type_num != type_num)
        ier=-1;
    if (isVect
        && (isPyFtnArrayValid(arrayin2,RPN_DT_ANY)<0
        || arrayin2->descr->type_num != type_num))
        ier=-1;
    if (ier<0) {
        PyErr_SetString(FstdcError,"Input arrays should be Fortran/Continuous of type float32");
        return NULL;
    }

    gdid_src = getGridHandle(niS,njS,grtypS,grrefS,ig1S,ig2S,ig3S,ig4S,i0S,j0S,xsS,ysS);
    gdid_dst = getGridHandle(niD,njD,grtypD,grrefD,ig1D,ig2D,ig3D,ig4D,i0D,j0D,xsD,ysD);
    if (gdid_src<0 || gdid_dst<0) {
        PyErr_SetString(FstdcError,"Invalid Grid Desc");
        return NULL;
    }
    ier = c_ezdefset(gdid_dst,gdid_src);
    if (ier<0) {
        PyErr_SetString(FstdcError,"Problem defining a grid interpolation set");
        return NULL;
    }

    dims[0] = (niD>1) ? niD : 1;
    dims[1] = (njD>1) ? njD : 1;
    dims[2] = 1 ;
    if (njD>1) ndims=2;
    else ndims=1;

    newarray = (PyArrayObject*)PyArray_EMPTY(ndims, (npy_intp*)dims, type_num, withFortranOrder);
    if (isVect)
        newarray2 = (PyArrayObject*)PyArray_EMPTY(ndims, (npy_intp*)dims, type_num, withFortranOrder);
    if (newarray == NULL || (isVect && newarray2 == NULL)) {
        Py_XDECREF(newarray);
        if (isVect) {
            Py_XDECREF(newarray2);
        }
        PyErr_SetString(FstdcError,"Problem allocating mem");
        return NULL;
    }

    if (isVect) {
        ier = c_ezuvint((void *)newarray->data,(void *)newarray2->data,
            (void *)arrayin->data,(void *)arrayin2->data);
        if (ier>=0)
            return Py_BuildValue("OO",newarray,newarray2);
    } else {
        ier = c_ezsint((void *)newarray->data,(void *)arrayin->data);
        if (ier>=0)
            return Py_BuildValue("O",newarray);
    }

    Py_DECREF(newarray);
    if (isVect) {
        Py_DECREF(newarray2);
    }
    PyErr_SetString(FstdcError,"Interpolation problem in ezscint");
    return NULL;
}

/*
static char Fstdc_mapdscrpt__doc__[] =
"Interface to get map descriptors for use with PyNGL\nmyMapDescDict = Fstdc_mapdscrpt(x1,y1,x2,y2,ni,nj,cgrtyp,ig1,ig2,ig3,ig4)\n@param ...TODO... \n@return python dict with keys/values";

static PyObject *Fstdc_mapdscrpt(PyObject *self, PyObject *args) {
    int ig1, ig2, ig3, ig4, one=1, ni, nj, proj;
    char *cgrtyp;
    float x1,y1, x2,y2, polat,polong, rot, lat1,lon1, lat2,lon2;

    if (!PyArg_ParseTuple(args, "ffffiisiiii",&x1,&y1,&x2,&y2,&ni,&nj,&cgrtyp,&ig1,&ig2,&ig3,&ig4)) {
        return NULL;
    }
#if defined(DEBUG)
    printf("Debug apres parse tuple cgrtyp[0]=%c\n",cgrtyp[0]);
    printf("Debug appel Mapdesc_PyNGL\n");
#endif
    f77name(mapdesc_pyngl)(cgrtyp,&one,&ig1,&ig2,&ig3,&ig4,&x1,&y1,&x2,&y2,
        &ni,&nj,&proj,&polat,&polong,&rot,&lat1,&lon1,&lat2,&lon2,1);
#if defined(DEBUG)
    printf("Fstdc.mapdscrpt ig1=%d ig2=%d ig3=%d ig4=%d\n",ig1,ig2,ig3,ig4);
    printf("Fstdc.mapdscrpt polat=%f polong=%f rot=%f, lat1=%f lon1=%f lat2=%f, lon2=%f\n",polat,polong,rot,lat1,lon1,lat2,lon2);
#endif
    return Py_BuildValue("{s:f,s:f,s:f,s:f,s:f,s:f,s:f}",
        "polat",polat,"polong",polong,"rot",rot,
        "lat1",lat1,"lon1",lon1,"lat2",lat2,"lon2",lon2);
    // return Py_BuildValue("f",rot);
}
*/
static char Fstdc_ezgetopt__doc__[] =
    "Get the string representation of one of the internal ezscint options\n\
        opt_val = Fstrc.ezgetopt(option)\n\
        @type option: A string\n\
        @param option: The option to query \n\
        @rtype: A string\n\
        @return: The string result of the query";
static PyObject *Fstdc_ezgetopt(PyObject *self, PyObject *args) {

    char upper_value[15]; // Returned option value
    char *in_option=0; // User-input option to get
    int ier = 0;

    if (!PyArg_ParseTuple(args,"s",&in_option) || !in_option) {
        // No valid arguments passed
        return NULL;
    }

    ier = c_ezgetopt(in_option,upper_value);
    if (ier != 0) {
        char error_string[100];
        snprintf(error_string,99,"ezgetopt returned with error code %d on input option %s",ier,in_option);
        PyErr_SetString(FstdcError,error_string);
        return NULL;
    }

    return Py_BuildValue("s",upper_value);
}

static char Fstdc_ezsetopt__doc__[] =
    "Get the string representation of one of the internal ezscint options\n\
        opt_val = Fstrc.ezsetopt(option,value)\n\
        @type option, value: A string\n\
        @param option: The ezscint option \n\
        @param value: The value to set";
static PyObject * Fstdc_ezsetopt(PyObject *self, PyObject *args) {

    char *in_option=0; // User-input option to set
    char *in_value=0; // User-input value
    int ier = 0;

    if (!PyArg_ParseTuple(args,"ss",&in_option,&in_value) || !in_option || !in_value) {
        // No valid arguments passed
        return NULL;
    }

    ier = c_ezsetopt(in_option,in_value);
    if (ier != 0) {
        char error_string[100];
        snprintf(error_string,99,"ezsetopt returned with error code %d on input option %s",ier,in_option);
        PyErr_SetString(FstdcError,error_string);
        return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;

}

static char Fstdc_ezgetval__doc__[] =
    "Get an internal ezscint float or integer value by keyword\n\
        opt_val = Fstdc.ezgetval(option)\n\
        @type option: A string\n\
        @param option: The keyword of the option to retrieve\n\
        @return: The value of the option as returned by ezget[i]val";

static PyObject * Fstdc_ezgetval(PyObject *self, PyObject *args) {
    char * in_option = 0; // User-input option
    float float_value = 0; // The value of a floating-point option
    int int_value = 0; // The value of an integer option

    int ier = 0; // ezget[i]val error code

    if (!PyArg_ParseTuple(args,"s",&in_option) || !in_option) {
        // No valid arguments
        return NULL;
    }

    // The C and Fortran APIs use different functions for integer
    // and floating-point values; this is unnecessary in python.
    // However, we need to call the proper function based on 
    // option keyword.  Here, we'll check against a list of known-
    // integer options and use the ival function for those; otherwise
    // we'll pass the call to the floating-point function.

    // Remember to use case-insensitive search
    if (!strcasecmp("weight_number",in_option) ||
        !strcasecmp("missing_points_tolerance",in_option)) {
        ier = c_ezgetival(in_option,&int_value);
        if (ier != 0) {
            char error_string[100];
            snprintf(error_string,99,"ezgetival returned with error code %d on input option %s",ier,in_option);
            PyErr_SetString(FstdcError,error_string);
            return NULL;
        }
        return Py_BuildValue("i",int_value);
    }
    // Otherwise we can assume we have a floating-point option
    ier = c_ezgetval(in_option,&float_value);
    if (ier != 0) {
        char error_string[100];
        snprintf(error_string,99,"ezgetval returned with error code %d on input option %s",ier,in_option);
        PyErr_SetString(FstdcError,error_string);
        return NULL;
    }
    return Py_BuildValue("f",float_value);
}
static char Fstdc_ezsetval__doc__[] =
    "Set an internal ezscint float or integer value by keyword\n\
        opt_val = Fstdc.ezgetval(option,value)\n\
        @type option: A string\n\
        @param option: The keyword of the option to retrieve\n\
        @type value: Float or integer, as appropriate for the option\n\
        @param value: The value to set";
static PyObject * Fstdc_ezsetval(PyObject *self, PyObject *args) {
    char * in_option = 0; // User-input option
    float float_value = 0; // The value of a floating-point option
    int int_value = NAN; // The value of an integer option

    int ier = 0; // ezget[i]val error code

    if (!PyArg_ParseTuple(args,"s|f",&in_option,&float_value) || !in_option) {
        // No valid arguments
        return NULL;
    }

    // The C and Fortran APIs use different functions for integer
    // and floating-point values; this is unnecessary in python.
    // However, we need to call the proper function based on 
    // option keyword.  Here, we'll check against a list of known-
    // integer options and use the ival function for those; otherwise
    // we'll pass the call to the floating-point function.

    // Remember to use case-insensitive search
    if (!strcasecmp("weight_number",in_option) ||
        !strcasecmp("missing_points_tolerance",in_option)) {
        // Re-parse the options to get an integer value
        if (!PyArg_ParseTuple(args,"si",&in_option,&int_value)) {
            return NULL;
        }
        // c_ezsetival appears to int_value by value rather than reference,
        // so do not include the address-of operator.
        ier = c_ezsetival(in_option,int_value); 
        if (ier != 0) {
            char error_string[100];
            snprintf(error_string,99,"ezsetival returned with error code %d on input option %s and value %d",ier,in_option,int_value);
            PyErr_SetString(FstdcError,error_string);
            return NULL;
        }
        Py_INCREF(Py_None);
        return Py_None;
    }
    // Otherwise we can assume we have a floating-point option
    if (!PyArg_ParseTuple(args,"sf",&in_option,&float_value) || !in_option) {
        // No valid arguments
        return NULL;
    }
    ier = c_ezsetval(in_option,&float_value);
    if (ier != 0) {
        char error_string[100];
        snprintf(error_string,99,"ezsetval returned with error code %d on input option %s and value %f",ier,in_option,float_value);
        PyErr_SetString(FstdcError,error_string);
        return NULL;
    }
    Py_INCREF(Py_None);
    return Py_None;
}




/* List of methods defined in the module */

static struct PyMethodDef Fstdc_methods[] = {
    {"version", (PyCFunction)Fstdc_version, METH_VARARGS, Fstdc_version__doc__},
    {"fstouv",	(PyCFunction)Fstdc_fstouv,	METH_VARARGS,	Fstdc_fstouv__doc__},
    {"fstrwd",	(PyCFunction)Fstdc_fstrwd,	METH_VARARGS,	Fstdc_fstrwd__doc__},
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
    //{"mapdscrpt",	(PyCFunction)Fstdc_mapdscrpt,	METH_VARARGS,	Fstdc_mapdscrpt__doc__},
    {"ezinterp",	(PyCFunction)Fstdc_ezinterp,	METH_VARARGS,	Fstdc_ezinterp__doc__},
    {"cxgaig",	(PyCFunction)Fstdc_cxgaig,	METH_VARARGS,	Fstdc_cxgaig__doc__},
    {"cigaxg",	(PyCFunction)Fstdc_cigaxg,	METH_VARARGS,	Fstdc_cigaxg__doc__},
    {"ezgetlalo",	(PyCFunction)Fstdc_ezgetlalo,	METH_VARARGS,	Fstdc_ezgetlalo__doc__},
    {"ezgetopt", (PyCFunction) Fstdc_ezgetopt, METH_VARARGS, Fstdc_ezgetopt__doc__},
    {"ezsetopt", (PyCFunction) Fstdc_ezsetopt, METH_VARARGS, Fstdc_ezsetopt__doc__},
    {"ezgetval", (PyCFunction) Fstdc_ezgetval, METH_VARARGS, Fstdc_ezgetval__doc__},
    {"ezsetval", (PyCFunction) Fstdc_ezsetval, METH_VARARGS, Fstdc_ezsetval__doc__},
    {NULL,	 (PyCFunction)NULL, 0, NULL}		/* sentinel */
};


/* Initialization function for the module (*must* be called initFstdc) */

static char Fstdc_module_documentation[] =
"Module Fstdc contains the classes used to access RPN Standard Files (rev 2000)\n@author: Mario Lepine <mario.lepine@ec.gc.ca>\n@author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>";

void initFstdc(void) {
    PyObject *m, *d;
    int istat;
    char *msglvl="MSGLVL";
    char *tolrnc="TOLRNC";

    m = Py_InitModule4("Fstdc", Fstdc_methods,
            Fstdc_module_documentation,
            (PyObject*)NULL,PYTHON_API_VERSION);

    import_array();

    FstdcError = PyErr_NewException("Fstdc.error", NULL, NULL);
    Py_INCREF(FstdcError);
    PyModule_AddObject(m, "error", FstdcError);
    FstdcTooManyRecError = PyErr_NewException("Fstdc.tooManyRecError", NULL, NULL);
    Py_INCREF(FstdcTooManyRecError);
    PyModule_AddObject(m, "tooManyRecError", FstdcTooManyRecError);

    d = PyModule_GetDict(m);
    PyDict_SetItemString(d, "LEVEL_KIND_MSL", PyInt_FromLong((long)LEVEL_KIND_MSL));
    PyDict_SetItemString(d, "LEVEL_KIND_SIG", PyInt_FromLong((long)LEVEL_KIND_SIG));
    PyDict_SetItemString(d, "LEVEL_KIND_PMB", PyInt_FromLong((long)LEVEL_KIND_PMB));
    PyDict_SetItemString(d, "LEVEL_KIND_ANY", PyInt_FromLong((long)LEVEL_KIND_ANY));
    PyDict_SetItemString(d, "LEVEL_KIND_MGL", PyInt_FromLong((long)LEVEL_KIND_MGL));
    PyDict_SetItemString(d, "LEVEL_KIND_HYB", PyInt_FromLong((long)LEVEL_KIND_HYB));
    PyDict_SetItemString(d, "LEVEL_KIND_TH", PyInt_FromLong((long)LEVEL_KIND_TH));

    PyDict_SetItemString(d, "FSTDC_FILE_RO", PyString_FromString((const char*)FSTDC_FILE_RO));
    PyDict_SetItemString(d, "FSTDC_FILE_RW", PyString_FromString((const char*)FSTDC_FILE_RW));
PyDict_SetItemString(d, "FSTDC_FILE_RW_OLD", PyString_FromString((const char*)FSTDC_FILE_RW_OLD));

//#TODO: define named Cst for newdate kinds

    istat = c_fstopi(msglvl,8,0); //8 - print fatal error messages and up;10 - print system (internal) error messages only
    istat = c_fstopi(tolrnc,6,0); //6 - tolerate warning level and lower;8 - tolerate error level and lower

    if (PyErr_Occurred())
        Py_FatalError("can't initialize module Fstdc");
}

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


/* -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*- */
// vim: set expandtab ts=4 sw=4:
// kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
