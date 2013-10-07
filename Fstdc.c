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
#include "convert_ip.h"

#include <strings.h>

#include "utils/py_capi_ftn_utils.h"
#include "utils/get_corners_xy.h"
#include "rpn_version.h"

#define FSTDC_FILE_RW "RND+R/W"
#define FSTDC_FILE_RW_OLD "RND+R/W+OLD"
#define FSTDC_FILE_RO "RND+R/O"

#define LEVEL_KIND_MSL KIND_ABOVE_SEA
#define LEVEL_KIND_SIG KIND_SIGMA
#define LEVEL_KIND_PMB KIND_PRESSURE
#define LEVEL_KIND_ANY KIND_ARBITRARY
#define LEVEL_KIND_MGL KIND_ABOVE_GND
#define LEVEL_KIND_HYB KIND_HYBRID
#define LEVEL_KIND_TH KIND_THETA
#define LEVEL_KIND_MPR KIND_M_PRES
#define TIME_KIND_HR KIND_HOURS

#define CONVIP_STYLE_DEFAULT 1
#define CONVIP_STYLE_NEW 2
#define CONVIP_STYLE_OLD 3
#define CONVIP_IP2P_DEFAULT -1
#define CONVIP_IP2P_31BITS 0

#define NEWDATE_PRINT2TRUE 2
#define NEWDATE_TRUE2PRINT -2
#define NEWDATE_PRINT2STAMP 3
#define NEWDATE_STAMP2PRINT -3

//TODO: define other named Cst for newdate modes

/* Provide some fallback definitions for older numpy API versions;
   as of this writing (Dec 2013) the standard version of numpy
   installed is 1.3, which is quite out of date.  Much of the
   development since, however, has relied on features present in
   numpy version 1.7 (which also deprecates the older API).  For
   compatibility, provide wrappers for missing pieces of the
   API as needed; this is mostly in array flags processing and
   array creation */
#if NPY_API_VERSION < 7
   /* Numpy 1.7 introduced NPY_ARRAY prefixes for #defined constants
      as the new preferred form */
   #ifndef NPY_ARRAY_C_CONTIGUOUS
      #define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
   #endif
   #ifndef NPY_ARRAY_F_CONTIGUOUS
      #define NPY_ARRAY_F_CONTIGUOUS NPY_F_CONTIGUOUS
   #endif
   #ifndef NPY_KEEPORDER
      #define NPY_KEEPORDER 2
   #endif
   #if NPY_API_VERSION < 6
      /* NewLikeArray doesn't exist prior to 1.6, so we have to
         fake it.  It is only called in this code with (KEEPORDER,
         NULL, 1) as parameters (to keep the parent arrays' strides,
         to not override the descriptor, and to permit subtypes),
         which aren't terribly special; we can supply such a
         reduced function here.  The logic applied is based on the
         Numpy 1.7 source's implementation of the same function */
      static PyObject * PyArray_NewLikeArray(PyArrayObject* prototype,
            NPY_ORDER order, PyArray_Descr* in_descr, int subok) {
         /* Use PyArray_NewFromDescr to create a new array matching
            the prototype's datatype and shape */
         PyArray_Descr * descr;
         if (in_descr) { // Use in_descr for type if supplied
            descr = in_descr;
         } else { // Otherwise copy the protype's
            descr = prototype->descr;
            Py_INCREF(descr);
         }
         PyTypeObject * thetype;
         if (subok) {
            thetype = Py_TYPE(prototype);
         } else {
            thetype = &PyArray_Type;
         }

         int flags = 0;
         /* The full processing of order=NPY_KEEPORDER involves complicated
            parsing of strides, which is not necessary in our limited use-
            cases (because rmnlib needs contiguous memory access); we can
            use the truncated logic based on the prototype's flags */
         switch (order) {
            case NPY_CORDER:
               flags = NPY_CORDER; break;
            case NPY_FORTRANORDER:
               flags = NPY_FORTRANORDER; break;
            case NPY_ANYORDER: // Fortran iff prototype is forder, otherwise C
               if (prototype->flags & NPY_F_CONTIGUOUS) {
                  flags = NPY_FORTRANORDER;
               }
               else {
                  flags = NPY_CORDER;
               }
               break;
            case NPY_KEEPORDER:
               if (prototype->nd <= 1 || (prototype->flags & NPY_C_CONTIGUOUS)) {
                  flags = NPY_CORDER;
               } else { // Assume fortran order
                  flags = NPY_FORTRANORDER;
               }
               break;
            default:
               PyErr_SetString(PyExc_ValueError,"Invalid order flags in call to NewLikeArray stub");
               return NULL;
         }

         return PyArray_NewFromDescr(thetype, // Data type
                                    descr, // Array descriptor
                                    prototype->nd, // number of dimensions
                                    prototype->dimensions, // size by dimension
                                    NULL, // Strides are automatically generated
                                    NULL, // Data -- allocate new memory
                                    flags, // Default flags
                                    subok ? (PyObject *) prototype : NULL);
      }
   #endif
#endif


static const int withFortranOrder = 1;

//TODO: add more Error distinctions to catch more specific things
static PyObject *FstdcError;
static PyObject *FstdcTooManyRecError;

static PyObject *Fstdc_version(PyObject *self, PyObject *args);
static PyObject *Fstdc_fstouv(PyObject *self, PyObject *args);
static PyObject *Fstdc_fstvoi(PyObject *self, PyObject *args);
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
static PyObject *Fstdc_ConvertP2Ip(PyObject *self, PyObject *args);
static PyObject *Fstdc_ConvertIp2P(PyObject *self, PyObject *args);
static PyObject *Fstdc_EncodeIp(PyObject *self, PyObject *args);
static PyObject *Fstdc_DecodeIp(PyObject *self, PyObject *args);
static PyObject *Fstdc_newdate(PyObject *self, PyObject *args);
static PyObject *Fstdc_difdatr(PyObject *self, PyObject *args);
static PyObject *Fstdc_incdatr(PyObject *self, PyObject *args);
static PyObject *Fstdc_datematch(PyObject *self, PyObject *args);
static PyObject *Fstdc_ezgetlalo(PyObject *self, PyObject *args);
static PyObject *Fstdc_ezgetopt(PyObject *self, PyObject *args);
static PyObject *Fstdc_ezsetopt(PyObject *self, PyObject *args);
static PyObject *Fstdc_ezsetval(PyObject *self, PyObject *args);
static PyObject *Fstdc_ezgetval(PyObject *self, PyObject *args);
// Add wind conversion routines
static PyObject *Fstdc_gduvfwd(PyObject *self, PyObject *args);
static PyObject *Fstdc_gdwdfuv(PyObject *self, PyObject *args);
// Thin wrappers for (x,y) <-> (lat,lon)
static PyObject *Fstdc_gdxyfll(PyObject *self, PyObject *args);
static PyObject *Fstdc_gdllfxy(PyObject *self, PyObject *args);
/* Scattered-point scalar and vector interpolation ((lat,lon)
   and (x,y) formulation).  Note that this is not a wrapper
   of a single function, since this encompasses both scalar
   and vector options */
static PyObject *Fstdc_gdllval(PyObject *self, PyObject *args);
static PyObject *Fstdc_gdxyval(PyObject *self, PyObject *args);


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
    if (!PyArg_ParseTuple(args, "iss:fstouv",&iun,&filename,&options)) {
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
        @param option 'STYLE_OLD' or 'STYLE_NEW' (sting)\n\
        @return None\n\
        @exception TypeError";

static PyObject *Fstdc_fstvoi(PyObject *self, PyObject *args) {
    char *options="STYLE_NEW";
    int iun;
    if (!PyArg_ParseTuple(args, "is:fstvoi",&iun,&options)) {
        return NULL;
    }
    c_fstvoi(iun,options);
    Py_INCREF(Py_None);
    return Py_None;
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

    if (!PyArg_ParseTuple(args, "isssiiiii:fstinf",&iun,&nomvar,&typvar,&etiket,&ip1,&ip2,&ip3,&datev,&inhandle)) {
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

    if (!PyArg_ParseTuple(args, "isssiiii:fstinl",&iun,&nomvar,&typvar,&etiket,&ip1,&ip2,&ip3,&datev)) {
        return NULL;
    }
    recParamList = PyList_New(0);
    //Py_INCREF(recParamList);
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
        // PyList_Append copies the reference (SetItem doesn't)
        Py_XDECREF(recParam);
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
    if (!PyArg_ParseTuple(args, "i:fstsui",&iun)) {
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

    if (!PyArg_ParseTuple(args, "i:fstluk",&handle)) {
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

    if (!PyArg_ParseTuple(args, "iiiiiiiiiissssiiiii:fst_edit_dir",
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

    if (!PyArg_ParseTuple(args, "Oisssiiiisiiiiiiii:fstecr",
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
    if (!PyArg_ParseTuple(args, "i:fsteff",&handle)) {
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
    if (!PyArg_ParseTuple(args, "i:fstfrm",&iun)) {
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
    if (!PyArg_ParseTuple(args, "sffff:cxgaig",&grtyp,&xg1,&xg2,&xg3,&xg4)) {
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
    if (!PyArg_ParseTuple(args, "siiii:cigaxg",&grtyp,&ig1,&ig2,&ig3,&ig4)) {
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

    if (!PyArg_ParseTuple(args, "Oi:level_to_ip1",&level_list,&kind)) {
        return NULL;
    }
    fkind = (F77_INTEGER)kind;
    nelm = PyList_Size(level_list);
    ip1_list = PyList_New(0);
    //Py_INCREF(ip1_list);
    for (i=0; i < nelm; i++) {
        item = PyList_GetItem(level_list,i);
        flevel = (F77_REAL)PyFloat_AsDouble(item);
        fmode = 2;
        f77name(convip)(&fipnew,&flevel,&fkind,&fmode,strg,&flag,(F77_INTEGER)strglen);
        fmode = 3;
        f77name(convip)(&fipold,&flevel,&fkind,&fmode,strg,&flag,(F77_INTEGER)strglen);
        ipnewold_obj = Py_BuildValue("(l,l)",(long)fipnew,(long)fipold);
        PyList_Append(ip1_list,ipnewold_obj);
        Py_XDECREF(ipnewold_obj);
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

    if (!PyArg_ParseTuple(args, "O:ip1_to_level",&ip1_list)) {
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
        Py_XDECREF(level_kind_obj);
    }
    return (level_list);
}


static char Fstdc_ConvertP2Ip__doc__[] =
        "Encoding of P (real value,kind) into IP123 RPN-STD files tags\n\
        ip123 = Fstdc.ConvertP2Ip(pvalue,pkind,istyle)\n\
        @param  pvalue, value to encode, units depends on the kind (float)\n\
        @param  pkind,  kind of pvalue (int)\n\
        @param  istyle, CONVIP_STYLE_NEW/OLD/DEFAULT (int)\n\
        @return IP encoded value (int)\n\
        @exception TypeError\n\
        @exception Fstdc.error";
static PyObject *Fstdc_ConvertP2Ip(PyObject *self, PyObject *args) {
    float pvalue; //TODO: make it a list?
    int pkind,istyle,ip123=0;

    if (!PyArg_ParseTuple(args, "fii:ConvertP2Ip",&pvalue,&pkind,&istyle)) {
        return NULL;
    }
    if (istyle != CONVIP_STYLE_DEFAULT && istyle != CONVIP_STYLE_NEW && istyle != CONVIP_STYLE_OLD) {
      PyErr_SetString(FstdcError,"Invalid style in ConvertP2Ip");
      return NULL;
    }
    if (pkind < 0 || (pkind > 6 && pkind != KIND_HOURS && pkind != KIND_SAMPLES && pkind != KIND_MTX_IND && pkind != KIND_M_PRES)) {
      PyErr_SetString(FstdcError,"Invalid pkind in ConvertP2Ip");
      return NULL;
    }
    ConvertIp(&ip123,&pvalue,&pkind,istyle);
    //TODO: check return value
    return Py_BuildValue("i",ip123);
}


static char Fstdc_ConvertIp2P__doc__[] =
        "Decoding of IP123 RPN-STD files tags into P (real values, kind)\n\
        (pvalue,pkind) = Fstdc.ConvertIp2P(ip123,imode)\n\
        @param  ip123, IP encoded value (int)\n\
        @param  imode, CONVIP_IP2P_DEFAULT or CONVIP_IP2P_31BITS (int)\n\
        @return pvalue, real decoded value, units depends on the kind (float)\n\
        @return pkind, kind of pvalue (int)\n\
        @exception TypeError\n\
        @exception Fstdc.error";
static PyObject *Fstdc_ConvertIp2P(PyObject *self, PyObject *args) {
    float pvalue=0.; //TODO: make it a list?
    int pkind=0,imode,ip123;

    if (!PyArg_ParseTuple(args, "ii:ConvertIp2P",&ip123,&imode)) {
        return NULL;
    }
    if (imode != CONVIP_IP2P_DEFAULT && imode != CONVIP_IP2P_31BITS) {
      PyErr_SetString(FstdcError,"Invalid mode in ConvertIp2P");
      return NULL;
    }
    ConvertIp(&ip123,&pvalue,&pkind,imode);
    //TODO: check return value
    return Py_BuildValue("(fi)",pvalue,pkind);
}


static char Fstdc_EncodeIp__doc__[] = 
        "Encoding of real level and time values+kind into the ip1+ip2+ip3 files tags triolets\n\
        (ip1,ip2,ip3) = Fstdc.EncodeIp(pvalues)\n\
        @param  pvalues, real level and time values/intervals, units depends on the kind\n\
                pvalues must have has the format of list/tuple of tuples\n\
                [(rp1.v1, rp1.v2, rp1.kind), (rp2.v1, rp2.v2, rp2.kind), (rp3.v1, rp3.v2, rp3.kind)]\n\
                where v1,v2 are float, kind is an int (named constant KIND_*)\n\
                RP1 must contain a level (or a pair of levels) in the atmosphere\n\
                RP2 must contain  a time (or a pair of times)\n\
                RP3 may contain anything, RP3%hi will be ignored (if RP1 or RP2 contains a pair, RP3 is ignored)\n\
                If RP1 is not a level or RP2 is not a time, Fstdc.error is raised\n\
                If RP1 and RP2 both contains a range , Fstdc.error is raised\n\
        @return IP encoded values, tuple of int\n\
        @exception TypeError\n\
        @exception Fstdc.error";
static PyObject *Fstdc_EncodeIp(PyObject *self, PyObject *args) {
  PyObject *rp123_list=Py_None, *item,*item1,*item2,*item3;
    int nelm,ip1,ip2,ip3,istat;
    ip_info rp1,rp2,rp3;

    if (!PyArg_ParseTuple(args, "((ffi)(ffi)(ffi)):EncodeIp",&rp1.v1,&rp1.v2,&rp1.kind,&rp2.v1,&rp2.v2,&rp2.kind,&rp3.v1,&rp3.v2,&rp3.kind)) {
      return NULL;
    }

    istat = EncodeIp(&ip1,&ip2,&ip3,&rp1,&rp2,&rp3);
    if (istat == CONVERT_ERROR) {
      PyErr_SetString(FstdcError,"Proleme encoding provided values to ip123 in EncodeIp");
      return NULL;
    }
    Py_BuildValue("(iii)",ip1,ip2,ip3);
}


static char Fstdc_DecodeIp__doc__[] = 
        "Decoding of ip1+ip2+ip3 files tags triolets into level+times values or interval\n\
        pvalues = Fstdc.DecodeIp([ip1,ip2,ip3])\n\
        @param  [ip1,ip2,ip3], tuple/list of int\n\
        @return pvalues, real decoded level and time values/intervals, units depends on the kind\n\
                pvalues has the format of list/tuple of tuples\n\
                [(rp1.v1, rp1.v2, rp1.kind), (rp2.v1, rp2.v2, rp2.kind), (rp3.v1, rp3.v2, rp3.kind)]\n\
                where v1,v2 are float, kind is an int (named constant KIND_*)\n\
                RP1 will contain a level (or a pair of levels in atmospheric ascending order) in the atmosphere\n\
                RP2 will contain a time (or a pair of times in ascending order)\n\
                RP3.v1 will be the same as RP3.v2 (if RP1 or RP2 contains a pair, RP3 is ignored)\n\
        @exception TypeError\n\
        @exception Fstdc.error";
static PyObject *Fstdc_DecodeIp(PyObject *self, PyObject *args) {
    PyObject *ip123_list=Py_None, *item;
    int nelm,ip1=-1,ip2=-1,ip3=-1,istat;
    ip_info rp1,rp2,rp3;

    if (!PyArg_ParseTuple(args, "(iii):DecodeIp",&ip1,&ip2,&ip3)) {
      return NULL;
    }

    /*
    if (!PyArg_ParseTuple(args, "O",&ip123_list)) {
        return NULL;
    }
    nelm = PyList_Size(ip123_list);
    if (nelm == -1) {
      if (!PyArg_ParseTuple(ip123_list, "iii",&ip1,&ip2,&ip3)) {
          return NULL;
        }
    } else if (nelm == 3) {
      item = PyList_GetItem(ip123_list,0); ip1  = (int)PyLong_AsLong(item);
      item = PyList_GetItem(ip123_list,1); ip2  = (int)PyLong_AsLong(item);
      item = PyList_GetItem(ip123_list,2); ip3  = (int)PyLong_AsLong(item);
    } else {
      PyErr_SetString(FstdcError,"DecodeIp argument should be a tuple/list with (ip1,ip2,ip3)");
      return NULL;
    }
    */

    INIT_ip_info(rp1)
    INIT_ip_info(rp2)
    INIT_ip_info(rp3)
    istat = DecodeIp(&rp1,&rp2,&rp3,ip1,ip2,ip3);
    if (istat == CONVERT_ERROR) {
      PyErr_SetString(FstdcError,"Proleme decoding ip123 in DecodeIp");
      return NULL;
    }
    return Py_BuildValue("((ffi),(ffi),(ffi))",rp1.v1,rp1.v2,rp1.kind,rp2.v1,rp2.v2,rp2.kind,rp3.v1,rp3.v2,rp3.kind);
}


static char Fstdc_newdate__doc__[] =
        "Convert data to/from printable format and CMC stamp (Interface to newdate)\n\
        (fdat1,fdat2,fdat3) = Fstdc.newdate(dat1,dat2,dat3,mode)\n\
        @param ...see newdate doc... \n\
        @return tuple with converted date values ...see newdate doc...\n\
        @exception TypeError\n\
        @exception Fstdc.error\n\
\n\
1.1 ARGUMENTS\n\
mode can take the following values:-3,-2,-1,1,2,3\n\
mode=1 : stamp to (true_date and run_number)\n\
   out  fdat1  the truedate corresponding to dat2       integer\n\
    in  dat2   cmc date-time stamp (old or new style)   integer\n\
   out  fdat3  run number of the date-time stamp        integer\n\
    in  mode   set to 1                                 integer \n\
mode=-1 : (true_date and run_number) to stamp\n\
    in  dat1   truedate to be converted                 integer\n\
   out  fdat2  cmc date-time stamp (old or new style)   integer\n\
    in  dat3   run number of the date-time stamp        integer\n\
    in  mode   set to -1                                integer\n\
mode=2 : printable to true_date\n\
   out  fdat1  true_date                                integer\n\
    in  dat2   date of the printable date (YYYYMMDD)    integer\n\
    in  dat3   time of the printable date (HHMMSShh)    integer\n\
    in  mode   set to 2                                 integer\n\
mode=-2 : true_date to printable\n\
    in  dat1   true_date                                integer\n\
   out  fdat2  date of the printable date (YYYYMMDD)    integer\n\
   out  fdat3  time of the printable date (HHMMSShh)    integer\n\
    in  mode   set to -2                                integer\n\
mode=3 : printable to stamp\n\
   out  fdat1  cmc date-time stamp (old or new style)   integer\n\
    in  dat2   date of the printable date (YYYYMMDD)    integer\n\
    in  dat3   time of the printable date (HHMMSShh)    integer\n\
    in  mode   set to 3                                 integer\n\
mode=-3 : stamp to printable\n\
    in  dat1   cmc date-time stamp (old or new style)   integer\n\
   out  fdat2  date of the printable date (YYYYMMDD)    integer\n\
   out  fdat3  time of the printable date (HHMMSShh)    integer\n\
    in  mode   set to -3                                integer\n\
mode=4 : 14 word old style DATE array to stamp and array(14)\n\
   out  fdat1  CMC date-time stamp (old or new style)   integer\n\
    in  dat2   14 word old style DATE array             integer\n\
    in  dat3   unused                                   integer\n\
    in  mode   set to 4                                 integer\n\
mode=-4 : stamp to 14 word old style DATE array\n\
    in  dat1   CMC date-time stamp (old or new style)   integer\n\
   out  fdat2  14 word old style DATE array             integer\n\
   out  fdat3  unused                                   integer\n\
    in  mode   set to -4                                integer\n\
";
static PyObject *Fstdc_newdate(PyObject *self, PyObject *args) {
    int date1,date2,date3,mode,istat;
    F77_INTEGER fdat1,fdat2,fdat3,fmode;
    if (!PyArg_ParseTuple(args, "iiii:newdate",&date1,&date2,&date3,&mode)) {
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
    if (!PyArg_ParseTuple(args, "ii:difdatr",&date1,&date2)) {
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
    if (!PyArg_ParseTuple(args, "id:incdatr",&date2,&nhours)) {
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

    if (!PyArg_ParseTuple(args, "iiif:datematch",&datelu,&debut,&fin,&delta)) {
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
    if (!PyArg_ParseTuple(args, "(ii)s(siiii)(OO)i(ii)i:ezgetlalo",
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


// Add wind conversion routines
static char Fstdc_gdwdfuv__doc__[] = 
        "Translate grid-directed winds to (magnitude,direction) pairs\n\
        (UV, WD) = Fstdc.gdwdfuv(uuin,vvin,lat,lon,\n   \
        (ni,nj),grtyp,(grref,ig1,ig2,ig3,ig4),(xs,ys),hasAxis,(i0,j0))\n\
        @param uuin, vvin -- Grid-directed wind fields\n\
        @param lat,lon -- Latitude and longitude coordinates of those winds\n\
        @param ni ... j0 -- grid definition parameters\n\
        @return (UV, WD): Wind modulus and direction as numpy.ndarray";

static PyObject *Fstdc_gdwdfuv(PyObject *self, PyObject *args) {
    // Parameters for grid definition
    int ni, nj, ig1, ig2, ig3, ig4, i0, j0;
    int hasAxis;
    char (*grtyp), (*grref);
    PyArrayObject *uuin, *vvin, *lat, *lon, *xs, *ys;

    // Return arrays
    PyArrayObject *uvout, *wdout;
    PyObject *retval = 0; // Return value

    // Grid ID
    int gdid = -1;

    // Parse input arguments
    if (!PyArg_ParseTuple(args,"OOOO(ii)s(siiii)(OO)i(ii)",
            &uuin,&vvin,&lat,&lon,&ni,&nj,&grtyp,&grref,
            &ig1,&ig2,&ig3,&ig4,&xs,&ys,&hasAxis,&i0,&j0)) {
        // If that returned 0, then there's some error with
        // the parameter list
        return NULL;
    }

    // Check input arrays for validity
    if (isPyFtnArrayValid(uuin,RPN_DT_ANY) < 0 || uuin->descr->type_num != NPY_FLOAT) {
        PyErr_SetString(FstdcError,"Parameter uuin is not a Fortran/Contiguous, float32 array");
        return NULL;
    }
    if (isPyFtnArrayValid(vvin,RPN_DT_ANY) < 0 || vvin->descr->type_num != NPY_FLOAT) {
        PyErr_SetString(FstdcError,"Parameter vvin is not a Fortran/Contiguous, float32 array");
        return NULL;
    }
    if (isPyFtnArrayValid(lat,RPN_DT_ANY) < 0 || lat->descr->type_num != NPY_FLOAT) {
        PyErr_SetString(FstdcError,"Parameter lat is not a Fortran/Contiguous, float32 array");
        return NULL;
    }
    if (isPyFtnArrayValid(lon,RPN_DT_ANY) < 0 || lon->descr->type_num != NPY_FLOAT) {
        PyErr_SetString(FstdcError,"Parameter lon is not a Fortran/Contiguous, float32 array");
        return NULL;
    }

    // Verify that all input arrays have the same sizes
    int ndim = PyArray_NDIM(uuin);
    if (ndim != PyArray_NDIM(vvin) || ndim != PyArray_NDIM(lat) || ndim != PyArray_NDIM(lon)) {
        PyErr_SetString(FstdcError,"All input arrays must have the same number of dimensions");
        return NULL;
    }
    int ii; // Loop counter
    for (ii = 0; ii < PyArray_NDIM(uuin); ii++) {
        int sz = PyArray_DIM(uuin,ii);
        if (sz != PyArray_DIM(vvin,ii) || sz != PyArray_DIM(lat,ii) ||
                sz != PyArray_DIM(lon,ii)) {
            PyErr_SetString(FstdcError,"Input array size mismatch");
            return NULL;
        }
    }

    // Verify that all arrays have the same flags.  Since this is a "dumb" converter that
    // internally works on 1D representations, we don't have to care about Fortran or C-
    // ordering, but it's important that they match.

    int aflags = PyArray_FLAGS(uuin);
    // Check for contiguous allocation
    if (!((aflags & NPY_ARRAY_C_CONTIGUOUS) || (aflags & NPY_ARRAY_F_CONTIGUOUS))) {
        PyErr_SetString(FstdcError,"Input array is not contiguous");
        return NULL;
    }
    // Check that flags match between input arrays
    if (aflags != PyArray_FLAGS(vvin) || aflags != PyArray_FLAGS(lon) ||
            aflags != PyArray_FLAGS(lat)) {
        PyErr_SetString(FstdcError,"Input array flags mismatch; check array ordering");
        return NULL;
    }

    // Allocate output arrays, using uuin as a template
    uvout = (PyArrayObject *) PyArray_NewLikeArray(uuin,NPY_KEEPORDER,NULL,1);
    wdout = (PyArrayObject *) PyArray_NewLikeArray(uuin,NPY_KEEPORDER,NULL,1);

    if (uvout == NULL || wdout == NULL) {
        PyErr_SetString(FstdcError,"Error allocating output arrays");
        Py_XDECREF(uvout); Py_XDECREF(wdout);
        return NULL;
    }


    // Get grid handle for the call to ezscint
    gdid = getGridHandle(ni,nj,grtyp,grref,ig1,ig2,ig3,ig4,i0,j0,xs,ys);
    if (gdid < 0) {
        // If gdid < 0, then the parameters passed in don't form a valid
        // grid, or alternately there was an error allocating the grid
        // by ezscint
        PyErr_SetString(FstdcError,"Invalid grid description or error in allocation");
        return NULL;
    }

    // At last, perform the wind conversion
    int ier = c_gdwdfuv(gdid,PyArray_DATA(uvout),PyArray_DATA(wdout),
                           PyArray_DATA(uuin),PyArray_DATA(vvin),
                           PyArray_DATA(lat),PyArray_DATA(lon),
                           PyArray_SIZE((PyObject *)uuin));
    if (ier < 0) {
        PyErr_SetString(FstdcError,"Error in call of gduvfwd");
        Py_XDECREF(uvout); Py_XDECREF(wdout);
    }

    // Return
    retval = Py_BuildValue("OO",uvout,wdout);
    Py_XDECREF(uvout); Py_XDECREF(wdout);
    return retval;


    return NULL;
}


static char Fstdc_gduvfwd__doc__[] = 
        "Translate (magnitude,direction) winds to grid-directed\n\
        (UV, WD) = Fstdc.gduvfwd(uvin,wdin,lat,lon,\n   \
        (ni,nj),grtyp,(grref,ig1,ig2,ig3,ig4),(xs,ys),hasAxis,(i0,j0))\n \
        @param uvin, wdin -- Grid-directed wind fields \n\
        @param lat,lon -- Latitude and longitude coordinates of those winds\n\
        @param ni ... j0 -- grid definition parameters\n\
        @return (UU, VV): Grid-directed winds as numpy.ndarray";

static PyObject *Fstdc_gduvfwd(PyObject *self, PyObject *args) {
    // Parameters for grid definition
    int ni, nj, ig1, ig2, ig3, ig4, i0, j0;
    int hasAxis;
    char (*grtyp), (*grref);
    PyArrayObject *uvin, *wdin, *lat, *lon, *xs, *ys;

    // Return arrays
    PyArrayObject *uuout, *vvout;
    PyObject *retval = 0; // Return value

    // Grid ID
    int gdid = -1;

    // Parse input arguments
    if (!PyArg_ParseTuple(args,"OOOO(ii)s(siiii)(OO)i(ii)",
            &uvin,&wdin,&lat,&lon,&ni,&nj,&grtyp,&grref,
            &ig1,&ig2,&ig3,&ig4,&xs,&ys,&hasAxis,&i0,&j0)) {
        // If that returned 0, then there's some error with
        // the parameter list
        return NULL;
    }

    // Check input arrays for validity
    if (isPyFtnArrayValid(uvin,RPN_DT_ANY) < 0 || uvin->descr->type_num != NPY_FLOAT) {
        PyErr_SetString(FstdcError,"Parameter uvin is not a Fortran/Contiguous, float32 array");
        return NULL;
    }
    if (isPyFtnArrayValid(wdin,RPN_DT_ANY) < 0 || wdin->descr->type_num != NPY_FLOAT) {
        PyErr_SetString(FstdcError,"Parameter wdin is not a Fortran/Contiguous, float32 array");
        return NULL;
    }
    if (isPyFtnArrayValid(lat,RPN_DT_ANY) < 0 || lat->descr->type_num != NPY_FLOAT) {
        PyErr_SetString(FstdcError,"Parameter lat is not a Fortran/Contiguous, float32 array");
        return NULL;
    }
    if (isPyFtnArrayValid(lon,RPN_DT_ANY) < 0 || lon->descr->type_num != NPY_FLOAT) {
        PyErr_SetString(FstdcError,"Parameter lon is not a Fortran/Contiguous, float32 array");
        return NULL;
    }

    // Verify that all input arrays have the same sizes
    int ndim = PyArray_NDIM(uvin);
    if (ndim != PyArray_NDIM(wdin) || ndim != PyArray_NDIM(lat) || ndim != PyArray_NDIM(lon)) {
        PyErr_SetString(FstdcError,"All input arrays must have the same number of dimensions");
        return NULL;
    }
    int ii; // Loop counter
    for (ii = 0; ii < PyArray_NDIM(uvin); ii++) {
        int sz = PyArray_DIM(uvin,ii);
        if (sz != PyArray_DIM(wdin,ii) || sz != PyArray_DIM(lat,ii) ||
                sz != PyArray_DIM(lon,ii)) {
            PyErr_SetString(FstdcError,"Input array size mismatch");
            return NULL;
        }
    }

    // Verify that all arrays have the same flags.  Since this is a "dumb" converter that
    // internally works on 1D representations, we don't have to care about Fortran or C-
    // ordering, but it's important that they match.

    int aflags = PyArray_FLAGS(uvin);
    // Check for contiguous allocation
    if (!((aflags & NPY_ARRAY_C_CONTIGUOUS) || (aflags & NPY_ARRAY_F_CONTIGUOUS))) {
        PyErr_SetString(FstdcError,"Input array is not contiguous");
        return NULL;
    }
    // Check that flags match between input arrays
    if (aflags != PyArray_FLAGS(wdin) || aflags != PyArray_FLAGS(lon) ||
            aflags != PyArray_FLAGS(lat)) {
        PyErr_SetString(FstdcError,"Input array flags mismatch; check array ordering");
        return NULL;
    }

    // Allocate output arrays, using uvin as a template
    uuout = (PyArrayObject *) PyArray_NewLikeArray(uvin,NPY_KEEPORDER,NULL,1);
    vvout = (PyArrayObject *) PyArray_NewLikeArray(uvin,NPY_KEEPORDER,NULL,1);

    if (uuout == NULL || vvout == NULL) {
        PyErr_SetString(FstdcError,"Error allocating output arrays");
        Py_XDECREF(uuout); Py_XDECREF(vvout);
        return NULL;
    }


    // Get grid handle for the call to ezscint
    gdid = getGridHandle(ni,nj,grtyp,grref,ig1,ig2,ig3,ig4,i0,j0,xs,ys);
    if (gdid < 0) {
        // If gdid < 0, then the parameters passed in don't form a valid
        // grid, or alternately there was an error allocating the grid
        // by ezscint
        PyErr_SetString(FstdcError,"Invalid grid description or error in allocation");
        return NULL;
    }

    // At last, perform the wind conversion
    int ier = c_gduvfwd(gdid,PyArray_DATA(uuout),PyArray_DATA(vvout),
                           PyArray_DATA(uvin),PyArray_DATA(wdin),
                           PyArray_DATA(lat),PyArray_DATA(lon),
                           PyArray_SIZE((PyObject *)uvin));
    if (ier < 0) {
        PyErr_SetString(FstdcError,"Error in call of gduvfwd");
        Py_XDECREF(uuout); Py_XDECREF(vvout);
    }

    // Return
    retval = Py_BuildValue("OO",uuout,vvout);
    Py_XDECREF(uuout); Py_XDECREF(vvout);
    return retval;


    return NULL;
}

static char Fstdc_gdllfxy__doc__[] =
        "Get (latitude,longitude) pairs cooresponding to (x,y) grid coordinates\n\
        (lat,lon) = Fstdc.gdllfxy(xin,yin,\n   \
                (ni,nj),grtyp,(grref,ig1,ig2,ig3,ig4),(xs,ys),hasAxis,(i0,j0))\n\
        @param xin,yin: input coordinates (as float32 numpy array)\n\
        @param (ni,nj) .. (i0,j0): Grid definition parameters\n\
        @return (lat,lon): Computed latitude and longitude coordinates";

static PyObject * Fstdc_gdllfxy(PyObject *self, PyObject *args) {
    // Input arrays
    PyArrayObject *xin, *yin;
    // Output arrays
    PyArrayObject *latout, *lonout;
    // Grid definition parameters
    char *grtyp, *grref;
    int ni, nj, ig1, ig2, ig3, ig4, i0, j0, hasAxis;
    PyArrayObject *xs, *ys;

    PyObject * retval = 0;

    // Grid ID
    int gdid = -1;

    // Parse input arguments
    if (!PyArg_ParseTuple(args,"OO(ii)s(siiii)(OO)i(ii)",
            &xin,&yin,&ni,&nj,&grtyp,&grref,
            &ig1,&ig2,&ig3,&ig4,&xs,&ys,&hasAxis,&i0,&j0)) {
        // If that returned 0, then there's some error with
        // the parameter list
        return NULL;
    }

    // Check input arrays for validity
    if (isPyFtnArrayValid(xin,RPN_DT_ANY) < 0 || xin->descr->type_num != NPY_FLOAT) {
        PyErr_SetString(FstdcError,"Parameter xin is not a Fortran/Contiguous, float32 array");
        return NULL;
    }
    if (isPyFtnArrayValid(yin,RPN_DT_ANY) < 0 || yin->descr->type_num != NPY_FLOAT) {
        PyErr_SetString(FstdcError,"Parameter yin is not a Fortran/Contiguous, float32 array");
        return NULL;
    }

    // Verify that all input arrays have the same sizes
    int ndim = PyArray_NDIM(xin);
    if (ndim != PyArray_NDIM(yin)) {
        PyErr_SetString(FstdcError,"All input arrays must have the same number of dimensions");
        return NULL;
    }
    int ii; // Loop counter
    for (ii = 0; ii < PyArray_NDIM(xin); ii++) {
        int sz = PyArray_DIM(xin,ii);
        if (sz != PyArray_DIM(yin,ii)) {
            PyErr_SetString(FstdcError,"Input array size mismatch");
            return NULL;
        }
    }

    // Verify that all arrays have the same flags.  Since this is a "dumb" converter that
    // internally works on 1D representations, we don't have to care about Fortran or C-
    // ordering, but it's important that they match.

    int aflags = PyArray_FLAGS(xin);
    // Check for contiguous allocation
    if (!((aflags & NPY_ARRAY_C_CONTIGUOUS) || (aflags & NPY_ARRAY_F_CONTIGUOUS))) {
        PyErr_SetString(FstdcError,"Input array is not contiguous");
        return NULL;
    }
    // Check that flags match between input arrays
    if (aflags != PyArray_FLAGS(yin)) {
        PyErr_SetString(FstdcError,"Input array flags mismatch; check array ordering");
        return NULL;
    }

    // Allocate output arrays, using uvin as a template
    latout = (PyArrayObject *) PyArray_NewLikeArray(xin,NPY_KEEPORDER,NULL,1);
    lonout = (PyArrayObject *) PyArray_NewLikeArray(xin,NPY_KEEPORDER,NULL,1);

    if (latout == NULL || lonout == NULL) {
        PyErr_SetString(FstdcError,"Error allocating output arrays");
        Py_XDECREF(latout); Py_XDECREF(lonout);
        return NULL;
    }


    // Get grid handle for the call to ezscint
    gdid = getGridHandle(ni,nj,grtyp,grref,ig1,ig2,ig3,ig4,i0,j0,xs,ys);
    if (gdid < 0) {
        // If gdid < 0, then the parameters passed in don't form a valid
        // grid, or alternately there was an error allocating the grid
        // by ezscint
        PyErr_SetString(FstdcError,"Invalid grid description or error in allocation");
        return NULL;
    }

    // At last, perform the coordinate conversion
    int ier = c_gdllfxy(gdid,PyArray_DATA(latout),PyArray_DATA(lonout),
                           PyArray_DATA(xin),PyArray_DATA(yin),
                           PyArray_SIZE((PyObject *)xin));
    if (ier < 0) {
        PyErr_SetString(FstdcError,"Error in call of gdllfxy");
        Py_XDECREF(latout); Py_XDECREF(lonout);
    }

    // Return
    retval = Py_BuildValue("OO",latout,lonout);
    Py_XDECREF(latout); Py_XDECREF(lonout);
    return retval;
}

static char Fstdc_gdxyfll__doc__[] =
        "Get (x,y) pairs cooresponding to (lat,lon) grid coordinates\n\
        (x,y) = Fstdc.gdxyfll(latin,lonin,\n   \
                (ni,nj),grtyp,(grref,ig1,ig2,ig3,ig4),(xs,ys),hasAxis,(i0,j0))\n\
        @param latin,lonin: input coordinates (as float32 numpy array)\n\
        @param (ni,nj) .. (i0,j0): Grid definition parameters\n\
        @return (x,y): Computed x and y coordinates (as floating point)";

static PyObject * Fstdc_gdxyfll(PyObject *self, PyObject *args) {
    // Input arrays
    PyArrayObject *latin, *lonin;
    // Output arrays
    PyArrayObject *xout, *yout;
    // Grid definition parameters
    char *grtyp, *grref;
    int ni, nj, ig1, ig2, ig3, ig4, i0, j0, hasAxis;
    PyArrayObject *xs, *ys;

    PyObject * retval = 0;

    // Grid ID
    int gdid = -1;

    // Parse input arguments
    if (!PyArg_ParseTuple(args,"OO(ii)s(siiii)(OO)i(ii)",
            &latin,&lonin,&ni,&nj,&grtyp,&grref,
            &ig1,&ig2,&ig3,&ig4,&xs,&ys,&hasAxis,&i0,&j0)) {
        // If that returned 0, then there's some error with
        // the parameter list
        return NULL;
    }

    // Check input arrays for validity
    if (isPyFtnArrayValid(latin,RPN_DT_ANY) < 0 || latin->descr->type_num != NPY_FLOAT) {
        PyErr_SetString(FstdcError,"Parameter latin is not a Fortran/Contiguous, float32 array");
        return NULL;
    }
    if (isPyFtnArrayValid(lonin,RPN_DT_ANY) < 0 || lonin->descr->type_num != NPY_FLOAT) {
        PyErr_SetString(FstdcError,"Parameter lonin is not a Fortran/Contiguous, float32 array");
        return NULL;
    }

    // Verify that all input arrays have the same sizes
    int ndim = PyArray_NDIM(latin);
    if (ndim != PyArray_NDIM(lonin)) {
        PyErr_SetString(FstdcError,"All input arrays must have the same number of dimensions");
        return NULL;
    }
    int ii; // Loop counter
    for (ii = 0; ii < PyArray_NDIM(latin); ii++) {
        int sz = PyArray_DIM(latin,ii);
        if (sz != PyArray_DIM(lonin,ii)) {
            PyErr_SetString(FstdcError,"Input array size mismatch");
            return NULL;
        }
    }

    // Verify that all arrays have the same flags.  Since this is a "dumb" converter that
    // internally works on 1D representations, we don't have to care about Fortran or C-
    // ordering, but it's important that they match.

    int aflags = PyArray_FLAGS(latin);
    // Check for contiguous allocation
    if (!((aflags & NPY_ARRAY_C_CONTIGUOUS) || (aflags & NPY_ARRAY_F_CONTIGUOUS))) {
        PyErr_SetString(FstdcError,"Input array is not contiguous");
        return NULL;
    }
    // Check that flags match between input arrays
    if (aflags != PyArray_FLAGS(lonin)) {
        PyErr_SetString(FstdcError,"Input array flags mismatch; check array ordering");
        return NULL;
    }

    // Allocate output arrays, using uvin as a template
    xout = (PyArrayObject *) PyArray_NewLikeArray(latin,NPY_KEEPORDER,NULL,1);
    yout = (PyArrayObject *) PyArray_NewLikeArray(latin,NPY_KEEPORDER,NULL,1);

    if (xout == NULL || yout == NULL) {
        PyErr_SetString(FstdcError,"Error allocating output arrays");
        Py_XDECREF(xout); Py_XDECREF(yout);
        return NULL;
    }


    // Get grid handle for the call to ezscint
    gdid = getGridHandle(ni,nj,grtyp,grref,ig1,ig2,ig3,ig4,i0,j0,xs,ys);
    if (gdid < 0) {
        // If gdid < 0, then the parameters passed in don't form a valid
        // grid, or alternately there was an error allocating the grid
        // by ezscint
        PyErr_SetString(FstdcError,"Invalid grid description or error in allocation");
        return NULL;
    }

    // At last, perform the coordinate conversion
    int ier = c_gdxyfll(gdid,PyArray_DATA(xout),PyArray_DATA(yout),
                           PyArray_DATA(latin),PyArray_DATA(lonin),
                           PyArray_SIZE((PyObject *)latin));
    if (ier < 0) {
        PyErr_SetString(FstdcError,"Error in call of gdxyfll");
        Py_XDECREF(xout); Py_XDECREF(yout);
    }

    // Return
    retval = Py_BuildValue("OO",xout,yout);
    Py_XDECREF(xout); Py_XDECREF(yout);
    return retval;
}

static char Fstdc_gdllval__doc__[] = 
        "Interpolate scalar or vector fields to scattered (lat,lon) points\n\
        vararg = Fstdc.gdllval(uuin,vvin,lat,lon,\n   \
        (ni,nj),grtyp,(grref,ig1,ig2,ig3,ig4),(xs,ys),hasAxis,(i0,j0))\n\
        @param (uuin,vvin): Fields to interpolate from; if vvin is None then\n \
                            perform scalar interpolation\n\
        @param lat,lon:  Latitude and longitude coordinates for interpolation\n\
        @param ni ... j0: grid definition parameters\n\
        @return Zout or (UU,VV): scalar or tuple of grid-directed, vector-interpolated\n \
                                 fields, as appropriate";

static PyObject *Fstdc_gdllval(PyObject *self, PyObject *args) {
    // Parameters for grid definition
    int ni, nj, ig1, ig2, ig3, ig4, i0, j0;
    int hasAxis;
    char (*grtyp), (*grref);
    PyArrayObject *uuin, *vvin, *lat, *lon, *xs, *ys;

    // Return arrays
    PyArrayObject *uuout = 0, *vvout = 0;
    PyObject *retval = 0; // Return value

    // Flag for vector interoplation
    int vector = 0; // Default to false

    // Grid ID
    int gdid = -1;

    // Parse input arguments
    if (!PyArg_ParseTuple(args,"OOOO(ii)s(siiii)(OO)i(ii)",
            &uuin,&vvin,&lat,&lon,&ni,&nj,&grtyp,&grref,
            &ig1,&ig2,&ig3,&ig4,&xs,&ys,&hasAxis,&i0,&j0)) {
        // If that returned 0, then there's some error with
        // the parameter list
        return NULL;
    }

    // Check for status of vvin; if it is None then we're performing
    // scalar interpolation.  Since Python None is a singular object,
    // that's done by direct comparison with the Py_None pointer
    if ((PyObject *) vvin == Py_None) {
        vector = 0; // Scalar interpolation
    } else {
        vector = 1; // Vector interpolation
    }

    // We have two sets of conditions for array validity.  uuin and
    // (for vector interpolation) vvin must conform to the grid, but
    // lat and lon are more flexible, as they're interpreted as 1D
    // arrays by rmnlib (we'll also replicate their format for output).

    if (!(PyArray_FLAGS(uuin) & NPY_ARRAY_F_CONTIGUOUS)) {
        PyErr_SetString(FstdcError,"Parameter uuin is not a Fortran-Contiguous array");
        return NULL;
    }
    if (vector && !(PyArray_FLAGS(vvin) & NPY_ARRAY_F_CONTIGUOUS)) {
        PyErr_SetString(FstdcError,"Parameter vvin is not a Fortran-Contiguous array");
        return NULL;
    }

    if (uuin->descr->type_num != NPY_FLOAT || (vector && vvin->descr->type_num != NPY_FLOAT)) {
        PyErr_SetString(FstdcError,"Interpolation arrays must be of type numpy.float32");
        return NULL;
    }

    // Check input arrays for validity
    if (isPyFtnArrayValid(lat,RPN_DT_ANY) < 0 || lat->descr->type_num != NPY_FLOAT) {
        PyErr_SetString(FstdcError,"Parameter lat is not a Contiguous, float32 array");
        return NULL;
    }
    if (isPyFtnArrayValid(lon,RPN_DT_ANY) < 0 || lon->descr->type_num != NPY_FLOAT) {
        PyErr_SetString(FstdcError,"Parameter lon is not a Contiguous, float32 array");
        return NULL;
    }

    // Verify that uuin and (if present) vvin match the grid size.
    if (PyArray_NDIM(uuin) != 2 || (vector && PyArray_NDIM(vvin) != 2)) {
        PyErr_SetString(FstdcError,"Interpolation arrays must be two dimensional and match the grid size");
        return NULL;
    }
    if (PyArray_DIM(uuin,0) != ni || PyArray_DIM(uuin,1) != nj ||
            (vector && (PyArray_DIM(vvin,0) != ni || PyArray_DIM(vvin,1) != nj))) {
        PyErr_SetString(FstdcError,"Interpolation arrays must match the grid size");
        return NULL;
    }

    if (PyArray_NDIM(lat) != PyArray_NDIM(lon)) {
        PyErr_SetString(FstdcError,"Arrays lat and lon must have the same shape");
        return NULL;
    }
    int ii; // Loop counter
    for (ii = 0; ii < PyArray_NDIM(lat); ii++) {
        if (PyArray_DIM(lat,ii) != PyArray_DIM(lon,ii)) {
            PyErr_SetString(FstdcError,"Arrays lat and lon must have the same shape");
            return NULL;
        }
    }

    // We're indifferent on whether lat and lon are fortran or C-style arrays, but they
    // must match; use the array flags to verify.  (uuin and vvin were checked above,
    // because they must be Fortran-ordered arrays).

    if (PyArray_FLAGS(lat) != PyArray_FLAGS(lon)) {
        PyErr_SetString(FstdcError,"Arrays lat and lon must have the same ordering");
        return NULL;
    }

    // Allocate output arrays, using lat as a template
    uuout = (PyArrayObject *) PyArray_NewLikeArray(lat,NPY_KEEPORDER,NULL,1);
    if (vector) {
        vvout = (PyArrayObject *) PyArray_NewLikeArray(lon,NPY_KEEPORDER,NULL,1);
    }

    if (uuout == NULL || (vector && vvout == NULL)) {
        PyErr_SetString(FstdcError,"Error allocating output arrays");
        Py_XDECREF(uuout); Py_XDECREF(vvout);
        return NULL;
    }


    // Get grid handle for the call to ezscint
    gdid = getGridHandle(ni,nj,grtyp,grref,ig1,ig2,ig3,ig4,i0,j0,xs,ys);
    if (gdid < 0) {
        // If gdid < 0, then the parameters passed in don't form a valid
        // grid, or alternately there was an error allocating the grid
        // by ezscint
        PyErr_SetString(FstdcError,"Invalid grid description or error in allocation");
        Py_XDECREF(uuout); Py_XDECREF(vvout);
        return NULL;
    }

    if (!vector) { // Scalar interpolation
        int ier = c_gdllsval(gdid, PyArray_DATA(uuout), PyArray_DATA(uuin),
                            PyArray_DATA(lat), PyArray_DATA(lon),
                            PyArray_SIZE((PyObject *) lon));
        if (ier < 0) {
            PyErr_SetString(FstdcError,"Error in call of gdllsval (scalar)");
            Py_XDECREF(uuout); Py_XDECREF(vvout);
            return NULL;
        }
        return (PyObject *) uuout;
    } else { // Vector interpolation
        int ier = c_gdllvval(gdid, PyArray_DATA(uuout), PyArray_DATA(vvout),
                                PyArray_DATA(uuin), PyArray_DATA(vvin),
                                PyArray_DATA(lat), PyArray_DATA(lon),
                                PyArray_SIZE((PyObject *) lon));
        if (ier < 0) {
            PyErr_SetString(FstdcError,"Error in call of gdllvval (vector)");
            Py_XDECREF(uuout); Py_XDECREF(vvout);
            return NULL;
        }
        retval = Py_BuildValue("OO",uuout,vvout);
        Py_XDECREF(uuout); Py_XDECREF(vvout);
        return retval;
    }
}

static char Fstdc_gdxyval__doc__[] = 
        "Interpolate scalar or vector fields to scattered (x,y) points\n\
        vararg = Fstdc.gdxyval(uuin,vvin,x,y,\n   \
        (ni,nj),grtyp,(grref,ig1,ig2,ig3,ig4),(xs,ys),hasAxis,(i0,j0))\n\
        @param (uuin,vvin): Fields to interpolate from; if vvin is None then\n \
                            perform scalar interpolation\n\
        @param x,y:  X and Y-coordinates for interpolation\n\
        @param ni ... j0: grid definition parameters\n\
        @return Zout or (UU,VV): scalar or tuple of grid-directed, vector-interpolated\n \
                                 fields, as appropriate";

static PyObject *Fstdc_gdxyval(PyObject *self, PyObject *args) {
    // Parameters for grid definition
    int ni, nj, ig1, ig2, ig3, ig4, i0, j0;
    int hasAxis;
    char (*grtyp), (*grref);
    PyArrayObject *uuin, *vvin, *x, *y, *xs, *ys;

    // Return arrays
    PyArrayObject *uuout = 0, *vvout = 0;
    PyObject *retval = 0; // Return value

    // Flag for vector interopxion
    int vector = 0; // Default to false

    // Grid ID
    int gdid = -1;

    // Parse input arguments
    if (!PyArg_ParseTuple(args,"OOOO(ii)s(siiii)(OO)i(ii)",
            &uuin,&vvin,&x,&y,&ni,&nj,&grtyp,&grref,
            &ig1,&ig2,&ig3,&ig4,&xs,&ys,&hasAxis,&i0,&j0)) {
        // If that returned 0, then there's some error with
        // the parameter list
        return NULL;
    }

    // Check for status of vvin; if it is None then we're performing
    // scalar interpolation.  Since Python None is a singular object,
    // that's done by direct comparison with the Py_None pointer
    if ((PyObject *) vvin == Py_None) {
        vector = 0; // Scalar interpolation
    } else {
        vector = 1; // Vector interpolation
    }

    // We have two sets of conditions for array validity.  uuin and
    // (for vector interpolation) vvin must conform to the grid, but
    // x and y are more flexible, as they're interpreted as 1D
    // arrays by rmnlib (we'll also replicate their format for output).

    if (!(PyArray_FLAGS(uuin) & NPY_ARRAY_F_CONTIGUOUS)) {
        PyErr_SetString(FstdcError,"Parameter uuin is not a Fortran-Contiguous array");
        return NULL;
    }
    if (vector && !(PyArray_FLAGS(vvin) & NPY_ARRAY_F_CONTIGUOUS)) {
        PyErr_SetString(FstdcError,"Parameter vvin is not a Fortran-Contiguous array");
        return NULL;
    }

    if (uuin->descr->type_num != NPY_FLOAT || (vector && vvin->descr->type_num != NPY_FLOAT)) {
        PyErr_SetString(FstdcError,"Interpolation arrays must be of type numpy.float32");
        return NULL;
    }

    // Check input arrays for validity
    if (isPyFtnArrayValid(x,RPN_DT_ANY) < 0 || x->descr->type_num != NPY_FLOAT) {
        PyErr_SetString(FstdcError,"Parameter x is not a Contiguous, float32 array");
        return NULL;
    }
    if (isPyFtnArrayValid(y,RPN_DT_ANY) < 0 || y->descr->type_num != NPY_FLOAT) {
        PyErr_SetString(FstdcError,"Parameter y is not a Contiguous, float32 array");
        return NULL;
    }

    // Verify that uuin and (if present) vvin match the grid size.
    if (PyArray_NDIM(uuin) != 2 || (vector && PyArray_NDIM(vvin) != 2)) {
        PyErr_SetString(FstdcError,"Interpolation arrays must be two dimensional and match the grid size");
        return NULL;
    }
    if (PyArray_DIM(uuin,0) != ni || PyArray_DIM(uuin,1) != nj ||
            (vector && (PyArray_DIM(vvin,0) != ni || PyArray_DIM(vvin,1) != nj))) {
        PyErr_SetString(FstdcError,"Interpolation arrays must match the grid size");
        return NULL;
    }

    if (PyArray_NDIM(x) != PyArray_NDIM(y)) {
        PyErr_SetString(FstdcError,"Arrays x and y must have the same shape");
        return NULL;
    }
    int ii; // Loop counter
    for (ii = 0; ii < PyArray_NDIM(x); ii++) {
        if (PyArray_DIM(x,ii) != PyArray_DIM(y,ii)) {
            PyErr_SetString(FstdcError,"Arrays x and y must have the same shape");
            return NULL;
        }
    }

    // We're indifferent on whether x and y are fortran or C-style arrays, but they
    // must match; use the array flags to verify.  (uuin and vvin were checked above,
    // because they must be Fortran-ordered arrays).

    if (PyArray_FLAGS(x) != PyArray_FLAGS(y)) {
        PyErr_SetString(FstdcError,"Arrays x and y must have the same ordering");
        return NULL;
    }

    // Allocate output arrays, using x as a tempxe
    uuout = (PyArrayObject *) PyArray_NewLikeArray(x,NPY_KEEPORDER,NULL,1);
    if (vector) {
        vvout = (PyArrayObject *) PyArray_NewLikeArray(y,NPY_KEEPORDER,NULL,1);
    }

    if (uuout == NULL || (vector && vvout == NULL)) {
        PyErr_SetString(FstdcError,"Error allocating output arrays");
        Py_XDECREF(uuout); Py_XDECREF(vvout);
        return NULL;
    }


    // Get grid handle for the call to ezscint
    gdid = getGridHandle(ni,nj,grtyp,grref,ig1,ig2,ig3,ig4,i0,j0,xs,ys);
    if (gdid < 0) {
        // If gdid < 0, then the parameters passed in don't form a valid
        // grid, or alternately there was an error allocating the grid
        // by ezscint
        PyErr_SetString(FstdcError,"Invalid grid description or error in allocation");
        Py_XDECREF(uuout); Py_XDECREF(vvout);
        return NULL;
    }

    if (!vector) { // Scalar interpolation
        int ier = c_gdxysval(gdid, PyArray_DATA(uuout), PyArray_DATA(uuin),
                            PyArray_DATA(x), PyArray_DATA(y),
                            PyArray_SIZE((PyObject *) y));
        if (ier < 0) {
            PyErr_SetString(FstdcError,"Error in call of gdxysval (scalar)");
            Py_XDECREF(uuout); Py_XDECREF(vvout);
            return NULL;
        }
        return (PyObject *) uuout;
    } else { // Vector interpolation
        int ier = c_gdxyvval(gdid, PyArray_DATA(uuout), PyArray_DATA(vvout),
                                PyArray_DATA(uuin), PyArray_DATA(vvin),
                                PyArray_DATA(x), PyArray_DATA(y),
                                PyArray_SIZE((PyObject *) y));
        if (ier < 0) {
            PyErr_SetString(FstdcError,"Error in call of gdxyvval (vector)");
            Py_XDECREF(uuout); Py_XDECREF(vvout);
            return NULL;
        }
        retval = Py_BuildValue("OO",uuout,vvout);
        Py_XDECREF(uuout); Py_XDECREF(vvout);
        return retval;
    }
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
    
    PyObject * retval = 0; // Object pointer for return value

    newarray2=NULL;                     // shut up the compiler

    if (!PyArg_ParseTuple(args, "OO(ii)s(siiii)(OO)i(ii)(ii)s(siiii)(OO)i(ii)i:ezinterp",
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
        if (ier>=0) {
            retval = Py_BuildValue("OO",newarray,newarray2);
            Py_XDECREF(newarray);
            Py_XDECREF(newarray2);
            return retval;
        }
    } else {
        ier = c_ezsint((void *)newarray->data,(void *)arrayin->data);
        if (ier>=0) {
            retval = Py_BuildValue("O",newarray);
            Py_XDECREF(newarray);
            return retval;
        }
    }

    Py_DECREF(newarray);
    if (isVect) {
        Py_DECREF(newarray2);
    }
    PyErr_SetString(FstdcError,"Interpolation problem in ezscint");
    return NULL;
}


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

    if (!PyArg_ParseTuple(args,"s:ezgetopt",&in_option) || !in_option) {
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

    if (!PyArg_ParseTuple(args,"ss:ezsetopt",&in_option,&in_value) || !in_option || !in_value) {
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

    if (!PyArg_ParseTuple(args,"s:ezgetval",&in_option) || !in_option) {
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
    float float_value = NAN; // The value of a floating-point option
    int int_value = 0; // The value of an integer option

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
        if (!PyArg_ParseTuple(args,"si:ezsetval",&in_option,&int_value)) {
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
    if (!PyArg_ParseTuple(args,"sf:ezsetval",&in_option,&float_value) || !in_option) {
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
    /* {"fstrwd",	(PyCFunction)Fstdc_fstrwd,	METH_VARARGS,	Fstdc_fstrwd__doc__}, */
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
    {"ConvertP2Ip",(PyCFunction)Fstdc_ConvertP2Ip,METH_VARARGS,	Fstdc_ConvertP2Ip__doc__},
    {"ConvertIp2P",(PyCFunction)Fstdc_ConvertIp2P,METH_VARARGS,	Fstdc_ConvertIp2P__doc__},
    {"EncodeIp",(PyCFunction)Fstdc_EncodeIp,METH_VARARGS,	Fstdc_EncodeIp__doc__},
    {"DecodeIp",(PyCFunction)Fstdc_DecodeIp,METH_VARARGS,	Fstdc_DecodeIp__doc__},
    {"ezinterp",	(PyCFunction)Fstdc_ezinterp,	METH_VARARGS,	Fstdc_ezinterp__doc__},
    {"cxgaig",	(PyCFunction)Fstdc_cxgaig,	METH_VARARGS,	Fstdc_cxgaig__doc__},
    {"cigaxg",	(PyCFunction)Fstdc_cigaxg,	METH_VARARGS,	Fstdc_cigaxg__doc__},
    {"ezgetlalo",	(PyCFunction)Fstdc_ezgetlalo,	METH_VARARGS,	Fstdc_ezgetlalo__doc__},
    {"ezgetopt", (PyCFunction) Fstdc_ezgetopt, METH_VARARGS, Fstdc_ezgetopt__doc__},
    {"ezsetopt", (PyCFunction) Fstdc_ezsetopt, METH_VARARGS, Fstdc_ezsetopt__doc__},
    {"ezgetval", (PyCFunction) Fstdc_ezgetval, METH_VARARGS, Fstdc_ezgetval__doc__},
    {"ezsetval", (PyCFunction) Fstdc_ezsetval, METH_VARARGS, Fstdc_ezsetval__doc__},
    {"gdwdfuv", (PyCFunction) Fstdc_gdwdfuv, METH_VARARGS, Fstdc_gdwdfuv__doc__},
    {"gduvfwd", (PyCFunction) Fstdc_gduvfwd, METH_VARARGS, Fstdc_gduvfwd__doc__},
    {"gdllfxy", (PyCFunction) Fstdc_gdllfxy, METH_VARARGS, Fstdc_gdllfxy__doc__},
    {"gdxyfll", (PyCFunction) Fstdc_gdxyfll, METH_VARARGS, Fstdc_gdxyfll__doc__},
    {"gdllval", (PyCFunction) Fstdc_gdllval, METH_VARARGS, Fstdc_gdllval__doc__},
    {"gdxyval", (PyCFunction) Fstdc_gdxyval, METH_VARARGS, Fstdc_gdxyval__doc__},
    {NULL,	 (PyCFunction)NULL, 0, NULL}		/* sentinel */
};


/* Initialization function for the module (*must* be called initFstdc) */

static char Fstdc_module_documentation[] =
"Module Fstdc contains the classes used to access RPN Standard Files (rev 2000)\n@author: Michel Valin <Michel.Valin@ec.gc.ca>\n@author: Mario Lepine <mario.lepine@ec.gc.ca>\n@author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>";

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
    PyDict_SetItemString(d, "LEVEL_KIND_MPR", PyInt_FromLong((long)LEVEL_KIND_MPR));
    PyDict_SetItemString(d, "TIME_KIND_HR", PyInt_FromLong((long)TIME_KIND_HR));

    PyDict_SetItemString(d, "KIND_ABOVE_SEA", PyInt_FromLong((long)KIND_ABOVE_SEA));
    PyDict_SetItemString(d, "KIND_SIGMA", PyInt_FromLong((long)KIND_SIGMA));
    PyDict_SetItemString(d, "KIND_PRESSURE", PyInt_FromLong((long)KIND_PRESSURE));
    PyDict_SetItemString(d, "KIND_ARBITRARY", PyInt_FromLong((long)KIND_ARBITRARY));
    PyDict_SetItemString(d, "KIND_ABOVE_GND", PyInt_FromLong((long)KIND_ABOVE_GND));
    PyDict_SetItemString(d, "KIND_HYBRID", PyInt_FromLong((long)KIND_HYBRID));
    PyDict_SetItemString(d, "KIND_THETA", PyInt_FromLong((long)KIND_THETA));
    PyDict_SetItemString(d, "KIND_HOURS", PyInt_FromLong((long)KIND_HOURS));
    PyDict_SetItemString(d, "KIND_SAMPLES", PyInt_FromLong((long)KIND_SAMPLES));
    PyDict_SetItemString(d, "KIND_MTX_IND", PyInt_FromLong((long)KIND_MTX_IND));
    PyDict_SetItemString(d, "KIND_M_PRES", PyInt_FromLong((long)KIND_M_PRES));

    PyDict_SetItemString(d, "CONVIP_STYLE_DEFAULT", PyInt_FromLong((long)CONVIP_STYLE_DEFAULT));
    PyDict_SetItemString(d, "CONVIP_STYLE_NEW", PyInt_FromLong((long)CONVIP_STYLE_NEW));
    PyDict_SetItemString(d, "CONVIP_STYLE_OLD", PyInt_FromLong((long)CONVIP_STYLE_OLD));
    PyDict_SetItemString(d, "CONVIP_IP2P_DEFAULT", PyInt_FromLong((long)CONVIP_IP2P_DEFAULT));
    PyDict_SetItemString(d, "CONVIP_IP2P_31BITS", PyInt_FromLong((long)CONVIP_IP2P_31BITS));

    PyDict_SetItemString(d, "FSTDC_FILE_RO", PyString_FromString((const char*)FSTDC_FILE_RO));
    PyDict_SetItemString(d, "FSTDC_FILE_RW", PyString_FromString((const char*)FSTDC_FILE_RW));
    PyDict_SetItemString(d, "FSTDC_FILE_RW_OLD", PyString_FromString((const char*)FSTDC_FILE_RW_OLD));

    PyDict_SetItemString(d, "NEWDATE_PRINT2TRUE", PyInt_FromLong((long)NEWDATE_PRINT2TRUE));
    PyDict_SetItemString(d, "NEWDATE_TRUE2PRINT", PyInt_FromLong((long)NEWDATE_TRUE2PRINT));
    PyDict_SetItemString(d, "NEWDATE_PRINT2STAMP", PyInt_FromLong((long)NEWDATE_PRINT2STAMP));
    PyDict_SetItemString(d, "NEWDATE_STAMP2PRINT", PyInt_FromLong((long)NEWDATE_STAMP2PRINT));

//#TODO: define other named Cst for newdate modes
//#TODO: define other named Cst for ezsetsval

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
