/*
Module scripc contains the classes used to use the SCRIP interpolation package
@author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
@date: 2009-09
*/
/* #define DEBUG On */
#include <rpnmacros.h>
#include <Python.h>
#include <numpy/arrayobject.h>

static char version[] = "0.1-dev";
static char lastmodified[] = "2009-09";

int c_fst_data_length(int length_type);

#define FTN_Style_Array NPY_FARRAY
//#define FTN_Style_Array 1
static PyObject *ErrorObject;

static int datyps[32]={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
                       -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
static int lentab[32]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

/* initialize the table giving the length of the array element types */
static init_lentab() {
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
printf("__dshiftr4 called\n");
exit(1);
}
void __mask4() {
printf("__mask4 called\n");
exit(1);
}
void __dshiftl4() {
printf("__dshiftl4 called\n");
exit(1);
}
#endif

static char scripc_addr_wts__doc__[] =
"Get addresses and weights for intepolation\n (fromAddr,toAddr,weights) = scripc_addr_wts(...)\n@param \n@return python tuple with 3 numpy.ndarray for (addr1,addr2,wts)";

static PyObject *
scripc_addr_wts(PyObject *self, PyObject *args) {
    PyArrayObject *addr1,*addr2,*wts;
    PyArrayObject *g1_center_lat,*g1_center_lon,*g1_corner_lat,*g1_corner_lon,
                  *g2_center_lat,*g2_center_lon,*g2_corner_lat,*g2_corner_lon;
    int nbin;
    char *methode,*typ_norm,*typ_restric;
    int dims[3]={1,1,1}, strides[3]={0,0,0}, ndims=3;
    int type_num=NPY_FLOAT;

    int istat=-1;
    wordint *faddr1,*faddr2;
    float   *fwts;
    wordint fnwts,fnlinks,fnbin,fg1_size,fg2_size,fg1_ncorn,fg2_ncorn;
    wordint fg1_dims[3]={0,0,0},fg2_dims[3]={0,0,0};

    if (!PyArg_ParseTuple(args, "OOOOOOOOisss",
            g1_center_lat,g1_center_lon,
            g1_corner_lat,g1_corner_lon,
            g2_center_lat,g2_center_lon,
            g2_corner_lat,g2_corner_lon,
            &nbin,&methode,&typ_norm,&typ_restric)) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    fg1_ncorn   = (wordint) g1_corner_lat->dimensions[0];
    fg1_dims[0] = (wordint) g1_center_lat->dimensions[0];
    fg1_size    = fg1_dims[0];
    if (g1_center_lat->dimensions[1]>0) {
        fg1_dims[1]  = (wordint) g1_center_lat->dimensions[1];
        fg1_size    *= fg1_dims[1];
    }
    fg2_ncorn   = (wordint) g2_corner_lat->dimensions[0];
    fg2_dims[0] = (wordint) g2_center_lat->dimensions[0];
    fg2_size    = fg2_dims[0];
    if (g2_center_lat->dimensions[1]>0) {
        fg2_dims[1]  = (wordint) g1_center_lat->dimensions[1];
        fg2_size    *= fg2_dims[1];
    }


    fnbin = (wordint) nbin;
    istat = f77name(scrip_addr_wts)(faddr1,faddr2,fwts,&fnwts,&fnlinks,
                &fnbin,methode,typ_norm,typ_restric,
                &fg1_size,fg1_dims,&fg1_ncorn,
                g1_center_lat->data,g1_center_lon->data,
                g1_corner_lat->data,g1_corner_lon->data,
                &fg2_size,fg2_dims,&fg2_ncorn,
                g2_center_lat->data,g2_center_lon->data,
                g2_corner_lat->data,g2_corner_lon->data);

    if(istat >= 0) {
        ndims   = 1;
        dims[0] = (int) fnlinks;
        dims[1] = 0;
        type_num=NPY_INT;
        strides[0]=sizeof(faddr1[0]); //4;
        strides[1]=0;
        strides[2]=0;
        addr1 = PyArray_NewFromDescr(&PyArray_Type,
                                    PyArray_DescrFromType(type_num),
                                    ndims,dims,
                                    strides, faddr1, FTN_Style_Array,
                                    NULL);
        addr2 = PyArray_NewFromDescr(&PyArray_Type,
                                    PyArray_DescrFromType(type_num),
                                    ndims,dims,
                                    strides, faddr2, FTN_Style_Array,
                                    NULL);
        ndims   = 2;
        dims[0] = (int) fnwts;
        dims[1] = (int) fnlinks;
        type_num=NPY_FLOAT;
        strides[0]=sizeof(fwts[0]); //4;
        strides[1]=strides[0]*dims[0];
        strides[2]=0; //strides[1]*dims[1];
        wts = PyArray_NewFromDescr(&PyArray_Type,
                                    PyArray_DescrFromType(type_num),
                                    ndims,dims,
                                    strides, fwts, FTN_Style_Array,
                                    NULL);

        return Py_BuildValue("(O,O,O)",addr1,addr2,wts);
    }

    Py_INCREF(Py_None);
    return Py_None;
}


static char scripc_addr_wts_free__doc__[] =
"Free Memory allocated for addr1,addr2,wts done in scripc_addr_wts(addr1,addr2,wts)\n@param addr1 (numpy.ndarray)\n@param addr2 (numpy.ndarray)\n@param wts (numpy.ndarray)\nreturn None";

static PyObject *
scripc_addr_wts_free(PyObject *self, PyObject *args) {
    PyArrayObject *addr1,*addr2,*wts;

    if (!PyArg_ParseTuple(args, "OOO",addr1,addr2,wts)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    //TODO: free mem for *addr1,*addr2,*wts;
    Py_INCREF(Py_None);
    return Py_None;
}


static char scripc_interp_o1__doc__[] =
"Interpolate a field using previsouly computed remapping addresses and weights\ntoData = scripc_interp_o1(fromData,fromGridAddr,toGridAddr,weights,nbpts)\n@param Field to interpolate (numpy.ndarray)\n@param Remapping FromGrid Addresses (numpy.ndarray)\n@param Remapping ToGrid Addresses (numpy.ndarray)\n@param Remapping Weights (numpy.ndarray)\nparam nbpts number of points in toGrid (int)/toData@return Interpolated field (numpy.ndarray)";

static PyObject *
scripc_interp_o1(PyObject *self, PyObject *args) {
    PyArrayObject *data,*addr1,*addr2,*wts,*data2;
    int dims[3]={1,1,1}, ndims=1;
    int type_num=NPY_FLOAT;

    int nbpts,istat=-1;
    wordint fnwts,fnlinks;

    if (!PyArg_ParseTuple(args, "OOOOi",data,addr1,addr2,wts,&nbpts)) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    //TODO: make sure data,addr,wts are F_CONTIGUOUS
    dims[0] = nbpts;
    data2 = PyArray_NewFromDescr(&PyArray_Type,
                                PyArray_DescrFromType(type_num),
                                ndims,dims,
                                NULL, NULL, FTN_Style_Array,
                                NULL);

    fnwts   = (wordint) wts->dimensions[0];
    fnlinks = (wordint) addr1->dimensions[0];
    istat = f77name(scrip_remap_o1)(data2->data,data->data,
                            wts->data,addr2->data,addr1->data,
                            &fnwts,&fnlinks);

    if(istat>=0) return Py_BuildValue("O",data2);

    Py_INCREF(Py_None);
    return Py_None;
}


/* List of methods defined in the module */

static struct PyMethodDef scripc_methods[] = {
    {"scripc_addr_wts",	(PyCFunction)scripc_addr_wts,	METH_VARARGS, scripc_addr_wts__doc__},
    {"scripc_addr_wts_free",	(PyCFunction)scripc_addr_wts_free,	METH_VARARGS, scripc_addr_wts_free__doc__},
    {"scripc_interp_o1",	(PyCFunction)scripc_interp_o1,	METH_VARARGS, scripc_interp_o1__doc__},

    {NULL,	 (PyCFunction)NULL, 0, NULL}		/* sentinel */
};


/* Initialization function for the module (*must* be called initFstdc) */

static char scripc_module_documentation[] =
"Module scripc contains the classes used to use the SCRIP interpolation package\n@author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>";

void initscripc() {
    PyObject *m, *d;

    /* Create the module and add the functions */
    m = Py_InitModule4("scripc", scripc_methods,
            scripc_module_documentation,
            (PyObject*)NULL,PYTHON_API_VERSION);

    /* Import the array object */
    import_array();

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);
    ErrorObject = PyString_FromString("scripc.error");
    PyDict_SetItemString(d, "error", ErrorObject);

    /* XXXX Add constants here */

    /* Check for errors */
    if (PyErr_Occurred())
            Py_FatalError("can't initialize module SCRIPc");
    printf("SCRIPc module V-%s (%s) initialized\n",version,lastmodified);
    init_lentab();
}

