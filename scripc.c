/*
Module scripc contains the classes used to use the SCRIP interpolation package
@author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
@date: 2009-09
*/
#include <Python.h>
#include <numpy/arrayobject.h>

#include <rpnmacros.h>

#include "../utils/py_capi_ftn_utils.h"
#include "rpn_version.h"

#include "scrip/scrip.h"

#define INPUT_GRID_NB 1
#define OUTPUT_GRID_NB 2
#define INTERP_FORWARD_NB 1
#define INTERP_BACKWARD_NB 2

static PyObject *ScripcError;

static PyObject *scripc_version(PyObject *self, PyObject *args);
static PyObject *scripc_initOptions(PyObject *self, PyObject *args);
static PyObject *scripc_setGridLatLonRad(PyObject *self, PyObject *args);
static PyObject *scripc_setGridMask(PyObject *self, PyObject *args);
static PyObject *scripc_getAddrWeights(PyObject *self, PyObject *args);
static PyObject *scripc_finalize(PyObject *self, PyObject *args);
static PyObject *scripc_interp_o1(PyObject *self, PyObject *args);


static char scripc_version__doc__[] = "(version,lastUpdate) = scripc_version()";

static PyObject *scripc_version(PyObject *self, PyObject *args) {
    return Py_BuildValue("(ss)",VERSION,LASTUPDATE);
}


static char scripc_initOptions__doc__[] = " \
    Init SCRIP with provided options.\n\
    scripc_initOptions(nbins,map_method,normalize_opt,restrict_type,nb_maps)\n\
    @param nbins         num of bins for restricted srch (int)\n\
    @param map_method    choice for mapping method/type (int)\n\
    @param normalize_opt option for normalizing weights (int)\n\
    @param restrict_type type of bins to use (int)\n\
    @param nb_maps       nb of remappings for this grid pair (int)\
    ";

static PyObject *scripc_initOptions(PyObject *self, PyObject *args) {
    int nbin,methode,typ_norm,typ_restric,nb_maps;
    wordint fnbin,fmethode,ftyp_norm,ftyp_restric,fnb_maps;

    if (!PyArg_ParseTuple(args, "iiiii",
         &nbin,&methode,&typ_norm,&typ_restric,&nb_maps)) {
        return NULL;
    }
    fnbin = (wordint)nbin;
    methode = (wordint)methode ;
    typ_norm = (wordint)typ_norm ;
    typ_restric = (wordint)typ_restric;
    nb_maps= (nb_maps<1) ? 1 : ((nb_maps>2) ? 2 : nb_maps);
    fnb_maps = (wordint)nb_maps;
    f77name(scrip_init_options)(&fnbin,&fmethode,&ftyp_norm,&ftyp_restric,&fnb_maps);
    return Py_None;
}


static char scripc_setGridLatLonRad__doc__[] = " \
    Set grid points center and corners Lat/Lon (in rad) for said grid.\n\
    scripc_setGridLatLonRad(gridNb,centerLat,centerLon,cornersLat,cornersLon)\n\
    @param gridNb INPUT_GRID or OUTPUT_GRID (int)\n\
    @param centerLat (numpy.ndarray : float32(ni,nj)\n\
    @param centerLon (numpy.ndarray : float32(ni,nj)\n\
    @param cornersLat (numpy.ndarray : float32(nbCorners,ni,nj)\n\
    @param cornersLon (numpy.ndarray : float32(nbCorners,ni,nj)\n\
    ";

static PyObject *scripc_setGridLatLonRad(PyObject *self, PyObject *args) {
    PyArrayObject *g1_center_lat,*g1_center_lon,*g1_corner_lat,*g1_corner_lon;
    int gridNb;
    wordint fg1_ncorn,fg1_size,fgridNb;
    wordint fg1_dims[3]={0,0,0};

    if (!PyArg_ParseTuple(args, "iOOOO",&gridNb,
        &g1_center_lat,&g1_center_lon, &g1_corner_lat,&g1_corner_lon)) {
        return NULL;
    }
    if (isPyFtnArrayValid(g1_center_lat,RPN_DT_FLOAT) == PYCAPIFTN_ERR ||
        isPyFtnArrayValid(g1_center_lon,RPN_DT_FLOAT) == PYCAPIFTN_ERR ||
        isPyFtnArrayValid(g1_corner_lat,RPN_DT_FLOAT) == PYCAPIFTN_ERR ||
        isPyFtnArrayValid(g1_corner_lon,RPN_DT_FLOAT) == PYCAPIFTN_ERR) {
        PyErr_SetString(PyExc_TypeError,"Invalid input Args");
        return NULL;
    }

    gridNb = (gridNb<1) ? 1 : ((gridNb>2) ? 2 : gridNb);
    fgridNb = (wordint)gridNb;
    fg1_ncorn   = (wordint) g1_corner_lat->dimensions[0];
    fg1_dims[0] = (wordint) g1_center_lat->dimensions[0];
    fg1_size    = fg1_dims[0];
    if (g1_center_lat->dimensions[1]>0) {
        fg1_dims[1]  = (wordint) g1_center_lat->dimensions[1];
        fg1_size    *= fg1_dims[1];
    }
    f77name(scrip_set_grid_latlon_rad)(
        &fgridNb,&fg1_size,fg1_dims,&fg1_ncorn,
        g1_center_lat->data,g1_center_lon->data,
        g1_corner_lat->data,g1_corner_lon->data);
    return Py_None;
}


static char scripc_setGridMask__doc__[] = " \
    Set grid Mask for said grid.\n\
    scripc_setGridMask(gridNb,gridMask)\n\
    @param gridNb INPUT_GRID or OUTPUT_GRID (int)\n\
    @param gridMask (numpy.ndarray : int(ni,nj)\n\
    ";

static PyObject *scripc_setGridMask(PyObject *self, PyObject *args) {
    PyArrayObject *gridMask;
    int gridNb;
    int istat=-1;
    wordint fg1_size,fgridNb;
    wordint fg1_dims[3]={0,0,0};

    if (!PyArg_ParseTuple(args, "iO",&gridNb,&gridMask)) {
        return NULL;
    }
    if (isPyFtnArrayValid(gridMask,RPN_DT_INT)==PYCAPIFTN_ERR) {
        PyErr_SetString(PyExc_TypeError,"Invalid input Args");
        return NULL;
    }

    gridNb = (gridNb<1) ? 1 : ((gridNb>2) ? 2 : gridNb);
    fgridNb = (wordint)gridNb;
    fg1_dims[0] = (wordint) gridMask->dimensions[0];
    fg1_size    = fg1_dims[0];
    if (gridMask->dimensions[1]>0) {
        fg1_dims[1]  = (wordint) gridMask->dimensions[1];
        fg1_size    *= fg1_dims[1];
    }
    istat = f77name(scrip_set_grid_mask)(&fgridNb,&fg1_size,gridMask->data);
    if (istat<0) {
        PyErr_SetString(PyExc_TypeError,"Problem setting Mask - probably incompatible dims with previously provided grid lat/lon");
    }
    return Py_None;
}


static char scripc_getAddrWeights__doc__[] = " \
    Get addresses and weights for intepolation\n \
    (fromAddr,toAddr,weights) =  getAddrWeights(mapNb)\n \
    @param mapNb INTERP_FORWARD or INTERP_BACKWARD(int)\n \
    @return fromAddr [numpy.ndarray: int(nlinks)]\n\
    @return toAddr   [numpy.ndarray: int(nlinks)]\n\
    @return weights  [numpy.ndarray: float32(nWeights,nlinks)]\
    ";

static PyObject *scripc_getAddrWeights(PyObject *self, PyObject *args) {
    PyArrayObject *addr1,*addr2,*wts;
    int mapNb,dims[3]={1,1,1},ndims=3;
    int type_num=NPY_FLOAT;
    int istat=-1;
    wordint fmapNb,fnwts,fnlinks,ferrorCode;

    if (!PyArg_ParseTuple(args, "i",&mapNb)) {
        return NULL;
    }
    mapNb = (mapNb<1) ? 1 : ((mapNb>2) ? 2 : mapNb);
    fmapNb = (wordint)mapNb;

    istat = f77name(scrip_compute_addr_wts)();
    f77name(scrip_get_dims)(&fmapNb,&fnwts,&fnlinks);
    if(istat<0 || fnwts<1 || fnlinks<1) {
        PyErr_SetString(ScripcError,"Problem computing Addresses and Weights");
        return NULL;
    }

    ndims   = 1;
    dims[0] = (int)fnlinks;
    dims[1] = 0;
    type_num=NPY_INT;
    addr1 = PyArray_NewFromDescr(&PyArray_Type,PyArray_DescrFromType(type_num),
                                ndims,dims,NULL, NULL, FTN_Style_Array,NULL);
    addr2 = PyArray_NewFromDescr(&PyArray_Type,PyArray_DescrFromType(type_num),
                                ndims,dims,NULL, NULL, FTN_Style_Array,NULL);
    ndims   = 2;
    dims[0] = (int)fnwts;
    dims[1] = (int)fnlinks;
    type_num=NPY_FLOAT;
    wts = PyArray_NewFromDescr(&PyArray_Type,PyArray_DescrFromType(type_num),
                                ndims,dims,NULL, NULL, FTN_Style_Array,NULL);
    if(addr1==NULL || addr1==NULL || wts==NULL) {
        Py_DECREF(addr1);
        Py_DECREF(addr1);
        Py_DECREF(wts);
        return NULL;
    }

    f77name(scrip_get_addr_wts)(&fmapNb,&fnwts,&fnlinks, &
            addr1->data,addr2->data,wts->data, &ferrorCode);
    if(ferrorCode<0) {
        PyErr_SetString(ScripcError,"Problem computing Addresses and Weights");
        return NULL;
    }
    return Py_BuildValue("(O,O,O)",addr1,addr2,wts);
}


static char scripc_finalize__doc__[] = "Free scripc allocated memory";

static PyObject *scripc_finalize(PyObject *self, PyObject *args) {
    f77name(scrip_finalize)();
    return Py_None;
}


static char scripc_interp_o1__doc__[] =
    "Interpolate a field using previsouly computed remapping addresses and weights\n\
        toData = scripc_interp_o1(fromData,fromGridAddr,toGridAddr,weights,nbpts)\n\
        @param fromData Field to interpolate (numpy.ndarray)\n\
        @param Remapping FromGrid Addresses (numpy.ndarray)\n\
        @param Remapping ToGrid Addresses (numpy.ndarray)\n\
        @param Remapping Weights (numpy.ndarray)\n\
        @param nbpts number of points in toGrid (int)\n\
        @return toData Interpolated field (numpy.ndarray)";

static PyObject *scripc_interp_o1(PyObject *self, PyObject *args) {
    PyArrayObject *data,*addr1,*addr2,*wts,*data2;
    int dims[3]={1,1,1}, ndims=1;
    int type_num=NPY_FLOAT;

    int nbpts,srcSize;
    wordint fnwts,fnlinks,fsrcSize,fnbpts;

    if (!PyArg_ParseTuple(args, "OOOOi",&data,&addr1,&addr2,&wts,&nbpts)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    if (isPyFtnArrayValid(data,RPN_DT_FLOAT)==PYCAPIFTN_ERR ||
        isPyFtnArrayValid(addr1,RPN_DT_INT)==PYCAPIFTN_ERR ||
        isPyFtnArrayValid(addr2,RPN_DT_INT)==PYCAPIFTN_ERR ||
        isPyFtnArrayValid(wts,RPN_DT_FLOAT)==PYCAPIFTN_ERR ||
        nbpts<1) {
        PyErr_SetString(PyExc_TypeError,"Invalid input Args");
        return NULL;
    }
    //TODO: check nbpts
    //TODO: check dims of data compared to addr1,addr2,wts

    dims[0] = nbpts;
    data2 = PyArray_NewFromDescr(&PyArray_Type,PyArray_DescrFromType(type_num),
                                ndims,dims,NULL, NULL, FTN_Style_Array,NULL);
    if (data2 == NULL) return NULL;

    fnwts   = (wordint) wts->dimensions[0];
    fnlinks = (wordint) addr1->dimensions[0];
    srcSize = data->dimensions[0];
    if (data->nd > 1) srcSize *= data->dimensions[1];
    fnbpts = (wordint)nbpts;
    fsrcSize  = (wordint)srcSize;
    f77name(scrip_interpol)(data2->data,data->data,
                            addr2->data,addr1->data,wts->data,
                            &fnwts,&fnlinks,fnbpts,fsrcSize);
    return Py_BuildValue("O",data2);
}


/* List of methods defined in the module */

static struct PyMethodDef scripc_methods[] = {
    {"version", (PyCFunction)scripc_version, METH_VARARGS, scripc_version__doc__},
    {"initOptions", (PyCFunction)scripc_initOptions, METH_VARARGS, scripc_initOptions__doc__},
    {"setGridLatLonRad", (PyCFunction)scripc_setGridLatLonRad, METH_VARARGS, scripc_setGridLatLonRad__doc__},
    {"setGridMask", (PyCFunction)scripc_setGridMask, METH_VARARGS, scripc_setGridMask__doc__},
    {"getAddrWeights", (PyCFunction)scripc_getAddrWeights, METH_VARARGS, scripc_getAddrWeights__doc__},
    {"finalize", (PyCFunction)scripc_finalize, METH_VARARGS, scripc_finalize__doc__},
    {"interp_o1", (PyCFunction)scripc_interp_o1, METH_VARARGS, scripc_interp_o1__doc__},

    {NULL,	 (PyCFunction)NULL, 0, NULL}		/* sentinel */
};


/* Initialization function for the module (*must* be called initName) */

static char scripc_module_documentation[] = "\
        Module scripc contains the classes needed to use the SCRIP interpolation package\n\
        @author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>";

void initscripc() {
    PyObject *m, *d;

    m = Py_InitModule4("scripc", scripc_methods,
            scripc_module_documentation,
            (PyObject*)NULL,PYTHON_API_VERSION);

    import_array();

    ScripcError = PyString_FromString("scripc.error");
    Py_INCREF(ScripcError);
    PyModule_AddObject(m, "error", ScripcError);

    d = PyModule_GetDict(m);
    PyDict_SetItemString(d, "INPUT_GRID", PyInt_FromLong((long)INPUT_GRID_NB));
    PyDict_SetItemString(d, "OUTPUT_GRID", PyInt_FromLong((long)OUTPUT_GRID_NB));
    PyDict_SetItemString(d, "INTERP_FORWARD", PyInt_FromLong((long)INTERP_FORWARD_NB));
    PyDict_SetItemString(d, "INTERP_BACKWARD", PyInt_FromLong((long)INTERP_BACKWARD_NB));
    PyDict_SetItemString(d, "TYPE_CONVERV", PyInt_FromLong((long)SCRIP_CONSERVATIVE));
    PyDict_SetItemString(d, "TYPE_BILINEAR", PyInt_FromLong((long)SCRIP_BILINEAR));
    PyDict_SetItemString(d, "TYPE_BICUBIC", PyInt_FromLong((long)SCRIP_BICUBIC));
    PyDict_SetItemString(d, "TYPE_DISTWGT", PyInt_FromLong((long)SCRIP_DISTWGT));
    PyDict_SetItemString(d, "NORM_NONE", PyInt_FromLong((long)SCRIP_NORM_NONE));
    PyDict_SetItemString(d, "NORM_DESTAREA", PyInt_FromLong((long)SCRIP_NORM_DESTAREA));
    PyDict_SetItemString(d, "NORM_FRACAREA", PyInt_FromLong((long)SCRIP_NORM_FRACAREA));
    PyDict_SetItemString(d, "RESTRICT_LAT", PyInt_FromLong((long)SCRIP_RESTRICT_LAT));
    PyDict_SetItemString(d, "RESTRICT_LALO", PyInt_FromLong((long)SCRIP_RESTRICT_LALO));

    if (PyErr_Occurred()) Py_FatalError("Cannot initialize module SCRIPc");
}

/* -*- Mode: C; tab-width: 4 -*- ; indent-tabs-mode: nil -*- */
// vim: set expandtab ts=4 sw=4:
// kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
