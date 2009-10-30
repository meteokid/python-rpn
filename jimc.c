/*
Module jim_c contains the functions used to compute JIM (Icosahedral) grid points position
@author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
@date: 2009-09
*/
#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdio.h>

#include <rpnmacros.h>

#include "utils/py_capi_ftn_utils.h"
#include "rpn_version.h"

static const int withFortranOrder = 1;

static PyObject *JimcError;

static PyObject *jimc_version(PyObject *self, PyObject *args);
static PyObject *jimc_dims(PyObject *self, PyObject *args);
static void get_jim_dims(int ndiv, int *nijh, int *nij, int *halo);
static PyObject *jimc_new_array(PyObject *self, PyObject *args);
static int new_jim_array(PyArrayObject **p_newarray, int ndiv, int nk, int onlyOneGrid);
static PyObject *jimc_grid_la_lo(PyObject *self, PyObject *args);
static PyObject *jimc_grid_corners_la_lo(PyObject *self, PyObject *args);
static PyObject *jimc_xch_halo(PyObject *self, PyObject *args);


static char jimc_version__doc__[] = "(version,lastUpdate) = jimc.version()";

static PyObject *jimc_version(PyObject *self, PyObject *args) {
    return Py_BuildValue("(ss)",VERSION,LASTUPDATE);
}


static char jimc_dims__doc__[] =
    "Create a new numpy.ndarray with right dims for JIM grid of ndiv and nk\n\
    (nijh, nij, halo) = jimc.grid_dims(ndiv,halo)\n\
    @param ndiv number of grid divisons (int)\n\
    @param halo add halo points to nij in each dirs (-1 for default jim halo)   (int)\n\
    @return (nijh,nij,halo) where nijh = nij + 2*halo   (int,int,int)";

static PyObject *jimc_dims(PyObject *self, PyObject *args) {
    int ndiv,nhalo;
    int nijh,nij,halo;

    if (!PyArg_ParseTuple(args, "ii",&ndiv,&nhalo)) {
        return NULL;
    }
    get_jim_dims(ndiv, &nijh, &nij, &halo);
    if (nhalo>0) {
        nijh = nij+2*nhalo;
        halo = nhalo;
    } else if (nhalo==0) {
        nijh = nij;
        halo = 0;
    }
    return Py_BuildValue("iii",nijh,nij,halo);
}


static void get_jim_dims(int ndiv, int *nijh, int *nij, int *halo) {
    F77_INTEGER f_ndiv,f_nij,f_nijh,f_halo;

    f_ndiv = (F77_INTEGER)ndiv;
    f_halo = 2;
    f_nijh = f77name(jim_grid_dims)(&f_ndiv,&f_halo);
    f_halo = 0;
    f_nij  = f77name(jim_grid_dims)(&f_ndiv,&f_halo);
    nijh[0] = (int)f_nijh;
    nij[0]  = (int)f_nij;
    halo[0] = (nijh[0] - nij[0])/2;
    return;
}


static char jimc_new_array__doc__[] =
    "Create a new numpy.ndarray with right dims for JIM grid of ndiv and nk\n\
    newJIMarray = jimc.new_array(ndiv,nk,igrid)\n\
    @param ndiv number of grid divisons (int)\n\
    @param igrid Icosahedral Grid tile number [all grids tiles if iGrid=0] (int)\n\
    @param nk number of vertical levels (int)\n\
    @return newJIMarray (numpy.ndarray)\n\
    @exception jimc.error";

static PyObject *jimc_new_array(PyObject *self, PyObject *args) {
    PyArrayObject *newarray;
    int ndiv,nk=1,istat,igrid;

    if (!PyArg_ParseTuple(args, "iii",&ndiv,&nk,&igrid)) {
        return NULL;
    }
    istat = new_jim_array(&newarray,ndiv,nk,igrid);
    if (istat<0) {
        PyErr_SetString(JimcError,"Problem allocating memory");
        return NULL;
    } else {
        return Py_BuildValue("O",newarray);
    }
}


static int new_jim_array(PyArrayObject **p_newarray, int ndiv, int nk, int onlyOneGrid){
    PyArrayObject *newarray;
    int dims[4]={1,1,1,1}, ndims=4;
    int nijh,nij,halo;

    if(ndiv >= 0 && nk > 0) {
        get_jim_dims(ndiv,&nijh,&nij,&halo);
        dims[0] = (nijh>1) ? nijh : 1;
        dims[1] = dims[0];
        dims[2] = nk;
        dims[3] = (onlyOneGrid==0) ? 10 : 1;
        ndims = 4;
        newarray = PyArray_EMPTY(ndims, (npy_intp*)dims, NPY_FLOAT, withFortranOrder);
        if (newarray != NULL) {
            p_newarray[0] = newarray;
            return 0;
        } else {
            fprintf(stderr,"ERROR: new_jim_array(ndiv,nk) - Problem allocating memory\n");
        }
    } else {
        fprintf(stderr,"ERROR: jimc_new_array(ndiv,nk) - must provide ndiv>=0 and nk>0\n");
    }
    return -1;
}


static char jimc_grid_la_lo__doc__[] =
    "Get (lat,lon) of Icosahedral grid points\n\
    (lat,lon) = jimc_grid_la_lo(ndiv)\n\
    @param ndiv number of grid divisons (int)\n\
    @return python tuple with 2 numpy.ndarray for (lat,lon)\n\
    @exception jimc.error";

static PyObject *jimc_grid_la_lo(PyObject *self, PyObject *args) {
    PyArrayObject *lat,*lon;
    int ndiv,nijh,nij,halo,nk=1,istat=-1;
    F77_INTEGER f_ndiv,f_nij,f_halo;

    if (!PyArg_ParseTuple(args, "i",&ndiv)) {
        return NULL;
    }
    if(ndiv < 0) {
        PyErr_SetString(JimcError,"JIMc ndiv must be >= 0");
        return NULL;
    }
    istat  = new_jim_array(&lat,ndiv,nk,0);
    istat += new_jim_array(&lon,ndiv,nk,0);
    if (istat<0) {
        PyErr_SetString(JimcError,"Problem allocating memory");
        return NULL;
    }
    get_jim_dims(ndiv,&nijh,&nij,&halo);
    f_ndiv = (F77_INTEGER)ndiv;
    f_halo = (F77_INTEGER)halo;
    f_nij  = (F77_INTEGER)nij;
    istat = f77name(jim_grid_lalo)((F77_REAL*)lat->data,(F77_REAL*)lon->data,
                    &f_nij,&f_halo,&f_ndiv);
    if (istat < 0) {
        Py_DECREF(lat);
        Py_DECREF(lon);
        PyErr_SetString(JimcError,"Problem computing grid lat/lon");
        return NULL;
    }
    return Py_BuildValue("(O,O)",lat,lon);
 }


static char jimc_grid_corners_la_lo__doc__[] =
    "Get (lat,lon) of Icosahedral grid points centers and corners\n\
    (lat,lon,clat,clon) = jimc_grid_corners_la_lo(ndiv,igrid)\n\
    @param ndiv number of grid divisons (int)\n\
    @param igrid Icosahedral Grid tile number [all grids tiles if iGrid=0] (int)\n\
    @return python tuple with 4 numpy.ndarray for (lat,lon,clat,clon)\n\
    @exception jimc.error";

static PyObject *jimc_grid_corners_la_lo(PyObject *self, PyObject *args) {
    PyArrayObject *lat,*lon,*clat,*clon;
    int ndiv,nijh,nij,halo,nk=6,istat=-1,igrid;
    F77_INTEGER f_ndiv,f_nij,f_halo,f_igrid;

    if (!PyArg_ParseTuple(args, "ii",&ndiv,&igrid)) {
        return NULL;
    }
    if (ndiv < 0 || igrid < 0) {
        PyErr_SetString(JimcError,"JIMc ndiv must be >= 0 and igrid in range [0,10]");
        return NULL;
    }

    istat  = new_jim_array(&lat,ndiv,1,igrid);
    istat += new_jim_array(&lon,ndiv,1,igrid);
    istat += new_jim_array(&clat,ndiv,nk,igrid);
    istat += new_jim_array(&clon,ndiv,nk,igrid);
    if (istat<0) {
        PyErr_SetString(JimcError,"Problem allocating memory");
        return NULL;
    }
    get_jim_dims(ndiv,&nijh,&nij,&halo);
    f_ndiv = (F77_INTEGER)ndiv;
    f_halo = (F77_INTEGER)halo;
    f_nij  = (F77_INTEGER)nij;
    f_igrid = (F77_INTEGER)igrid;
    istat = f77name(jim_grid_corners_lalo)(
                    (F77_REAL*)lat->data,(F77_REAL*)lon->data,
                    (F77_REAL*)clat->data,(F77_REAL*)clon->data,
                    &f_nij,&f_halo,&f_ndiv,&f_igrid);
    if (istat < 0) {
        Py_DECREF(lat);
        Py_DECREF(lon);
        Py_DECREF(clat);
        Py_DECREF(clon);
        PyErr_SetString(JimcError,"Problem computing grid corners lat/lon");
        return NULL;
    }
    return Py_BuildValue("(O,O,O,O)",lat,lon,clat,clon);
}


static char jimc_xch_halo__doc__[] =
    "Exchange Halo of data organized as a stack of 10 icosahedral grids with halo\n\
    jimc_xch_halo(field)\n\
    @param field (numpy.ndarray)\n\
    @exception jimc.error";

static PyObject *jimc_xch_halo(PyObject *self, PyObject *args) {
    PyArrayObject *field;
    int nij,nk,ngrids,istat;
    F77_INTEGER f_nij,f_nk;

    if (!PyArg_ParseTuple(args, "O",&field)) {
        return NULL;
    }
    //TODO: accept double and int types
    if (!((PyArray_ISCONTIGUOUS(field) || (field->flags & NPY_FARRAY))
        && field->descr->type_num == NPY_FLOAT)) {
        PyErr_SetString(JimcError,"JIMc - Not suported data Type");
        return NULL;
    }

    nij = field->dimensions[0];
    nk  = 1;
    ngrids = field->dimensions[2];
    if (field->nd > 3) {
        nk  = field->dimensions[2];
        ngrids = field->dimensions[3];
    }
    if (field->dimensions[1]!=nij || ngrids!=10) {
        PyErr_SetString(JimcError,"JIMc - Wrong Array dims");
        return NULL;
    }
    f_nij = (F77_INTEGER)nij;
    f_nk  = (F77_INTEGER)nk;
    if (nk == 1) {
        istat = f77name(jim_xch_halo_nompi_2d_r4)(&f_nij,(F77_REAL*)field->data);
    } else {
        istat = f77name(jim_xch_halo_nompi_3d_r4)(&f_nij,&f_nk,(F77_REAL*)field->data);
    }
    if (istat < 0) {
        PyErr_SetString(JimcError,"Problem doing halo xch on JIM grid");
        return NULL;
    }
    return Py_None;
}


static struct PyMethodDef jimc_methods[] = {
    {"version",	(PyCFunction)jimc_version,	METH_VARARGS, jimc_version__doc__},
    {"grid_dims",	(PyCFunction)jimc_dims,	METH_VARARGS, jimc_dims__doc__},
    {"new_array",	(PyCFunction)jimc_new_array,	METH_VARARGS, jimc_new_array__doc__},
    {"grid_la_lo",	(PyCFunction)jimc_grid_la_lo,	METH_VARARGS, jimc_grid_la_lo__doc__},
    {"grid_corners_la_lo",	(PyCFunction)jimc_grid_corners_la_lo,	METH_VARARGS, jimc_grid_corners_la_lo__doc__},
    {"xch_halo",	(PyCFunction)jimc_xch_halo,	METH_VARARGS, jimc_xch_halo__doc__},

    {NULL,	 (PyCFunction)NULL, 0, NULL}		/* sentinel */
};


static char jimc_module_documentation[] =
    "JIMc\n\
    Module jim_c contains the functions used to compute JIM (Icosahedral) grid points position\n\
    @author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>";

void initjimc() {
    PyObject *m;

    m = Py_InitModule4("jimc", jimc_methods,
            jimc_module_documentation,
            (PyObject*)NULL,PYTHON_API_VERSION);
    if (m == NULL)
        return;

    import_array();

    JimcError = PyErr_NewException("jimc.error", NULL, NULL);
    Py_INCREF(JimcError);
    PyModule_AddObject(m, "error", JimcError);

    if (PyErr_Occurred()) Py_FatalError("can't initialize module jimc");
}


/* -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*- */
// vim: set expandtab ts=4 sw=4:
// kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
