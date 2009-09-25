/*
Module jim_c contains the classes used to compute JIM (Icosahedral) grid points position
@author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
@date: 2009-09
*/
/* #define DEBUG On */
#include <stdio.h>
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

void get_jim_dims(int ndiv, int *nijh, int *nij, int *halo) {
    wordint f_ndiv,f_nij,f_nijh,f_halo;

    f_ndiv = (wordint)ndiv;
    f_halo = 2;
    f_nijh = f77name(jim_grid_dims)(&f_ndiv,&f_halo);
    f_halo = 0;
    f_nij  = f77name(jim_grid_dims)(&f_ndiv,&f_halo);

    nijh[0] = (int)f_nijh;
    nij[0]  = (int)f_nij;
    halo[0] = (nijh[0] - nij[0])/2;
    //printf("get_jim_dims(ndiv=%d) = (%d,%d,%d)\n",ndiv,nijh[0],nij[0],halo[0]);
    return;
}


int new_jim_array(PyArrayObject **p_newarray, int ndiv, int nk){
	PyArrayObject *newarray;
	int dims[4]={1,1,1,1}, ndims=4;
	int type_num=NPY_FLOAT;
        int nijh,nij,halo;

	if(ndiv >= 0 && nk > 0) {
            get_jim_dims(ndiv,&nijh,&nij,&halo);

            dims[0] = (nijh>1) ? nijh : 1;
            dims[1] = dims[0];
            dims[2] = nk;
            dims[3] = 10;
            ndims = 4;

            //printf("new_jim_array(ndiv=%d,nk=%d): dims[%d] = (%d,%d,%d,%d)\n",ndiv,nk,ndims,dims[0],dims[1],dims[2],dims[3]);

            newarray = PyArray_NewFromDescr(&PyArray_Type,
                                        PyArray_DescrFromType(type_num),
                                        ndims,dims,
                                        NULL, NULL, FTN_Style_Array,
                                        NULL);
            //printf("new_jim_array(ndiv=%d,nk=%d): dims[%d] = (%d,%d,%d,%d)\n",ndiv,nk,newarray->nd,newarray->dimensions[0],newarray->dimensions[1],newarray->dimensions[2],newarray->dimensions[3]);
            if (newarray->nd == 4) {
                Py_INCREF(newarray);
                p_newarray[0] = newarray;
                return 0;
            } else {
                Py_DECREF(newarray);
                fprintf(stderr,"ERROR: new_jim_array(ndiv,nk) - Problem allocating memory\n");
            }
        } else {
            fprintf(stderr,"ERROR: jimc_new_array(ndiv,nk) - must provide ndiv>=0 and nk>0\n");
        }
        return -1;
}


static char jimc_dims__doc__[] =
"Create a new numpy.ndarray with right dims for JIM grid of ndiv and nk\n(nijh, nij, halo) = jimc_dims(ndiv,halo)\n@param ndiv number of grid divisons (int)\n@param halo add halo points to nij in each dirs (-1 for default jim halo) (int)\n@return (nijh,nij,halo) where nijh = nij + 2*halo (int,int,int)";

static PyObject *
jimc_dims(PyObject *self, PyObject *args) {
    int ndiv,nhalo;
    int nijh,nij,halo;

    if (!PyArg_ParseTuple(args, "ii",&ndiv,&nhalo)) {
        fprintf(stderr,"ERROR: jimc_dims(ndiv,nhalo) - wrong arg type\n");
        Py_INCREF(Py_None);
        return Py_None;
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


static char jimc_new_array__doc__[] =
"Create a new numpy.ndarray with right dims for JIM grid of ndiv and nk\nnewJIMarray = jimc_new_array(ndiv,nk)\n@param ndiv number of grid divisons (int)\n@param nk number of vertical levels (int)\n@return newJIMarray (numpy.ndarray)";

static PyObject *
jimc_new_array(PyObject *self, PyObject *args) {
	PyArrayObject *newarray;
        int ndiv,nk=1,istat;

	if (!PyArg_ParseTuple(args, "ii",&ndiv,&nk)) {
            fprintf(stderr,"ERROR: jimc_new_array(ndiv,nk) - wrong arg type\n");
            Py_INCREF(Py_None);
            return Py_None;
        }
        istat = new_jim_array(&newarray,ndiv,nk);
        //printf("jimc_new_array(ndiv=%d,nk=%d): dims[%d] = (%d,%d,%d,%d)\n",ndiv,nk,newarray->nd,newarray->dimensions[0],newarray->dimensions[1],newarray->dimensions[2],newarray->dimensions[3]);
        if (istat<0) {
            fprintf(stderr,"ERROR: jimc_new_array(ndiv,nk) - problem allocating mem\n");
        } else {
            return Py_BuildValue("O",newarray);
        }
        Py_INCREF(Py_None);
        return Py_None;
}


static char jimc_grid_la_lo__doc__[] =
"Get (lat,lon) of Icosahedral grid points\n(lat,lon) = jimc_grid_la_lo(ndiv)\n@param ndiv number of grid divisons (int)\n@return python tuple with 2 numpy.ndarray for (lat,lon)";

static PyObject *
jimc_grid_la_lo(PyObject *self, PyObject *args) {
	PyArrayObject *lat,*lon;
        int ndiv,nijh,nij,halo,nk=1,istat=-1;
        wordint f_ndiv,f_nij,f_halo;

	if (!PyArg_ParseTuple(args, "i",&ndiv)) {
            fprintf(stderr,"ERROR: jimc_grid_la_lo(ndiv) - wrong arg type\n");
            Py_INCREF(Py_None);
            return Py_None;
        }
	if(ndiv >= 0) {
            istat  = new_jim_array(&lat,ndiv,nk);
            istat += new_jim_array(&lon,ndiv,nk);
            if (istat<0) {
                fprintf(stderr,"ERROR: jimc_grid_la_lo(ndiv) - Problem allocating memory\n");
                Py_INCREF(Py_None);
                return Py_None;
            }

            get_jim_dims(ndiv,&nijh,&nij,&halo);

            f_ndiv = (wordint)ndiv;
            f_halo = (wordint)halo;
            f_nij  = (wordint)nij;
            istat = f77name(jim_grid_lalo)(lat->data,lon->data,
                            &f_nij,&f_halo,&f_ndiv);
            if (istat < 0) {
                fprintf(stderr,"ERROR: jimc_grid_la_lo(ndiv) - problem computing grid lat/lon\n");
                Py_DECREF(lat);
                Py_DECREF(lon);
            } else {
                return Py_BuildValue("(O,O)",lat,lon);
            }
        } else {
            fprintf(stderr,"ERROR: jimc_grid_la_lo(ndiv) - ndiv must be >= 0\n");
        }
        Py_INCREF(Py_None);
        return Py_None;
}


static char jimc_grid_corners_la_lo__doc__[] =
"Get (lat,lon) of Icosahedral grid points corners\n(lat,lon) = jimc_grid_corners_la_lo(ndiv)\n@param ndiv number of grid divisons (int)\n@return python tuple with 2 numpy.ndarray for (lat,lon)";

static PyObject *
jimc_grid_corners_la_lo(PyObject *self, PyObject *args) {
	PyArrayObject *lat,*lon;
        int ndiv,nijh,nij,halo,nk=6,istat=-1;
        wordint f_ndiv,f_nij,f_halo;

	if (!PyArg_ParseTuple(args, "i",&ndiv)) {
            fprintf(stderr,"ERROR: jimc_grid_corner_la_lo(ndiv) - wrong arg type\n");
            Py_INCREF(Py_None);
            return Py_None;
        }
	if(ndiv >= 0) {
            istat  = new_jim_array(&lat,ndiv,nk);
            istat += new_jim_array(&lon,ndiv,nk);
            if (istat<0) {
                fprintf(stderr,"ERROR: jimc_grid_corners_la_lo(ndiv) - Problem allocating memory\n");
                Py_INCREF(Py_None);
                return Py_None;
            }

            get_jim_dims(ndiv,&nijh,&nij,&halo);

            f_ndiv = (wordint)ndiv;
            f_halo = (wordint)halo;
            f_nij  = (wordint)nij;
            istat = f77name(jim_grid_corners_lalo)(lat->data,lon->data,
                            &f_nij,&f_halo,&f_ndiv);
            if (istat < 0) {
                fprintf(stderr,"ERROR: jimc_grid_corner_la_lo(ndiv) - problem computing grid corners lat/lon\n");
                Py_DECREF(lat);
                Py_DECREF(lon);
            } else {
                return Py_BuildValue("(O,O)",lat,lon);
            }
        } else {
            fprintf(stderr,"ERROR: jimc_grid_corner_la_lo(ndiv) - ndiv must be >= 0\n");
        }
        Py_INCREF(Py_None);
        return Py_None;
}


static char jimc_xch_halo__doc__[] =
"Exchange Halo of data organized as a stack of 10 icosahedral grids with halo\nistat = jimc_xch_halo(field)\n@param field (numpy.ndarray)\n@return istat status of the operation";

static PyObject *
jimc_xch_halo(PyObject *self, PyObject *args) {
	PyArrayObject *field;

        int nij,nk,ngrids,istat;
        wordint f_nij,f_nk;

	if (!PyArg_ParseTuple(args, "O",&field)) {
            fprintf(stderr,"ERROR: jimc_xch_halo(field) - Problem parsing input arg\n");
            Py_INCREF(Py_None);
            return Py_None;
        }

        //TODO: accept double and int types
        if ((PyArray_ISCONTIGUOUS(field) || (field->flags & NPY_FARRAY))
            && field->descr->type_num == NPY_FLOAT) {

            nij = field->dimensions[0];
            nk  = 1;
            ngrids = field->dimensions[2];
            if (field->nd > 3) {
                nk  = field->dimensions[2];
                ngrids = field->dimensions[3];
            }
            if (field->dimensions[1]!=nij || ngrids!=10) {
                fprintf(stderr,"ERROR: jimc_xch_halo(field) - Wrong Array dims: nd=%d, nijk=%d, %d,%d, ngrids=%d\n",field->nd,nij,field->dimensions[1],nk,ngrids);
                Py_INCREF(Py_None);
                return Py_None;
            }

            f_nij = (wordint) nij;
            f_nk  = (wordint) nk;
            if (nk == 1) {
                //printf("jimc_xch_halo_2d - Array dims: nd=%d, nijk=%d, %d,%d, ngrids=%d, strides=%d\n",field->nd,nij,field->dimensions[1],nk,ngrids,field->strides[0]);
                istat = f77name(jim_xch_halo_nompi_2d_r4)(&f_nij,field->data);
            } else {
                //printf("jimc_xch_halo_3d - Array dims: nd=%d, nijk=%d, %d,%d, ngrids=%d\n",field->nd,nij,field->dimensions[1],nk,ngrids);
                istat = f77name(jim_xch_halo_nompi_3d_r4)(&f_nij,&f_nk,field->data);
            }

            return Py_BuildValue("i",istat);
        } else {
            fprintf(stderr,"ERROR: jimc_xch_halo(field) - Wrong Array type\n");
        }

        Py_INCREF(Py_None);
        return Py_None;
}

/* List of methods defined in the module */

static struct PyMethodDef jimc_methods[] = {
    {"jimc_dims",	(PyCFunction)jimc_dims,	METH_VARARGS, jimc_dims__doc__},
    {"jimc_new_array",	(PyCFunction)jimc_new_array,	METH_VARARGS, jimc_new_array__doc__},
    {"jimc_grid_la_lo",	(PyCFunction)jimc_grid_la_lo,	METH_VARARGS, jimc_grid_la_lo__doc__},
    {"jimc_grid_corners_la_lo",	(PyCFunction)jimc_grid_corners_la_lo,	METH_VARARGS, jimc_grid_corners_la_lo__doc__},
    {"jimc_xch_halo",	(PyCFunction)jimc_xch_halo,	METH_VARARGS, jimc_xch_halo__doc__},

    {NULL,	 (PyCFunction)NULL, 0, NULL}		/* sentinel */
};


/* Initialization function for the module (*must* be called initjimc) */

static char jimc_module_documentation[] =
"Module jim_c contains the classes used to compute JIM (Icosahedral) grid points position\n@author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>";

void initjimc() {
	PyObject *m, *d;

	/* Create the module and add the functions */
	m = Py_InitModule4("jimc", jimc_methods,
		jimc_module_documentation,
		(PyObject*)NULL,PYTHON_API_VERSION);

	/* Import the array object */
	import_array();

	/* Add some symbolic constants to the module */
	d = PyModule_GetDict(m);
	ErrorObject = PyString_FromString("jimc.error");
	PyDict_SetItemString(d, "error", ErrorObject);

	/* XXXX Add constants here */

	/* Check for errors */
	if (PyErr_Occurred())
		Py_FatalError("can't initialize module jimc");
	printf("JIM module V-%s (%s) initialized\n",version,lastmodified);
	init_lentab();
}

