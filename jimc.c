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


static char jimc_new_array__doc__[] =
"Create a new numpy.ndarray with right dims for JIM grid of ndiv and nk\nnewJIMarray = jimc_new_array(ndiv,nk)\n@param ndiv number of grid divisons (int)\n@param nk number of vertical levels (int)\n@return newJIMarray (numpy.ndarray)";

static PyObject *
jimc_new_array(PyObject *self, PyObject *args) {
	PyArrayObject *newarray;
	int dims[3]={1,1,1}, strides[3]={0,0,0}, ndims=3;
	int type_num=NPY_FLOAT;

        int ndiv,nijh,nk=1;
        wordint f_ndiv,f_nij,f_nijh,f_halo;

	if (!PyArg_ParseTuple(args, "ii",&ndiv,&nk)) {
            fprintf(stderr,"ERROR: jimc_new_array(ndiv,nk) - wrong arg type\n");
            Py_INCREF(Py_None);
            return Py_None;
        }
	if(ndiv >= 0 && nk > 0) {
            f_ndiv  = (wordint)ndiv;

            f_halo  = 2;
            f_nijh  = f77name(jim_grid_dims)(&f_ndiv,&f_halo);
            nijh    = (int)f_nijh;

            dims[0] = (nijh>1) ? nijh : 1;
            dims[1] = dims[0];
            if (nk==1) {
                dims[2] = 10;
                ndims = 3;
            } else {
                dims[2] = nk;
                dims[3] = 10;
                ndims = 4;
            }

            newarray = PyArray_NewFromDescr(&PyArray_Type,
                                        PyArray_DescrFromType(type_num),
                                        ndims,dims,
                                        NULL, NULL, FTN_Style_Array,
                                        NULL);
            Py_INCREF(newarray);
            return newarray;
        } else {
            fprintf(stderr,"ERROR: jimc_new_array(ndiv,nk) - must provide ndiv>=0 and nk>0\n");
        }

        Py_INCREF(Py_None);
        return Py_None;
}


static char jimc_grid_la_lo__doc__[] =
"Get (lat,lon) of Icosahedral grid points\n(lat,lon) = jimc_grid_la_lo(ndiv)\n@param ndiv number of grid divisons (int)\n@return python tuple with 2 numpy.ndarray for (lat,lon)";

static PyObject *
jimc_grid_la_lo(PyObject *self, PyObject *args) {
	PyArrayObject *lat,*lon;
	int dims[3]={1,1,1}, strides[3]={0,0,0}, ndims=3;
	int type_num=NPY_FLOAT;

        int ndiv,nijh,nk=1,istat=-1;
        wordint f_ndiv,f_nij,f_nijh,f_halo;

	if (!PyArg_ParseTuple(args, "i",&ndiv)) {
            fprintf(stderr,"ERROR: jimc_grid(ndiv) - wrong arg type\n");
            Py_INCREF(Py_None);
            return Py_None;
        }
	if(ndiv >= 0) {
            f_ndiv  = (wordint)ndiv;

            f_halo  = 2;
            f_nijh  = f77name(jim_grid_dims)(&f_ndiv,&f_halo);
            nijh    = (int)f_nijh;

            dims[0] = (nijh>1) ? nijh : 1;
            dims[1] = dims[0];
            dims[2] = 10;
            ndims = 3;

            lat = PyArray_NewFromDescr(&PyArray_Type,
                                        PyArray_DescrFromType(type_num),
                                        ndims,dims,
                                        NULL, NULL, FTN_Style_Array,
                                        NULL);
            lon = PyArray_NewFromDescr(&PyArray_Type,
                                        PyArray_DescrFromType(type_num),
                                        ndims,dims,
                                        NULL, NULL, FTN_Style_Array,
                                        NULL);

            f_halo = 0;
            f_nij  = f77name(jim_grid_dims)(&f_ndiv,&f_halo);
            f_halo = (f_nijh - f_nij)/2;

            istat = f77name(jim_grid_lalo)(lat->data,lon->data,
                            &f_nij,&f_halo,&f_ndiv);
            if (istat < 0) {
                fprintf(stderr,"ERROR: jimc_grid(ndiv) - problem computing grid lat/lon\n");

                Py_DECREF(lat);
                Py_DECREF(lon);
            } else {
                //istat = f77name(jim_xch_halo_nompi_2d_r4)(&f_nij,lat->data);
                //istat = f77name(jim_xch_halo_nompi_2d_r4)(&f_nij,lon->data);
                return Py_BuildValue("(O,O)",lat,lon);
            }
        } else {
            fprintf(stderr,"ERROR: jimc_grid(ndiv) - ndiv must be >= 0\n");
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
    {"jimc_new_array",	(PyCFunction)jimc_new_array,	METH_VARARGS, jimc_new_array__doc__},
    {"jimc_grid_la_lo",	(PyCFunction)jimc_grid_la_lo,	METH_VARARGS, jimc_grid_la_lo__doc__},
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

