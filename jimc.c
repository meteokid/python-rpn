/*
Module jim_c contains the classes used to compute JIM (Icosahedral) grid points position
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

static char jimc_grid__doc__[] =
"Get (lat,lon) of Icosahedral grid points\n(lat,lon) = jimc_grid(ndiv,igrid)\n@param ndiv number of grid divisons (int) \n@param igrid grid number [0-9](int)\n@return python tuple with 2 numpy.ndarray for (lat,lon)";

static PyObject *
jimc_grid(PyObject *self, PyObject *args) {
	PyArrayObject *lat,*lon;
	int dims[3]={1,1,1}, strides[3]={0,0,0}, ndims=3;
	int type_num=NPY_FLOAT;

// 	int iun, ni=0, nj=0, nk=0, ip1=0, ip2=0, ip3=0, temp;
// 	char TYPVAR[3]={' ',' ','\0'};
// 	char NOMVAR[5]={' ',' ',' ',' ','\0'};
// 	char ETIKET[13]={' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','\0'};
// 	char GRTYP[2]={' ','\0'};
// 	int handle, junk;
// 	int dateo=0, deet=0, npas=0, nbits=0, datyp=0, ig1=0, ig2=0, ig3=0, ig4=0;
// 	int swa=0, lng=0, dltf=0, ubc=0, extra1=0, extra2=0, extra3=0;

        int ndiv,igrid,nij,nk=1,istat=-1;
        wordint f_ndiv,f_igrid,f_nij;

	if (!PyArg_ParseTuple(args, "ii",&ndiv,&igrid))
		return NULL;
	if(ndiv >= 0 && igrid >= 0) {
            f_ndiv  = (wordint)ndiv;
            f_igrid = (wordint)igrid;

            printf("jimc_grid: ndiv=%d, igrid=%d\n",ndiv,igrid);

            f_nij   = f77name(jim_grid_dims)(&f_ndiv);
            nij     = (int)f_nij;

            dims[0] = (nij>1) ? nij : 1;
            dims[1] = dims[0];
            dims[2] = 1;
/*            strides[0]=4;
            strides[1]=strides[0]*dims[0];
            strides[2]=strides[1]*dims[1];*/
            if(nk>1) ndims=3;
            else if(nij>1) ndims=2;
            else ndims=1;

            printf("jimc_grid: nij=%d, ndims=%d\n",nij,ndims);

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

            printf("jimc_grid: to jim_grid_lalo\n");

            istat = f77name(jim_grid_lalo)(lat->data,lon->data,&f_nij,&f_nij,&f_igrid,&f_ndiv);

            if (istat < 0) {
                Py_DECREF(lat);
                Py_DECREF(lon);
            }

            return Py_BuildValue("(O,O)",lat,lon);
        }

        Py_INCREF(Py_None);
        return Py_None;
}

/* List of methods defined in the module */

static struct PyMethodDef jimc_methods[] = {
    {"jimc_grid",	(PyCFunction)jimc_grid,	METH_VARARGS, jimc_grid__doc__},

    {NULL,	 (PyCFunction)NULL, 0, NULL}		/* sentinel */
};


/* Initialization function for the module (*must* be called initFstdc) */

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

