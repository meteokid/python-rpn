/*
Module ModulecName
@author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
@date: 2009-09
*/
#include <Python.h>
#include <numpy/arrayobject.h>

#include <rpnmacros.h>

#include "../utils/py_capi_ftn_utils.h"
#include "rpn_version.h"

static PyObject *ModulecError;

// List module fn prototypes
static PyObject *modulec_version(PyObject *self, PyObject *args);


static char modulec_version__doc__[] = "(version,lastUpdate) = version()";

static PyObject *modulec_version(PyObject *self, PyObject *args) {
    return Py_BuildValue("(ss)",VERSION,LASTUPDATE);
}


static char modulec_someFunction__doc__[] =
    "Description.\n\
    fnResult = someFunction(arg1,arg2)\n\
    @param arg1 Description arg1 (int)\n\
    @param arg2 Description arg2 (int)\n\
    @return fnResult (int)\
    @exception TypeError\n\
    @exception modulec.error";

static PyObject *modulec_someFunction(PyObject *self, PyObject *args) {
    int arg1,agr2,fnResult;

    if (!PyArg_ParseTuple(args, "ii",
         &arg1,&agr2)) {
        return NULL;
    }
    fnResult = someFunction(arg1,arg2);
    if(fnResult == PYCAPIFTN_ERR) {
        PyErr_SetString(ModulecError,"Error Message");
        return NULL;
    }
    //return Py_None;
    return Py_BuildValue("i",fnResult);
}


/* List of methods defined in the module */

static struct PyMethodDef modulec_methods[] = {
    {"version", (PyCFunction)modulec_version, METH_VARARGS, modulec_version__doc__},
    {"someFunction", (PyCFunction)modulec_someFunction, METH_VARARGS, modulec_someFunction__doc__},

    {NULL,	 (PyCFunction)NULL, 0, NULL}		/* sentinel */
};


/* Initialization function for the module
    (*must* be called initName and must be the sole non static declaration in the file) */

static char modulec_module_documentation[] =
    "ModulecName\n\
    Module ModulecName Description\n\
    @author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>";

void initmodulec() {
    PyObject *m, *d;

    m = Py_InitModule4("modulec", modulec_methods,
            modulec_module_documentation,
            (PyObject*)NULL,PYTHON_API_VERSION);
    if (m == NULL)
        return;

    import_array();

    ModulecError = PyErr_NewException("modulec.error", NULL, NULL);
    Py_INCREF(ModulecError);
    PyModule_AddObject(m, "error", ModulecError);

    d = PyModule_GetDict(m);
    PyDict_SetItemString(d, "EXAMPLE_CONSTANT", PyInt_FromLong((long)1));

    if (PyErr_Occurred()) Py_FatalError("Cannot initialize module ModulecName");
}

/* -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*- */
// vim: set expandtab ts=4 sw=4:
// kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
