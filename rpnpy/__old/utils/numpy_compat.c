 
#include <Python.h>
#include "numpy_compat.h"

/* Provide some fallback definitions for older numpy API versions;
   as of this writing (Dec 2013) the standard version of numpy
   installed is 1.3, which is quite out of date.  Much of the
   development since, however, has relied on features present in
   numpy version 1.7 (which also deprecates the older API).  For
   compatibility, provide wrappers for missing pieces of the
   API as needed; this is mostly in array flags processing and
   array creation */
#if NPY_API_VERSION < 7
   #if NPY_API_VERSION < 6
      /* NewLikeArray doesn't exist prior to 1.6, so we have to
         fake it.  It is only called in this code with (KEEPORDER,
         NULL, 1) as parameters (to keep the parent arrays' strides,
         to not override the descriptor, and to permit subtypes),
         which aren't terribly special; we can supply such a
         reduced function here.  The logic applied is based on the
         Numpy 1.7 source's implementation of the same function */
      PyObject * PyArray_NewLikeArray(PyArrayObject* prototype,
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
