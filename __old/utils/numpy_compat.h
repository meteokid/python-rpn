 
#include <numpy/arrayobject.h>

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
      PyObject * PyArray_NewLikeArray(PyArrayObject* prototype,
                                      NPY_ORDER order, PyArray_Descr* in_descr,
                                      int subok);
   #endif
#endif
