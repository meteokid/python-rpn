#!/usr/bin/env python
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 \
#                /ssm/net/rpn/libs/15.2 \
#                /ssm/net/cmdn/tests/vgrid/6.0.0-a3/intel13sp1u2

"""
Module vgd is a ctypes import of vgrid's library (libdescrip.so)

The libdescrip.so library is provided with the VGrid Descriptor package
developed at CMC/RPN by R.McTaggartCowan and A.Plante
 
The vgd python module includes
- python wrapper to main libdescrip's C functions
- helper functions
- prototypes for many libdescrip's C functions
- pre-defined constants
- along with comprenhensive inline documentation

See also:
help(vgd.proto)
help(vgd.const)
help(vgd.vgd)

Libdescrip 6.0.0.a3 was built w/o -lrmn, need to rebuild it:
   HPCS_BASE = hpcs/201402/02/base/
   COMPILER=intel13sp1u2
   HPCS_COMP = hpcs/201402/02/$(COMPILER)
   #COMPILER=xlf_13.1.0.10
   #HPCS_COMP = hpcs/ext/$(COMPILER)
   RPN_LIB=rpn/libs/15.2
   . ssmuse-sh -d $(HPCS_BASE) -d $(HPCS_COMP) -d $(RPN_LIB);\
   if [ $$ORDENV_PLAT = ubuntu-12.04-amd64-64 ]; then \
      s.f90 -shared -o libdescrip.so *.o \
            -Wl,-rpath,/ssm/net/hpcs/201311/master/intelcomp_2013sp1_multi/composer_xe_2013_sp1.0.080/mkl/lib/intel64 \
            -Wl,-rpath,/ssm/net/hpcs/201311/master/intelcomp_2013sp1_multi/composer_xe_2013_sp1.0.080/compiler/lib/intel64 \
            -lrmn ;\
   elif [ $$ORDENV_PLAT = aix-7.1-ppc7-64 ]; then \
      s.f90 -openmp -o libdescrip.so -qmkshrobj *.o ;\
   fi


@author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
"""

from rpnpy.version import *

__SUBMODULES__ = ['proto', 'const']
__all__ = ['loadVGDlib', 'libvgd', 'VGD_VERSION', 'VGD_LIBPATH',
           'VGDError'] + __SUBMODULES__

VGD_VERSION_DEFAULT = 'shared_6.0.0-a4'
#VGD_VERSION_DEFAULT = '_6.0.a3-mine'

class VGDError(Exception):
    """
    General VGD module error/exception
    """
    pass

def loadVGDlib(vgd_version=None):
    """
    Import libdescrip.so using ctypes

    Args:
       vgd_version (str): libdescrip version number to load
                          Default: RPNPY_VGD_VERSION Env.Var.
                                   VGD_VERSION_DEFAULT if not RPNPY_VGD_VERSION
    Returns:
       (VGD_VERSION, VGD_LIBPATH, libvgd)
       where:
       VGD_VERSION (str)  : loaded libvgd version
       VGD_LIBPATH (str)  : path to loaded libvgd shared lib
       libvgd      (CDLL) : ctypes library object for libvgd.so
       
    Library 'libvgdVERSION.so' is searched into the Env.Var. paths:
       PYTHONPATH, EC_LD_LIBRARY_PATH, LD_LIBRARY_PATH
    """
    import os
    import ctypes as ct
    ## import numpy  as np
    ## import numpy.ctypeslib as npct

    if vgd_version is None:
        VGD_VERSION = os.getenv('RPNPY_VGD_VERSION',
                                VGD_VERSION_DEFAULT).strip()
    else:
        VGD_VERSION = vgd_version
    vgd_libfile = 'libdescrip' + VGD_VERSION.strip() + '.so'

    pylibpath   = os.getenv('PYTHONPATH').split(':')
    ldlibpath   = os.getenv('LD_LIBRARY_PATH').split(':')
    eclibpath   = os.getenv('EC_LD_LIBRARY_PATH').split()
    VGD_LIBPATH = vgd_libfile
    if not os.path.exists(VGD_LIBPATH):
        for path in pylibpath + ldlibpath + eclibpath:
            VGD_LIBPATH = os.path.join(path.strip(), vgd_libfile)
            if os.path.exists(VGD_LIBPATH):
                break
            else:
                VGD_LIBPATH = None

    if not VGD_LIBPATH:
        raise IOError, (-1, 'Failed to find libdescrip.so: ', vgd_libfile)

    VGD_LIBPATH = os.path.abspath(VGD_LIBPATH)
    libvgd = None
    try:
        libvgd = ct.cdll.LoadLibrary(VGD_LIBPATH)
        #libvgd = np.ctypeslib.load_library(vgd_libfile, VGD_LIBPATH)
    except IOError:
        raise IOError('ERROR: cannot load libdescrip shared version: ' +
                      VGD_VERSION)
    return (VGD_VERSION, VGD_LIBPATH, libvgd)

(VGD_VERSION, VGD_LIBPATH, libvgd) = loadVGDlib()
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
