#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 \
#                /ssm/net/rpn/libs/15.2 \
#                /ssm/net/cmdn/tests/vgrid/6.0.0-a3/intel13sp1u2

"""
 Module burpc is a ctypes import of burp_c's library (libburp_c_shared.so)
 
 The burp_c_shared library is provided with the CMDA Libraries
 developed at CMC/CMDA
 
 The burpc python module includes
 - python wrapper to main burp_c's C functions
 - helper functions
 - prototypes for burp_c's C functions
 - pre-defined constants
 - along with comprenhensive inline documentation

 See also:
    rpnpy.burpc.proto
    rpnpy.burpc.const
    rpnpy.burpc.base

"""

from rpnpy.version import *

__SUBMODULES__ = ['proto', 'const', 'base']
__all__ = ['loadBURPClib', 'libburpc', 'BURPC_VERSION', 'BURPC_LIBPATH',
           'BurpcError'] + __SUBMODULES__

## BURPC_VERSION_DEFAULT = '_rpnpy'
BURPC_VERSION_DEFAULT = '*'

class BurpcError(Exception):
    """
    General BURPC module error/exception
    """
    pass

def checkBURPClibPath(libfile):
    """
    Return first matched filename for libfile wildcard
    Return None if no match
    """
    import os
    import glob
    LIBPATH_ALL = glob.glob(libfile)
    if len(LIBPATH_ALL) > 0:
        if os.path.isfile(LIBPATH_ALL[0]):
            return LIBPATH_ALL[0]
    return None

def loadBURPClib(burpc_version=None):
    """
    Import libburp_c_shared.so using ctypes

    Args:
       burpc_version (str): libburp_c_shared version number to load
                            Default: RPNPY_BURPC_VERSION Env.Var.
                                     BURPC_VERSION_DEFAULT if not RPNPY_BURPC_VERSION
    Returns:
       (BURPC_VERSION, BURPC_LIBPATH, libburpc)
       where:
       BURPC_VERSION (str)  : loaded libburp_c version
       BURPC_LIBPATH (str)  : path to loaded libburpc shared lib
       libburpc      (CDLL) : ctypes library object for libburpc.so
       
    Library 'libburp_c_sharedVERSION.so' is searched into the Env.Var. paths:
       PYTHONPATH, EC_LD_LIBRARY_PATH, LD_LIBRARY_PATH
    """
    import os
    import ctypes as ct
    ## import numpy  as np
    ## import numpy.ctypeslib as npct

    if burpc_version is None:
        BURPC_VERSION = os.getenv('RPNPY_BURPC_VERSION',
                                  BURPC_VERSION_DEFAULT).strip()
    else:
        BURPC_VERSION = burpc_version
    burpc_libfile = 'libburp_c_shared' + BURPC_VERSION.strip() + '.so'

    pylibpath   = os.getenv('PYTHONPATH','').split(':')
    ldlibpath   = os.getenv('LD_LIBRARY_PATH','').split(':')
    eclibpath   = os.getenv('EC_LD_LIBRARY_PATH','').split()
    BURPC_LIBPATH = checkBURPClibPath(burpc_libfile)
    if not BURPC_LIBPATH:
        for path in pylibpath + ldlibpath + eclibpath:
            BURPC_LIBPATH = checkBURPClibPath(os.path.join(path.strip(), burpc_libfile))
            if BURPC_LIBPATH:
                break

    if not BURPC_LIBPATH:
        raise IOError(-1, 'Failed to find libburp_c_shared.so: ', burpc_libfile)

    BURPC_LIBPATH = os.path.abspath(BURPC_LIBPATH)
    libburpc = None
    try:
        libburpc = ct.cdll.LoadLibrary(BURPC_LIBPATH)
        #libburpc = np.ctypeslib.load_library(burpc_libfile, BURPC_LIBPATH)
    except IOError:
        raise IOError('ERROR: cannot load libburp_c shared version: ' +
                      BURPC_VERSION)
    return (BURPC_VERSION, BURPC_LIBPATH, libburpc)

(BURPC_VERSION, BURPC_LIBPATH, libburpc) = loadBURPClib()
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
