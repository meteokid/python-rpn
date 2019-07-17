#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 \
#                /ssm/net/rpn/libs/15.2 \
#                /ssm/net/cmdn/tests/vgrid/6.0.0-a3/intel13sp1u2

"""
Module burpc is a ctypes import of [[CMDA]]'s [[Cmda_tools#Librairies.2FAPI_BURP_CMDA|burp_c]] library (libburp_c_shared.so)

{{roundboxtop}}
The functions described below are a very close ''port'' from the original
[[Cmda_tools#Librairies.2FAPI_BURP_CMDA|burp_c]] package.<br>
You may want to refer to the [[Cmda_tools#Librairies.2FAPI_BURP_CMDA|burp_c]]
documentation for more details.

This module is new in version 2.1.b2
{{roundboxbot}}

The burp_c_shared library is provided with the [[CMDA]] Libraries developed at [[CMC]]/[[CMDA]]

The burpc python module includes
* python wrapper to main burp_c's C functions
* helper functions
* prototypes for burp_c's C functions
* Object model for [[BURP]] elements
* pre-defined constants
* along with comprenhensive inline documentation

See also:
    rpnpy.burpc.brpobj
    rpnpy.burpc.base
    rpnpy.burpc.const
    rpnpy.burpc.proto

"""

from rpnpy.version import *

__SUBMODULES__ = ['proto', 'const', 'base', 'brpobj']
__all__ = ['load_burpc_lib', 'libburpc', 'BURPC_VERSION', 'BURPC_LIBPATH',
           'BurpcError'] + __SUBMODULES__

## BURPC_VERSION_DEFAULT = '_rpnpy'
BURPC_VERSION_DEFAULT = '*'

class BurpcError(Exception):
    """
    General BURPC module error/exception
    """
    pass

def check_burpc_libpath(libfile):
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

def load_burpc_lib(burpc_version=None):
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
    # Load librmn shared library here, to resolve symbols when running on MacOSX.
    from rpnpy import librmn
    ## import numpy  as np
    ## import numpy.ctypeslib as npct

    # For windows, need to change the current directory to see the .dll files.
    curdir = os.path.realpath(os.getcwd())
    os.chdir(os.path.join(os.path.dirname(__file__),os.pardir,'_sharedlibs'))

    if burpc_version is None:
        BURPC_VERSION = os.getenv('RPNPY_BURPC_VERSION',
                                  BURPC_VERSION_DEFAULT).strip()
    else:
        BURPC_VERSION = burpc_version
    burpc_libfile = 'libburp_c_shared' + BURPC_VERSION.strip() + '.*'

    localpath   = [os.path.realpath(os.getcwd())]
    pylibpath = os.getenv('PYTHONPATH', '').split(':')
    ldlibpath = os.getenv('LD_LIBRARY_PATH', '').split(':')
    eclibpath = os.getenv('EC_LD_LIBRARY_PATH', '').split()
    BURPC_LIBPATH = check_burpc_libpath(burpc_libfile)
    if not BURPC_LIBPATH:
        for path in localpath + pylibpath + ldlibpath + eclibpath:
            BURPC_LIBPATH = check_burpc_libpath(os.path.join(path.strip(),
                                                           burpc_libfile))
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
    os.chdir(curdir)
    return (BURPC_VERSION, BURPC_LIBPATH, libburpc)

(BURPC_VERSION, BURPC_LIBPATH, libburpc) = load_burpc_lib()

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
