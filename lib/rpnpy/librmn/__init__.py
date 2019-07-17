#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1
"""
 Module librmn is a ctypes import of librmnshared.so
 
 The librmn python module includes
 - python wrapper to main librmn's C functions
 - helper functions
 - prototypes for many librmn's C functions
 - pre-defined constants
 - along with comprenhensive inline documentation

 See also:
     rpnpy.librmn.proto
     rpnpy.librmn.const
     rpnpy.librmn.base
     rpnpy.librmn.fstd98
     rpnpy.librmn.interp
     rpnpy.librmn.grids
"""

from rpnpy.version import *

__SUBMODULES__ = ['proto', 'const', 'base', 'fstd98', 'interp', 'grids',
                  'proto_burp', 'burp_const', 'burp']
__all__ = ['loadRMNlib', 'librmn', 'RMN_VERSION', 'RMN_LIBPATH',
           'RMNError'] + __SUBMODULES__

## RMN_VERSION_DEFAULT = '_rpnpy'
RMN_VERSION_DEFAULT = '*'

class RMNError(Exception):
    """
    General RMN module error/exception
    """
    pass

def checkRMNlibPath(rmn_libfile):
    """
    Return first matched filename for rmn_libfile wildcard
    Return None if no match
    """
    import os
    import glob
    RMN_LIBPATH_ALL = glob.glob(rmn_libfile)
    if len(RMN_LIBPATH_ALL) > 0:
        if os.path.isfile(RMN_LIBPATH_ALL[0]):
            return RMN_LIBPATH_ALL[0]
    return None

def loadRMNlib(rmn_version=None):
    """
    Import librmnshared using ctypes

    Args:
       rmn_version (str): librmnshared version number to load
                          Default: RPNPY_RMN_VERSION Env.Var.
                                   RMN_VERSION_DEFAULT if not RPNPY_RMN_VERSION
    Returns:
       (RMN_VERSION, RMN_LIBPATH, librmn)
       where:
       RMN_VERSION (str)  : loaded librmn version
       RMN_LIBPATH (str)  : path to loaded librmn shared lib
       librmn      (CDLL) : ctypes library object for librmn.so

    Library 'librmnsharedVERSION.so' is searched into the Env.Var. paths:
       PYTHONPATH, EC_LD_LIBRARY_PATH, LD_LIBRARY_PATH
    """
    import os
    import ctypes as ct
    ## import numpy  as np
    ## import numpy.ctypeslib as npct

    # For windows, need to change the current directory to see the .dll files.
    curdir = os.path.realpath(os.getcwd())
    os.chdir(os.path.join(os.path.dirname(__file__),os.pardir,'_sharedlibs'))

    if rmn_version is None:
        RMN_VERSION = os.getenv('RPNPY_RMN_VERSION',
                                RMN_VERSION_DEFAULT).strip()
    else:
        RMN_VERSION = rmn_version
    rmn_libfile = 'librmnshared' + RMN_VERSION.strip() + '.*'

    localpath   = [os.path.realpath(os.getcwd())]
    pylibpath   = os.getenv('PYTHONPATH','').split(':')
    ldlibpath   = os.getenv('LD_LIBRARY_PATH','').split(':')
    eclibpath   = os.getenv('EC_LD_LIBRARY_PATH','').split()
    RMN_LIBPATH = checkRMNlibPath(rmn_libfile)
    if not RMN_LIBPATH:
        for path in localpath + pylibpath + ldlibpath + eclibpath:
            RMN_LIBPATH = checkRMNlibPath(os.path.join(path.strip(), rmn_libfile))
            if RMN_LIBPATH:
                break

    if not RMN_LIBPATH:
        raise IOError(-1, 'Failed to find librmn.so: ', rmn_libfile)

    RMN_LIBPATH = os.path.abspath(RMN_LIBPATH)
    librmn = None
    try:
        librmn = ct.cdll.LoadLibrary(RMN_LIBPATH)
        #librmn = np.ctypeslib.load_library(rmn_libfile, RMN_LIBPATH)
    except IOError as e:
        raise IOError('ERROR: cannot load librmn shared version: ' +
                      RMN_VERSION, e)
    os.chdir(curdir)
    return (RMN_VERSION, RMN_LIBPATH, librmn)

(RMN_VERSION, RMN_LIBPATH, librmn) = loadRMNlib()

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
