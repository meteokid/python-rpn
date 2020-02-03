#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 \
#                /ssm/net/rpn/libs/15.2 \
#                /ssm/net/cmdn/tests/vgrid/6.0.0-a3/intel13sp1u2

"""
 Module vgd is a ctypes import of vgrid's library (libvgrid.so)
 
 The libvgrid.so library is provided with the VGrid Descriptor package
 developed at CMC/RPN by R.McTaggartCowan and A.Plante
 
 The vgd python module includes
 - python wrapper to main libvgrid's C functions
 - helper functions
 - prototypes for many libvgrid's C functions
 - pre-defined constants
 - along with comprenhensive inline documentation

 See also:
    rpnpy.vgd.proto
    rpnpy.vgd.const
    rpnpy.vgd.base

"""

from rpnpy.version import *

__SUBMODULES__ = ['proto', 'const', 'base']
__all__ = ['loadVGDlib', 'libvgd', 'VGD_VERSION', 'VGD_LIBPATH',
           'VGDError'] + __SUBMODULES__

## VGD_VERSION_DEFAULT = '_rpnpy'
VGD_VERSION_DEFAULT = '*'

class VGDError(Exception):
    """
    General VGD module error/exception
    """
    pass

def checkVGDlibPath(libfile):
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

def loadVGDlib(vgd_version=None):
    """
    Import libvgrid.so using ctypes

    Args:
       vgd_version (str): libvgrid version number to load
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
    vgd_libfile = 'libvgridshared' + VGD_VERSION.strip() + '.so'

    pylibpath   = os.getenv('PYTHONPATH','').split(':')
    ldlibpath   = os.getenv('LD_LIBRARY_PATH','').split(':')
    eclibpath   = os.getenv('EC_LD_LIBRARY_PATH','').split()
    VGD_LIBPATH = checkVGDlibPath(vgd_libfile)
    if not VGD_LIBPATH:
        for path in pylibpath + ldlibpath + eclibpath:
            VGD_LIBPATH = checkVGDlibPath(os.path.join(path.strip(), vgd_libfile))
            if VGD_LIBPATH:
                break

    if not VGD_LIBPATH:
        raise IOError(-1, 'Failed to find libvgrid.so: ', vgd_libfile)

    VGD_LIBPATH = os.path.abspath(VGD_LIBPATH)
    libvgd = None
    try:
        libvgd = ct.cdll.LoadLibrary(VGD_LIBPATH)
        #libvgd = np.ctypeslib.load_library(vgd_libfile, VGD_LIBPATH)
    except IOError:
        raise IOError('ERROR: cannot load libvgrid shared version: ' +
                      VGD_VERSION)
    return (VGD_VERSION, VGD_LIBPATH, libvgd)

(VGD_VERSION, VGD_LIBPATH, libvgd) = loadVGDlib()

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
