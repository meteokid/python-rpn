#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 \
#                /ssm/net/rpn/libs/15.2 \
#                /ssm/net/cmdn/tests/vgrid/6.0.0-a3/intel13sp1u2

"""
 Module rpnpy.utils is collection of high level misc tools
  
 The rpnpy.utils python module includes
 - RPN STD files 3D fields read / write tool
 - burbfile class
 - tdpack thermodynamic constants and functions
 - grid coor. rotation / transformation functions

 See also:
     rpnpy.utils.fstd3d
     rpnpy.utils.burpfile
     rpnpy.utils.thermoconsts
     rpnpy.utils.thermofunc
     rpnpy.utils.llacar

"""

from rpnpy.version import *

__SUBMODULES__ = ['fstd3d', 'burpfile', 'thermoconsts', 'thermofunc', 'llacar']
__all__ = __SUBMODULES__


if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
