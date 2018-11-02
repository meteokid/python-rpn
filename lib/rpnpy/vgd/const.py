#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Module vgd.const defines constants for the vgd module

Notes:
    This module is a very close ''port'' from the original
    [[librmn]]'s [[Vgrid]] package.<br>
    You may want to refer to the [[Vgrid]] documentation for more details.

See Also:
    rpnpy.vgd.proto
    rpnpy.vgd.base

Details:
    See Source Code
"""
#import ctypes as _ct
## import numpy  as _np
## import numpy.ctypeslib as _npc

#TODO: add support for Vcode 5100 (SLEVE pressure) and Vcode 5999 (hyb unstaggered of unspecified origin e.g. ECMWF)

_MB2PA = 100.

##DETAILS_START
#== Constants Details ==
#<source lang="python">

VGD_OK       = 0
VGD_ERROR    = -1
VGD_MISSING  = -9999.
VGD_MAXSTR_NOMVAR = 5
VGD_MAXSTR_TYPVAR = 3
VGD_MAXSTR_ETIKET = 13
VGD_MAXSTR_GRTYP  = 2

VGD_ALLOW_RESHAPE  = 0
VGD_ALLOW_SIGMA    = 1
VGD_DISALLOW_SIGMA = 0

VGD_SIGM_KIND   = 1  # Sigma
VGD_SIGM_VER    = 1
VGD_ETA_KIND    = 1  # Eta
VGD_ETA_VER     = 2
VGD_HYBN_KIND   = 1  # Hybrid Normalized, cannot be generated, use Eta
VGD_HYBN_VER    = 3
VGD_PRES_KIND   = 2  # pressure
VGD_PRES_VER    = 1
VGD_HYB_KIND    = 5  # Hybrid Un-staggered
VGD_HYB_VER     = 1
VGD_HYBS_KIND   = 5  # Hybrid staggered
VGD_HYBS_VER    = 2
VGD_HYBT_KIND   = 5  # Hybrid staggered, first level is a thermo level,
                     # unstaggered last Thermo level
VGD_HYBT_VER    = 3
VGD_HYBM_KIND   = 5  # Hybrid staggered, first level is a momentum level,
                     # same number of thermo and momentum levels
VGD_HYBM_VER    = 4
VGD_HYBMD_KIND  = 5  # Hybrid staggered, first level is a momentum level,
                     # same number of thermo and momentum levels,
                     # Diag level heights (m AGL) encoded
VGD_HYBMD_VER  = 5
VGD_HYBPS_KIND = 5   # Hybrid staggered CP grid pressure SLEVE levels
VGD_HYBPS_VER  = 100 #
VGD_HYBH_KIND  = 21  # Hydrid staggered CP grid height levels
VGD_HYBH_VER   = 1   #
VGD_HYBHS_KIND = 21  # Hydrid staggered CP grid height SLEVE level
VGD_HYBHS_VER  = 1   #
VGD_HYBHL_KIND = 21  # Hydrid staggered Lorenz grid heightlevel
VGD_HYBHL_VER  = 2   #
VGD_HYBHLS_KIND= 21  # Hydrid staggered Lorenz grid height SLEVE level
VGD_HYBHLS_VER = 2   #

VGD_DIAG_LOGP  = 1   # vgd_diag_withref: output log pressure
VGD_DIAG_PRES  = 0   # vgd_diag_withref: output pressure
VGD_DIAG_DPI   = 1   # vgd_diag_withref: output pressure
VGD_DIAG_DPIS  = 0   # vgd_diag_withref: output hydrostatic pressure partial
                     # derivative with respect to surface hydrostatic pressure,
                     # default used in vgd_levels

VGD_RFLD_CONV = {  # Convert functions for RFLD/RFLS from RPNSTD files to VGD units (SI)
    "P0": lambda x: x * _MB2PA  # Convert P0 from [mb] to [Pa]
    }
VGD_RFLD_CONV_KEYS = VGD_RFLD_CONV.keys()

VGD_KIND_VER = {
    'sigm'   : (VGD_SIGM_KIND, VGD_SIGM_VER),    #1,1
    'eta'    : (VGD_ETA_KIND, VGD_ETA_VER),      #1,2
    'hybn'   : (VGD_HYBN_KIND, VGD_HYBN_VER),    #cannot be generated, use Eta
    'pres'   : (VGD_PRES_KIND, VGD_PRES_VER),    #2,1
    'hyb'    : (VGD_HYB_KIND, VGD_HYB_VER),      #5,1
    'hybs'   : (VGD_HYBS_KIND, VGD_HYBS_VER),    #5,2
    'hybt'   : (VGD_HYBT_KIND, VGD_HYBT_VER),    #5,3
    'hybm'   : (VGD_HYBM_KIND, VGD_HYBM_VER),    #5,4
    'hybmd'  : (VGD_HYBMD_KIND, VGD_HYBMD_VER),  #5,5
    'hybps'  : (VGD_HYBPS_KIND, VGD_HYBPS_VER),  #5,100
    'hybh'   : (VGD_HYBH_KIND, VGD_HYBH_VER),    #21,1
    'hybhs'  : (VGD_HYBHS_KIND, VGD_HYBHS_VER),  #21,1
    'hybhl'  : (VGD_HYBHL_KIND, VGD_HYBHL_VER),  #21,2
    'hybhls' : (VGD_HYBHLS_KIND, VGD_HYBHLS_VER),#21,2
    }
VGD_KIND_VER_INV_VCODE = dict([("{0:03d}{1:1d}".format(v[0]*100, v[1]), k)
                                for k, v in VGD_KIND_VER.items()])
VGD_KIND_VER_INV = dict([(v, k) for k, v in VGD_KIND_VER.items()])

VGD_OPR_KEYS = {
    'get_char'      : ["ETIK", "NAME", "RFLD", "RFLS"],
    'put_char'      : ["ETIK"],
    'get_int'       : ["NL_M", "NL_T", "NL_W", "KIND", "VERS", "DATE", "IG_1",
                       "IG_2", "IG_3", "IG_4", "IP_1", "IP_2", "DIPM", "DIPT",
                       "MIPG", "LOGP"],
    'put_int'       : ["DATE", "IG_1", "IG_2", "IG_3", "IG_4", "IP_1", "IP_2",
                       "IP_3", "DIPM", "DIPT"],
    'get_float'     : ["RC_1", "RC_2", "RC_3", "RC_4", "DHM", "DHT"],
    'get_int_1d'    : ["VIP1", "VIPM", "VIPT", "VIPW"],
    'get_float_1d'  : ["VCDM", "VIPM", "VCDT", "VIPT"],
    'put_double'    : [],
    'get_double'    : ["PTOP", "PREF", "RC_1", "RC_2"],
    'get_double_1d' : ["CA_M", "COFA", "CB_M", "COFB", "CA_T", "CB_T"],
    'get_double_3d' : ["VTBL"],
    'getopt_int'    : ["ALLOW_SIGMA"],
    'putopt_int'    : ["ALLOW_SIGMA"]
    }

VGD_KEYS = {
    'KIND' : ('Kind of the vertical coordinate ip1'),
    'VERS' : ('Vertical coordinate version. For a given kind there may be ' +
              'many versions, example kind=5 version=2 is hyb staggered GEM4.1'),
    'NL_M' : ('Number of momentum levels (verison 3.2.0 and up)'),
    'NL_T' : ('Number of thermodynamic levels (version 3.2.0 and up)'),
    'CA_M' : ('Values of coefficient A on momentum levels'),
    'CA_T' : ('Values of coefficient A on thermodynamic levels'),
    'CA_W' : ('Values of coefficient A on vertical velocity levels'),
    'CB_M' : ('Values of coefficient B on momentum levels'),
    'CB_T' : ('Values of coefficient B on thermodynamic levels'),
    'CB_W' : ('Values of coefficient B on vertical velocity levels'),
    'CC_M' : ('Values of coefficient C on momentum levels'),
    'CC_T' : ('Values of coefficient C on thermodynamic levels'),
    'CC_W' : ('Values of coefficient C on vertical velocity levels'),
    'COFA' : ('Values of coefficient A in unstaggered levelling'),
    'COFB' : ('Values of coefficient B in unstaggered levelling'),
    'DIPM' : ('The IP1 value of the momentum diagnostic level'),
    'DIPT' : ('The IP1 value of the thermodynamic diagnostic level'),
    'DIPW' : ('The IP1 value of the vertical velocity diagnostic level'),
    'DHM'  : ('Height of the momentum diagonstic level (m)'),
    'DHT'  : ('Height of the thermodynamic diagonstic level (m)'),
    'DHW'  : ('Height of the vertical velocity diagonstic level (m)'),
    'PREF' : ('Pressure of the reference level (Pa)'),
    'PTOP' : ('Pressure of the top level (Pa)'),
    'RC_1' : ('First coordinate recification R-coefficient'),
    'RC_2' : ('Second coordinate recification R-coefficient'),
    'RC_3' : ('Third coordinate recification R-coefficient (SLEVE ONLY)'),
    'RC_4' : ('Fourth coordinate recification R-coefficient (SLEVE ONLY)'),
    'RFLD' : ('Name of the reference field for the vertical coordinate in the FST file'),
    'RFLS' : ('Name of the large scale reference field for the vertical coordinate in the FST file (SLEVE ONLY)'),
    'VCDM' : ('List of momentum coordinate values'),
    'VCDT' : ('List of thermodynamic coordinate values'),
    'VCDW' : ('List of vertical velocity coordinate values'),
    'VIPM' : ('List of IP1 momentum values associated with this coordinate'),
    'VIPT' : ('List of IP1 thermodynamic values associated with this coordinate'),
    'VIPW' : ('List of IP1 vertical velocity values associated with this coordinate'),
    'VTBL' : ('real*8 Fortran 3d array containing all vgrid_descriptor information'),
    'LOGP' : ('furmula gives log(p) T/F (version 1.0.3 and greater). ' +
              'True -> Formula with A and B gives log(p), ' +
              'False -> Formula with A and B gives p'),
    'ALLOW_SIGMA' : ('Allow definition of sigma coor or not')
    }

#</source>
##DETAILS_END


if __name__ == "__main__":
    import doctest
    doctest.testmod()

# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
