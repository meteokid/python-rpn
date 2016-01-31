#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Module vgd.const defines constants for the vgd module
"""
import ctypes as _ct
## import numpy  as _np
## import numpy.ctypeslib as _npc

VGD_OK       = 0
VGD_ERROR    = -1
VGD_MISSING  = -9999.
VGD_MAXSTR_NOMVAR = 5
VGD_MAXSTR_TYPVAR = 3
VGD_MAXSTR_ETIKET = 13
VGD_MAXSTR_GRTYP  = 2

VGD_ALLOW_RESHAPE = 0

VGD_SIGM_KIND = 1 #Sigma
VGD_SIGM_VER  = 1
VGD_ETA_KIND = 1 #Eta
VGD_ETA_VER  = 2
VGD_HYBN_KIND = 1 #Hybrid Normalized
VGD_HYBN_VER  = 3
VGD_PRES_KIND = 2 #pressure
VGD_PRES_VER  = 1
VGD_HYB_KIND = 5 #Hybrid Un-staggered
VGD_HYB_VER  = 1
VGD_HYBS_KIND = 5 #Hybrid staggered
VGD_HYBS_VER  = 2
VGD_HYBT_KIND = 5 #Hybrid staggered, first level is a thermo level, unstaggered last Thermo level
VGD_HYBT_VER  = 3
VGD_HYBM_KIND = 5 #Hybrid staggered, first level is a momentum level, same number of thermo and momentum levels
VGD_HYBM_VER  = 4

VGD_KIND_VER = {
    'sigm' : (VGD_SIGM_KIND, VGD_SIGM_VER),
    'eta'  : (VGD_ETA_KIND, VGD_ETA_VER),
    'hybn' : (VGD_HYBN_KIND, VGD_HYBN_VER),
    'pres' : (VGD_PRES_KIND, VGD_PRES_VER),
    'hyb'  : (VGD_HYB_KIND, VGD_HYB_VER),
    'hybs' : (VGD_HYBS_KIND, VGD_HYBS_VER),
    'hybt' : (VGD_HYBT_KIND, VGD_HYBT_VER),
    'hybm' : (VGD_HYBM_KIND, VGD_HYBM_VER)
    }
     
VGD_KEYS = {
    'KIND' : (_ct.c_int, 'Kind of the vertical coordinate ip1'),
    'VERS' : (_ct.c_int, 'Vertical coordinate version. For a given kind there may be many versions, example kind=5 version=2 is hyb staggered GEM4.1'),
    'NL_M' : (_ct.c_int, 'Number of momentum levels (verison 3.2.0 and up)'),
    'NL_T' : (_ct.c_int, 'Number of thermodynamic levels (version 3.2.0 and up)'),
    'CA_M' : (_ct.POINTER(_ct.c_double), 'Values of coefficient A on momentum levels'),
    'CA_T' : (_ct.POINTER(_ct.c_double), 'Values of coefficient A on thermodynamic levels'),
    'CB_M' : (_ct.POINTER(_ct.c_double), 'Values of coefficient B on momentum levels'),
    'CB_T' : (_ct.POINTER(_ct.c_double), 'Values of coefficient B on thermodynamic levels'),
    'COFA' : (_ct.POINTER(_ct.c_double), 'Values of coefficient A in unstaggered levelling'),
    'COFB' : (_ct.POINTER(_ct.c_double), 'Values of coefficient B in unstaggered levelling'),
    'DIPM' : (_ct.c_int, 'The IP1 value of the momentum diagnostic level'),
    'DIPT' : (_ct.c_int, 'The IP1 value of the thermodynamic diagnostic level'),
    'DHM'  : (_ct.POINTER(_ct.c_float), 'Height of the momentum diagonstic level (m)'),
    'DHT'  : (_ct.POINTER(_ct.c_float), 'Height of the thermodynamic diagonstic level (m)'),
    'PREF' : (_ct.c_double, 'Pressure of the reference level (Pa)'),
    'PTOP' : (_ct.c_double, 'Pressure of the top level (Pa)'),
    'RC_1' : (_ct.c_float, 'First coordinate recification R-coefficient'),
    'RC_2' : (_ct.c_float, 'Second coordinate recification R-coefficient'),
    'RFLD' : (_ct.c_char_p, 'Name of the reference field for the vertical coordinate in the FST file'),
    'VCDM' : (_ct.POINTER(_ct.c_float), 'List of momentum coordinate values'),
    'VCDT' : (_ct.POINTER(_ct.c_float), 'List of thermodynamic coordinate values'),
    'VIPM' : (_ct.POINTER(_ct.c_int), 'List of IP1 momentum values associated with this coordinate'),
    'VIPT' : (_ct.POINTER(_ct.c_int), 'List of IP1 thermodynamic values associated with this coordinate'),
    'VTBL' : (_ct.POINTER(_ct.c_double), 'real*8 Fortran array containing all vgrid_descriptor information'),
    'LOGP' : (_ct.c_int, 'furmula gives log(p) T/F (version 1.0.3 and greater). True -> Formula with A and B gives log(p), False -> Formula with A and B gives p    ')
    }


# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
