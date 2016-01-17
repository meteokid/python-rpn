#!/usr/bin/env python

"""
Module vgd.const defines constants for the vgd module

@author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
"""
#import numpy  as _np

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
    }
