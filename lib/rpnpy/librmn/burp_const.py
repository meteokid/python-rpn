#!/usr/bin/env python
# -*- coding: utf-8 -*-
# . s.ssmuse.dot /ssm/net/hpcs/201402/02/base \
#                /ssm/net/hpcs/201402/02/intel13sp1u2 /ssm/net/rpn/libs/15.2
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Module librmn_burp_const defines a set of helper constants to make code
using the librmn burp module more readable.
"""
import numpy  as _np

#=== BURP Constants ===

#<source lang=python>
MRBCVT_DECODE = 0
MRBCVT_ENCODE = 1
#</source>

#==== File mode ====

#<source lang=python>
BURP_MODE_READ   = 'READ'
BURP_MODE_CREATE = 'CREATE'
BURP_MODE_APPEND = 'APPEND'
#</source>

#==== Report Header Flags ====

#<source lang=python>
BURP_FLAGS_IDX_NAME = { #TODO: review
    0  : 'assembled stations',
    ## 0  : '',#observations au-dessus de la terre (masque terre/mer)
    1  : 'surface wind used',
    2  : 'message unreliable (p/t)',
    ## 2  : '', #enregistrement contient des donnees sur la correction de radiation (stations aerologiques)
    3  : 'incorrect coordinates',
    ## 3  : '', #enregistrement contient correction de la position des bateaux, provenant du CQ des bateaux.
    4  : 'message corrected',
    ## 4  : '',#en reserve
    5  : 'message amended',
    ## 5  : '',# station hors du domaine d'interet
    6  : 'station rejected by AO',
    7  : 'station on black list',
    8  : 'station to evaluate',
    9  : 'superobservation',
    ## 9  : '',#decodeur rapporte position de station douteuse
    10 : 'data observed',
    11 : 'data derived',
    12 : 'residues',
    ## 12 : '', #enregistrement contient des données vues par l'AO
    13 : 'verifications',
    ## 13 : '', #enregistrement contient des résidus
    14 : 'TEMP, part RADAT',
    15 : 'TEMP, part A',
    16 : 'TEMP, part B',
    17 : 'TEMP, part C',
    18 : 'TEMP, part D',
    19 : 'reserved1', #enregistrement contient des données analysées5
    20 : 'reserved2', #enregistrement contient des données prévues
    21 : 'reserved3', #enregistrement contient des données de vérification
    22 : 'reserved4',
    23 : 'reserved5'
}
BURP_FLAGS_IDX = dict([(v, k) for k, v in BURP_FLAGS_IDX_NAME.items()])

#==== Data types ====

#<source lang=python>
## String lenght
BURP_STNID_STRLEN = 9

## BURP valid code for data types
BURP_DATYP_LIST = { #TODO: review
    'binary'  : 0,  # 0 = string of bits (bit string)  
    'uint'    : 2,  # 2 = unsigned integers  
    'char'    : 3,  # 3 = characters (NBIT must be equal to 8)  
    'int'     : 4,  # 4 = signed integers  
    'upchar'  : 5,  # 5 = uppercase characters (the lowercase characters
                    #     will be converted to uppercase during the read)
                    #     (NBIT must be equal to 8)  
    'float'   : 6,  # 6 = real*4 (ie: 32bits)
                    # ?? nombres complexes, partie réelle, simple précision (R4)
    'double'  : 7,  # 7 = real*8 (ie: 64bits)  
                    # ?? nombres complexes, partie réelle, double précision (R8)
    'complex' : 8,  # 8 = complex*4 (ie: 2 times 32bits)  
                    # ?? nombres complexes, partie imaginaire, simple précision (I4)
    'dcomplex': 9   # 9 = complex*8 (ie: 2 times 64bits)  
                    # ?? nombres complexes, partie imaginaire, simple précision (I8)
}
BURP_DATYP_NAMES = dict([(v, k) for k, v in BURP_DATYP_LIST.items()])

## Numpy versus BURP data type equivalence
BURP_DATYP2NUMPY_LIST = { #TODO: review
    0: _np.uint32,    # binary, transparent
    2: _np.uint32,    # unsigned integer
    3: _np.uint8,     # character string
    4: _np.int32,     # signed integer
    5: _np.uint8,     # character string (uppercase)
    6: _np.float32,   # floating point      #TODO: review
    7: _np.float64,   # double precision    #TODO: review
    8: _np.complex64, # complex IEEE        #TODO: review
    9: _np.complex128 # double complex IEEE #TODO: review
}
## Note: Type 3 and 5 are processed like strings of bits thus,
##       the user should do the data compression himself.
#</source>

#==== mrfopt (options) ====

#<source lang=python>
BURPOP_MISSING = 'MISSING'
BURPOP_MSGLVL  = 'MSGLVL'

BURPOP_MSG_TRIVIAL = 'TRIVIAL'
BURPOP_MSG_INFO    = 'INFORMATIF'
BURPOP_MSG_WARNING = 'WARNING'
BURPOP_MSG_ERROR   = 'ERROR'
BURPOP_MSG_FATAL   = 'FATAL'
BURPOP_MSG_SYSTEM  = 'SYSTEM'
#</source>
