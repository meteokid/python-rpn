#!/usr/bin/env python

"""
 Module librmn_const defines constants for the librmn module

 @author: Stephane Chamberland <stephane.chamberland@ec.gc.ca>
"""
import numpy  as _np

#--- primitives -----------------------------------------------------

WKOFFIT_TYPE_LIST = { #TODO:
    'STD_RND_89' : 1,
    'STD_RND_98' : 33
    }

#--- base -----------------------------------------------------------

NEWDATE_PRINT2TRUE  = 2
NEWDATE_TRUE2PRINT  = -2
NEWDATE_PRINT2STAMP = 3
NEWDATE_STAMP2PRINT = -3

#--- fstd98/fstd98 --------------------------------------------------

FSTOP_GET = 1
FSTOP_SET = 0

FSTOP_MSGLVL    = "MSGLVL"
FSTOP_TOLRNC    = "TOLRNC"
FSTOP_PRINTOPT  = "PRINTOPT"
FSTOP_TURBOCOMP = "TURBOCOMP"

FSTOPI_MSG_DEBUG   = 0
FSTOPI_MSG_INFO    = 2
FSTOPI_MSG_WARNING = 4
FSTOPI_MSG_ERROR   = 6
FSTOPI_MSG_FATAL   = 8
FSTOPI_MSG_SYSTEM  = 10
FSTOPI_MSG_CATAST  = 10

FSTOPI_TOL_NONE    = 0
FSTOPI_TOL_DEBUG   = 2
FSTOPI_TOL_INFO    = 4
FSTOPI_TOL_WARNING = 6
FSTOPI_TOL_ERROR   = 8
FSTOPI_TOL_FATAL   = 10
 
FSTOPS_MSG_DEBUG   = "DEBUG"
FSTOPS_MSG_INFO    = "INFORM"
FSTOPS_MSG_WARNING = "WARNIN"
FSTOPS_MSG_ERROR   = "ERRORS"
FSTOPS_MSG_FATAL   = "FATALE"
FSTOPS_MSG_SYSTEM  = "SYSTEM"
FSTOPS_MSG_CATAST  = "CATAST"

FSTOPS_TURBO_FAST  = "FAST"
FSTOPS_TURBO_BEST  = "BEST"

FST_RW     = "RND+R/W"
FST_RW_OLD = "RND+R/W+OLD"
FST_RO     = "RND+R/O"

FST_FIND_ANY_INT = -1
FST_FIND_ANY_STR = ' '

FST_GRTYP_LEN  = 1
FST_TYPVAR_LEN = 2
FST_NOMVAR_LEN = 4
FST_ETIKET_LEN = 12

##     0: binary, transparent
##     1: floating point
##     2: unsigned integer
##     3: character (R4A in an integer)
##     4: signed integer
##     5: IEEE floating point
##     6: floating point (16 bit, made for compressor)
##     7: character string
##     8: complex IEEE
##   130: compressed short integer  (128+2)
##   133: compressed IEEE           (128+5)
##   134: compressed floating point (128+6)
## +128 : second stage packer active
## +64  : missing value convention used
FST_DATYP_LIST = {
    'binary' : 0,
    'float' : 1,
    'uint' : 2,
    'char_r4a' : 3,
    'int' : 4,
    'float_IEEE' : 5,
    'float16' : 6,
    'char' : 7,
    'complex' : 8,
    'float_compressed' : 129,
    'uint_compressed' : 130,
    'char_r4a_compressed' : 131,
    'int_compressed' : 132,
    'float_IEEE_compressed' : 133,
    'float16_compressed' : 134,
    'char_compressed' : 135,
    'complex_compressed' : 136,
    }

FST_DATYP2NUMPY_LIST = { #TODO: review
    0: _np.uint32  ,# binary, transparent
    1: _np.float32 ,# floating point
    2: _np.uint32  ,# unsigned integer
    3: _np.uint32  ,# character (R4A in an integer)
    4: _np.int32   ,# signed integer
    5: _np.float32 ,# IEEE floating point
    6: _np.float16 ,# floating point (16 bit, made for compressor)
    7: _np.uint8   ,# character string
    8: _np.complex64 ,# complex IEEE
}

FST_RDE_META_DEFAULT = {
    'dateo' : 0,
    'deet'  : 0,
    'npas'  : 0,
    'ni'    : 1,
    'nj'    : 1,
    'nk'    : 1,
    'nbits' : 16,
    'datyp' : FST_DATYP_LIST['float_IEEE_compressed'],
    'ip1'   : 0,
    'ip2'   : 0,
    'ip3'   : 0,
    'typvar': 'P',
    'nomvar': ' ',
    'etiket': ' ',
    'grtyp' : 'G',
    'ig1'   : 0,
    'ig2'   : 0,
    'ig3'   : 0,
    'ig4'   : 0,
    'swa'   : 0,
    'lng'   : 0,
    'dltf'  : 0,
    'ubc'   : 0,
    'xtra1' : 0,
    'xtra2' : 0,
    'xtra3' : 0
}

#--- fstd98/convip_plus & convert_ip123 ---------------------------------

## KIND = 0, hauteur (m) par rapport au niveau de la mer (-20,000 -> 100,000)
## KIND = 1, sigma   (0.0 -> 1.0)
## KIND = 2, p est en pression (mb)  (0 -> 1100)
## KIND = 3, code arbitraire  (-4.8e8 -> 1.0e10)
## KIND = 4, hauteur (M) par rapport au niveau du sol    (-20,000 -> 100,000)
## KIND = 5, coordonnee hybride        (0.0 -> 1.0)
## KIND = 6, coordonnee theta (1 -> 200,000)
## KIND =10, temps en heure    (0.0 -> 200,000.0)
## KIND =15, reserve (entiers)        
## KIND =17, indice x de la matrice de conversion (1.0 -> 1.0e10)
##           (partage avec kind=1 a cause du range exclusif
## KIND =21, p est en metres-pression  (partage avec kind=5 a cause du range exclusif) (0 -> 1,000,000) fact=1e4
KIND_ABOVE_SEA = 0
KIND_SIGMA     = 1
KIND_PRESSURE  = 2
KIND_ARBITRARY = 3
KIND_ABOVE_GND = 4
KIND_HYBRID    = 5
KIND_THETA     = 6
KIND_HOURS     = 10
KIND_SAMPLES   = 15
KIND_MTX_IND   = 17
KIND_M_PRES    = 21
LEVEL_KIND_MSL = KIND_ABOVE_SEA
LEVEL_KIND_SIG = KIND_SIGMA
LEVEL_KIND_PMB = KIND_PRESSURE
LEVEL_KIND_ANY = KIND_ARBITRARY
LEVEL_KIND_MGL = KIND_ABOVE_GND
LEVEL_KIND_HYB = KIND_HYBRID
LEVEL_KIND_TH  = KIND_THETA
LEVEL_KIND_MPR = KIND_M_PRES
TIME_KIND_HR   = KIND_HOURS

CONVIP_P2IP = 1 #CONVIP_STYLE_DEFAULT = 1
CONVIP_P2IP_NEW = 2 #CONVIP_STYLE_NEW     = 2
CONVIP_P2IP_OLD = 3 #CONVIP_STYLE_OLD     = 3
CONVIP_IP2P = -1 #CONVIP_IP2P_DEFAULT  = -1
CONVIP_IP2P_31BITS   = 0

#---- interp (ezscint) ----------------------------------------------
EZ_YES = 'YES'
EZ_NO = 'NO'

EZ_OPT_INTERP_DEGREE = 'INTERP_DEGREE'
EZ_INTERP_NEAREST = 'NEAREST'
EZ_INTERP_LINEAR = 'LINEAR'
EZ_INTERP_CUBIC = 'CUBIC'

EZ_OPT_EXTRAP_DEGREE = 'EXTRAP_DEGREE'
EZ_EXTRAP_NONE = 'DO_NOTHING'
EZ_EXTRAP_MAX = 'MAXIMUM'
EZ_EXTRAP_MIN = 'MINIMUM'
EZ_EXTRAP_VALUE = 'VALUE'
EZ_EXTRAP_ABORT = 'ABORT'

EZ_OPT_EXTRAP_VALUE = 'EXTRAP_VALUE'

EZ_OPT_POLAR_CORRECTION = 'POLAR_CORRECTION'
## YES or NO

EZ_OPT_VERBOSE = 'VERBOSE'
## YES or NO

EZ_OPT_MISSING_INTERP_ALG = 'MISSING_INTERP_ALG'
## TBD

EZ_OPT_CLOUD_INTERP_ALG = 'CLOUD_INTERP_ALG'
EZ_CLOUD_INTERP_DISTANCE = 'DISTANCE'
EZ_CLOUD_INTERP_LINEAR = 'LINEAR'

EZ_OPT_USE_1SUBGRID = 'USE_1SUBGRID'
## YES or NO

