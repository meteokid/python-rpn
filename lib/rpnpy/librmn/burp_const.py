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
import numpy as _np

#=== BURP Constants ===

#TODO: cleanup
## See:
## * ls -lL $AFSISIO/datafiles/constants/ | grep -i burp
## ** $AFSISIO/datafiles/constants/tableburp_[ef].val 
##    table_b_bufr_[ef]_opsvalid_v23
## ** $AFSISIO/datafiles/constants/tableburp [fr]
##    $AFSISIO/datafiles/constants/table_b_bufr_e [en]
## * ls -lL $AFSISIO/datafiles/constants/ | grep -i bufr
##
## * 3 types of files:
## ** table_b_bufr_master, table_b_bufr_[ef], table_b_bufr_[ef]_opsvalid_v23
## ** table_d_bufr_[ef]
## ** tabloc_bufr_[ef]
##
## Also:
## * ade*bufr*
## * libecbufr_tables/*

#NOTE: BUFR numérote les bits de 1 à 7, le bit 1 étant celui de poids le plus élevé
#<source lang=python>
BURP2BIN = lambda v,l=32: "{{0:0{}b}}".format(l).format(v)
BURP2BIN2LIST_BUFR = lambda v,l=32: [int(i) for i in list(BURP2BIN(v,l))]
BURP2BIN2LIST = lambda v,l=32: BURP2BIN2LIST_BUFR(v,l)[::-1]

MRBCVT_DECODE = 0
MRBCVT_ENCODE = 1

BURP_TABLE_B_FILENAME = 'table_b_bufr_e'
#</source>

#<source lang=python>
BURP_STNID_STRLEN = 9
#</source>

#==== mrfopt (options) ====

#<source lang=python>
BURP_OPTC_STRLEN = 9

BURPOP_MISSING = 'MISSING'
BURPOP_MSGLVL  = 'MSGLVL'

BURPOP_MSG_TRIVIAL = 'TRIVIAL  '
BURPOP_MSG_INFO    = 'INFORMATIF'
BURPOP_MSG_WARNING = 'WARNING  '
BURPOP_MSG_ERROR   = 'ERROR    '
BURPOP_MSG_FATAL   = 'ERRFATAL '
BURPOP_MSG_SYSTEM  = 'SYSTEM   '
#</source>

#==== File mode ====

#<source lang=python>
BURP_MODE_READ   = 'READ'
BURP_MODE_CREATE = 'CREATE'
BURP_MODE_APPEND = 'APPEND'
#</source>

#==== Report Header Flags ====
#NOTE: BUFR numérote les bits de 1 à 7, le bit 1 étant celui de poids le plus élevé, soit «64» dans cet exemple.

#<source lang=python>
BURP_FLAGS_IDX_NAME = { #TODO: review
    0  : 'assembled stations',
    ## 0  : '',#observations au-dessus de la terre (masque terre/mer)
    1  : 'surface wind used',
    2  : 'message unreliable (p/t)',
    ## 2  : '', #data sur la correction de radiation (stations aerologiques)
    3  : 'incorrect coordinates',
    ## 3  : '', #correction de la position des bateaux, provenant du CQ des bateaux.
    4  : 'message corrected',
    ## 4  : '',#en reserve
    5  : 'message amended',
    ## 5  : '',# station hors du domaine d'interet
    6  : 'station rejected by AO',
    7  : 'station on black list',
    8  : 'station to evaluate',
    9  : 'superobservation',
    ## 9  : '',#decodeur rapporte position de station douteuse ?incorrect coordinates?
    10 : 'data observed',
    11 : 'data derived',
    12 : 'residues',
    ## 12 : '', #data vues by AO
    13 : 'verifications',
    ## 13 : '', #'residues'
    14 : 'TEMP part RADAT',
    15 : 'TEMP part A',
    16 : 'TEMP part B',
    17 : 'TEMP part C',
    18 : 'TEMP part D',
    19 : 'reserved1', #'data analysed'
    20 : 'reserved2', #'data forecast'
    21 : 'reserved3', #'verifications'
    22 : 'reserved4',
    23 : 'reserved5'
}
BURP_FLAGS_IDX = dict([(v, k) for k, v in BURP_FLAGS_IDX_NAME.items()])
#</source>


#<source lang=python>
BURP_IDTYP_DESC = { 
    '12' : 'SYNOP, NON AUTOMATIQUE',
    '13' : 'SHIP, NON AUTOMATIQUE',
    '14' : 'SYNOP MOBIL',
    '15' : 'METAR',
    '16' : 'SPECI',
    '18' : 'DRIFTER',
    '20' : 'RADOB',
    '22' : 'RADREP',
    '32' : 'PILOT',
    '33' : 'PILOT SHIP',
    '34' : 'PILOT MOBIL',
    '35' : 'TEMP',
    '36' : 'TEMP SHIP',
    '37' : 'TEMP DROP',
    '38' : 'TEMP MOBIL',
    '39' : 'ROCOB',
    '40' : 'ROCOB SHIP',
    '41' : 'CODAR',
    '42' : 'AMDAR (AIRCRAFT METEOROLOGICAL DATA REPORT)',
    '44' : 'ICEAN',
    '45' : 'IAC',
    '46' : 'IAC FLEET',
    '47' : 'GRID',
    '49' : 'GRAF',
    '50' : 'WINTEM',
    '51' : 'TAF',
    '53' : 'ARFOR',
    '54' : 'ROFOR',
    '57' : 'RADOF',
    '61' : 'MAFOR',
    '62' : 'TRACKOB',
    '63' : 'BATHY7',
    '64' : 'TESAC',
    '65' : 'WAVEOB',
    '67' : 'HYDRA',
    '68' : 'HYFOR',
    '71' : 'CLIMAT',
    '72' : 'CLIMAT SHIP',
    '73' : 'NACLI/CLINP/SPCLI/CLISA/INCLI',
    '75' : 'CLIMAT TEMP',
    '76' : 'CLIMAT TEMP SHIP',
    '81' : 'SFAZI',
    '82' : 'SFLOC',
    '83' : 'SFAZU',
    '85' : 'SAREP',
    '86' : 'SATEM',
    '87' : 'SARAD',
    '88' : 'SATOB',
    '92' : 'GRIB',
    '94' : 'BUFR',
    '127' : "DONNEES DE SURFACE DE QUALITE DE L’AIR",
    '128' : 'AIREP',
    '129' : 'PIREP',
    '130' : 'PROFILEUR DE VENT',
    '131' : 'SUPEROBS DE SYNOP',
    '132' : 'SUPEROBS DE AIREP',
    '133' : 'SA + SYNOP',
    '134' : "PAOBS (PSEUDO-DONNEES D'AUSTRALIE)",
    '135' : 'TEMP + PILOT',
    '136' : 'TEMP + SYNOP',
    '137' : 'PILOT + SYNOP',
    '138' : 'TEMP + PILOT + SYNOP',
    '139' : 'TEMP SHIP + PILOT SHIP',
    '140' : 'TEMP SHIP + SHIP',
    '141' : 'PILOT SHIP + SHIP',
    '142' : 'TEMPS SHIP + PILOT SHIP + SHIP',
    '143' : 'SAWR, STATION NON AUTOMATIQUE (REGULIER OU REGULIER SPECIAL)',
    '144' : 'SAWR, STATION AUTOMATIQUE (REGULIER OU REGULIER SPECIAL)',
    '145' : 'SYNOP ("PATROL SHIPS")',
    '146' : 'ASYNOP, STATION AUTOMATIQUE',
    '147' : 'ASHIP, STATION AUTOMATIQUE, (BOUEES FIXES, PLATES-FORMES.)',
    '148' : 'SAWR, STATION NON AUTOMATIQUE (SPECIAL)',
    '149' : 'SAWR, STATION AUTOMATIQUE (SPECIAL)',
    '150' : 'PSEUDO-DONNEES DU CMC, SURFACE, MODE ANALYSE',
    '151' : 'PSEUDO-DONNEES DU CMC, ALTITUDE, MODE ANALYSE',
    '152' : 'PSEUDO-DONNEES DU CMC, SURFACE, MODE REPARATION',
    '153' : 'PSEUDO-DONNEES DU CMC, ALTITUDE, MODE REPARATION',
    '154' : 'PREVISIONS DE VENTS DE TYPE FD',
    '155' : 'PREVISIONS DE VENTS DE TYPE FD AMENDEES',
    '156' : 'PREVISIONS STATISTIQUES DES ELEMENTS DU TEMPS',
    '157' : 'ACARS (AIRCRAFT METEOROLOGICAL DATA REPORT)',
    '158' : 'HUMSAT',
    '159' : 'TEMP MOBIL + PILOT MOBIL',
    '160' : 'TEMP MOBIL + SYNOP MOBIL',
    '161' : 'PILOT MOBIL + SYNOP MOBIL',
    '162' : 'TEMP MOBIL + PILOT MOBIL + SYNOP MOBIL',
    '163' : 'RADAR',
    '164' : 'RADIANCES TOVS AMSUA',
    '165' : 'PROFILS VERTICAUX ANALYSES OU PREVUS',
    '166' : 'MOS EVOLUTIF (PROJET PENSE)',
    '167' : 'DONNEES SATELLITAIRES PROVENANTDE SCATTEROMÈTRES (ERS, ADEOS, ETC.)',
    '168' : 'DONNEES SATELLITAIRES DE TYPE SSMI',
    '169' : 'RADIO-OCCULTATIONS',
    '170' : 'OZONE',
    '171' : 'METEOSAT',
    '172' : 'STANDARD HYDROMETEOROLOGICAL EXCHANGE FORMAT (S.H.E.F.)',
    '173' : 'VERIFICATIONS DES MODÈLES DU CMC',
    '174' : 'DONNEES SATELLITAIRES PROVENANTDE RADARS À OUVERTURE SYNTHETIQUE (ERS, ETC.)',
    '175' : 'DONNEES SATELLITAIRES PROVENANT D’ALTIMÈTRES RADAR (ERS, ETC.)',
    '176' : 'STATIONS D’UN RESEAU COOPERATIF (INTERDIT DE REDISTRIBUER LES DONNEES)',
    '177' : 'ADS AUTOMATED DEPENDANCE SURVEILLANCE (AIREP AUTOMATIQUE)',
    '178' : 'DONNEES PROVENANTDE ICEC POUR LES LACS',
    '179' : 'DONNEES PROVENANTDE ICEC POUR LES OCEANS',
    '180' : 'RADIANCES GOES',
    '181' : 'RADIANCES ATOVS AMSUB',
    '182' : 'RADIANCES MHS',
    '183' : 'DONNEES AIRS',
    '184' : 'RADIANCES (GENERIQUE)',
    '188' : 'DONNEES SATELLITAIRES DE VENT AMELIOREES (FORMAT BUFR)',
    '189' : 'DONNEES DE SURFACE GPS'
}

BURP_IDTYP_IDX = dict([(v, int(k)) for k, v in BURP_IDTYP_DESC.items()])
#</source>

#==== Data types ====

## BURP valid code for data types
#<source lang=python>
BURP_DATYP_LIST = { #TODO: review
    'binary'  : 0,  # 0 = string of bits (bit string)  
    'uint'    : 2,  # 2 = unsigned integers  
    'char'    : 3,  # 3 = characters (NBIT must be equal to 8)  
    'int'     : 4,  # 4 = signed integers  
    'upchar'  : 5,  # 5 = uppercase characters (the lowercase characters
                    #     will be converted to uppercase during the read)
                    #     (NBIT must be equal to 8)  
    'float'   : 6,  # 6 = real*4 (ie: 32bits)  #TODO: review
                    # ?? nombres complexes, partie réelle, simple précision (R4)
    'double'  : 7,  # 7 = real*8 (ie: 64bits)  #TODO: review
                    # ?? nombres complexes, partie réelle, double précision (R8)
    'complex' : 8,  # 8 = complex*4 (ie: 2 times 32bits)  #TODO: review
                    # ?? nombres complexes, partie imaginaire, simple précision (I4)
    'dcomplex': 9   # 9 = complex*8 (ie: 2 times 64bits)  #TODO: review
                    # ?? nombres complexes, partie imaginaire, simple précision (I8)
}
BURP_DATYP_NAMES = dict([(v, k) for k, v in BURP_DATYP_LIST.items()])
#</source>

## Numpy versus BURP data type equivalence
#<source lang=python>
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

#==== Data types ====
#<source lang=python>
BURP_BKNAT_MULTI_DESC = {
    0 : 'uni',
    1 : 'multi'
    }
BURP_BKNAT_KIND_DESC = {
    0 : 'data',
    1 : 'info',
    2 : 'desc3d',
    3 : 'flags'
    }

BURP_BKTYP_ALT_DESC = {
    0 : 'surf',
    1 : 'alt'
    }

BURP_BKTYP_KIND_DESC = {
    0 : 'observations (ADE)',
    1 : 'row observations (not decoded)',
    2 : 'derived data, entry to the OA at altitude, global model',
    3 : 'derived data, entry to the OA at altitude, regional model',
    4 : 'derived data, entry to the OA at surface, global model',
    5 : 'derived data, entry to the OA at surface, regional model',
    6 : 'data seen by OA at altitude, global model',
    7 : 'data seen by OA at altitude, regional model',
    8 : 'data seen by OA at surface, global model',
    9 : 'data seen by OA at surface, regional model',
    10 : 'vertical profils, AO at altitude, global model',
    11 : 'vertical profils, AO at altitude, regional model',
    12 : 'reserved',
    13 : 'reserved',
    14 : 'analysed value (incuding residues) by OA at altitude, global model',
    15 : 'analysed value (incuding residues) by OA at altitude, regional model',
    16 : 'analysed value (incuding residues) by OA at surface, global model',
    17 : 'analysed value (incuding residues) by OA at surface, regional model',
    18 : 'forecast, global model',
    19 : 'forecast, regional model',
    20 : 'statistics of weather elements (PENSE project)',
    21 : 'statistics of weather elements (Kalman filter, PENSE project)',
    22 : 'SSMI data',
    23 : 'forecast, GEPS',
    24 : 'forecast, REPS',
    25 : 'probabilistic forecast',
    25 : 'deterministic forecast',
    27 : 'QC weather elements (QCOBS)',
    28 : 'QA of DMSobservations'
    }

## (bktyp_alt, bktyp_kind, bkstp) : description
BURP_BKSTP_DESC = { #TODO
    (0, 0, 0) : 'observed value',
    (0, 1, 0) : 'observed value',
    (0, 0, 1) : 'correction to position and/or identificator',
    (0, 1, 1) : 'correction to position and/or identificator',
    (1, 0, 1) : 'RADAT (TEMP) ou Ts de SATOB ou SATEM ou TOVS/temperature',
    (1, 1, 1) : 'RADAT (TEMP) ou Ts de SATOB ou SATEM ou TOVS/temperature',
    (1, 0, 2) : "partie A (SATEM, TEMP, PILOT, SATOB) ouTOVS/luminance",
    (1, 1, 2) : "partie A (SATEM, TEMP, PILOT, SATOB) ou TOVS/luminance",
    (1, 0, 3) : "partie B (SATEM, TEMP, PILOT)",
    (1, 1, 3) : "partie B (SATEM, TEMP, PILOT)",
    (1, 0, 4) : "partie C (SATEM, TEMP, PILOT)",
    (1, 4, 4) : "partie C (SATEM, TEMP, PILOT)",
    (1, 0, 5) : "partie D (SATEM, TEMP, PILOT)",
    (1, 1, 5) : "partie D (SATEM, TEMP, PILOT)",
    (1, 0, 6) : "délais de réception pour messages à parties multiples (ex. SATEM, TEMP, PILOT et SATOB)",
    (1, 1, 6) : "délais de réception pour messages à parties multiples (ex. SATEM, TEMP, PILOT et SATOB)",
    (1, 0, 8) : "statistiques de champs",
    (1, 1, 8) : "statistiques de champs",
    (1, 0, 9) : "statistiques NN de champs",
    (1, 1, 9) : "statistiques NN de champs",
    (1, 0, 10) : "statistiques de différences",
    (1, 1, 10) : "statistiques de différences",
    (1, 0, 11) : "Bloc A, SSMI, (19H, 19V, 22V, 37H, 37V) *64",
    (1, 1, 11) : "Bloc A, SSMI, (19H, 19V, 22V, 37H, 37V) *64",
    (1, 0, 12) : "Bloc B, SSMI, (85H, 85V) *128 points, 1/2",
    (1, 1, 12) : "Bloc B, SSMI, (85H, 85V) *128 points, 1/2",
    (1, 0, 13) : "Bloc C, SSMI, (85H, 85V) *128 points, 2/2",
    (1, 1, 13) : "Bloc C, SSMI, (85H, 85V) *128 points, 2/2",
    (0, 2, 0) : "niveau de surface d’origine indéterminé ** (TEMP et PILOT only)",
    (0, 3, 0) : "niveau de surface d’origine indéterminé **  (TEMP et PILOT only)",
    (0, 2, 1) : "surface provenant d'un SYNOP **",
    (0, 3, 1) : "surface provenant d'un SYNOP **",
    (0, 2, 2) : "surface provenant d'un TEMP **",
    (0, 3, 2) : "surface provenant d'un TEMP **",
    (0, 2, 3) : "statistiques de champs **",
    (0, 3, 3) : "statistiques de champs **",
    (0, 2, 8) : "statistiques de champs",
    (0, 3, 8) : "statistiques de champs",
    (0, 2, 9) : "statistiques NN de champs",
    (0, 3, 9) : "statistiques NN de champs",
    (0, 2, 10) : "statistiques de différences",
    (0, 3, 10) : "statistiques de différences",
    (1, 2, 8) : "statistiques de champs",
    (1, 3, 8) : "statistiques de champs",
    (1, 2, 9) : "statistiques NN de champs",
    (1, 3, 9) : "statistiques NN de champs",
    (1, 2, 10) : "statistiques de différences",
    (1, 3, 10) : "statistiques de différences",
    (0, 4, 8) : "statistiques de champs",
    (0, 5, 8) : "statistiques de champs",
    (0, 4, 9) : "statistiques NN de champs",
    (0, 5, 9) : "statistiques NN de champs",
    (0, 4, 10) : "statistiques de différences",
    (0, 5, 10) : "statistiques de différences",
    (0, 6, 8) : "statistiques de champs",
    (0, 7, 8) : "statistiques de champs",
    (0, 6, 9) : "statistiques NN de champs",
    (0, 7, 9) : "statistiques NN de champs",
    (0, 6, 10) : "statistiques de différences (résidus)",
    (0, 7, 10) : "statistiques de différences (résidus)",
    (0, 6, 11) : "statistiques de résidus",
    (0, 7, 11) : "statistiques de résidus",
    (0, 6, 12) : "statistiques NN de résidus",
    (0, 7, 12) : "statistiques NN de résidus",
    (0, 6, 14) : "statistiques d'erreur d'observation",
    (0, 7, 14) : "statistiques d'erreur d'observation",
    (0, 6, 15) : "statistiques d'erreur de prévision",
    (0, 7, 15) : "statistiques d'erreur de prévision",
    (1, 6, 8) : "statistiques de champs",
    (1, 7, 8) : "statistiques de champs",
    (1, 6, 9) : "statistiques NN de champs",
    (1, 7, 9) : "statistiques NN de champs",
    (1, 6, 10) : "statistiques de différences (résidus)",
    (1, 7, 10) : "statistiques de différences (résidus)",
    (1, 6, 11) : "statistiques de résidus",
    (1, 7, 11) : "statistiques de résidus",
    (1, 6, 12) : "statistiques NN de résidus",
    (1, 7, 12) : "statistiques NN de résidus",
    (1, 6, 14) : "statistiques d'erreur d'observation",
    (1, 7, 14) : "statistiques d'erreur d'observation",
    (1, 6, 15) : "statistiques d'erreur de prévision",
    (1, 7, 15) : "statistiques d'erreur de prévision",
    (0, 8, 8) : 'statistiques de champs',
    (0, 9, 8) : 'statistiques de champs',
    (0, 8, 9) : 'statistiques NN de champs',
    (0, 9, 9) : 'statistiques NN de champs',
    (0, 8, 10) : 'statistiques de différences (résidus)',
    (0, 9, 10) : 'statistiques de différences (résidus)',
    (0, 8, 11) : 'statistiques de résidus',
    (0, 9, 11) : 'statistiques de résidus',
    (0, 8, 12) : 'statistiques NN de résidus',
    (0, 9, 12) : 'statistiques NN de résidus',
    (0, 8, 14) : "statistiques d'erreur d'observation",
    (0, 9, 14) : "statistiques d'erreur d'observation",
    (0, 8, 15) : "statistiques d'erreur de prévision",
    (0, 9, 15) : "statistiques d'erreur de prévision",
    (1, 10, 1) : "prévu par champ d’essai",
    (1, 11, 1) : "prévu par champ d’essai",
    (1, 10, 2) : 'prévu par modèle',
    (1, 11, 2) : 'prévu par modèle',
    (1, 10, 3) : 'analysé',
    (1, 11, 3) : 'analysé',
    (0, 14, 0) : 'O-A',
    (0, 15, 0) : 'O-A',
    (0, 16, 0) : 'O-A',
    (0, 17, 0) : 'O-A',
    (0, 14, 1) : 'O-F',
    (0, 15, 1) : 'O-F',
    (0, 16, 1) : 'O-F',
    (0, 17, 1) : 'O-F',
    (0, 14, 2) : 'O-I',
    (0, 15, 2) : 'O-I',
    (0, 16, 2) : 'O-I',
    (0, 17, 2) : 'O-I',
    (0, 14, 8) : 'statistiques de champs',
    (0, 15, 8) : 'statistiques de champs',
    (0, 16, 8) : 'statistiques de champs',
    (0, 17, 8) : 'statistiques de champs',
    (0, 14, 9) : 'statistiques NN de champs',
    (0, 15, 9) : 'statistiques NN de champs',
    (0, 16, 9) : 'statistiques NN de champs',
    (0, 17, 9) : 'statistiques NN de champs',
    (0, 14, 10) : 'statistiques de différences',
    (0, 15, 10) : 'statistiques de différences',
    (0, 16, 10) : 'statistiques de différences',
    (0, 17, 10) : 'statistiques de différences',
    (0, 18, 0) : "champs d'essai de AO",
    (0, 19, 0) : "champs d'essai de AO",
    (0, 18, 1) : 'sortie de modèle de prévisions',
    (0, 19, 1) : 'sortie de modèle de prévisions',
    (0, 18, 2) : 'statistiques',
    (0, 19, 2) : 'statistiques',
    (0, 18, 8) : 'statistiques de champs',
    (0, 19, 8) : 'statistiques de champs',
    (0, 18, 9) : 'statistiques NN de champs',
    (0, 19, 9) : 'statistiques NN de champs',
    (0, 18, 10) : 'statistiques de différences',
    (0, 19, 10) : 'statistiques de différences',
    (1, 22, 1) : 'A (données, Multi), basse densité',
    (1, 22, 2) : 'B (données, Multi), haute densité',
    (1, 22, 3) : 'C (3-D, Multi), basse densité',
    (1, 22, 4) : 'D (3-D, Multi), haute densité',
    (1, 22, 5) : 'E (données, Multi), type de surface, basse densité',
    (1, 22, 6) : 'F (données, Multi), type de surface, haute densité',
    }
#</source>

#TODO: BURP_BKNAT_KIND_DESC flags:
## Voici les MARQUEURS que l'on nomment primaires.
## Bits	Décimal	Type de données	REF	Description
## 12	4096	AO	1	Élément assimilé (c'est-à-dire ayant influencé l'analyse)
## 11	2048	AO	2	Élément rejeté par un processus de sélection (thinning ou canal)
## 10	1024	AO	3	Élément généré par l'AO
## 9	512	AO	4	Élément rejeté par le contrôle de la qualité de l'AO (Background Check ou QC-Var)
## 8	256	AO	5	Élément rejeté parce qu'il est sur une liste noire
## 7	128	DERIV	6	En réserve
## 6	64	DERIV	7	Élément corrigé par la séquence DERIVATE ou correction de biais
## 5	32	DERIV	8	Élément interpolé, généré par DERIVATE
## 4	16	DERIV	9	Élément douteux
## 3	8	ADE	10	Élément peut-être erroné
## 2	4	ADE	11	Élément erroné
## 1	2	ADE	12	Élément qui excède un extrême climatologique (ou) qui ne passe pas le test de consistance
## 0	1	ADE	13	Élément modifié ou généré par l'ADE
## Pour répondre à de nouveaux besoins, nous avons dû étendre le nombre de marqueurs avec une nouvelle liste qui constituera les marqueurs qu'on dira « secondaires » car ceux-ci seront utilisés en conjonction avec les marqueurs dits « primaires ». Nous avons la possibilité d'étendre jusqu'au bit 32.

## Voici les MARQUEURS que l'on nomment secondaires.
## Bits	Décimal	Type de données	REF	Description
## 13	8192	AO	0	Comparaison contre le champ d'essai, niveau 1
## 14	16384	AO	-1	Comparaison contre le champ d'essai, niveau 2
## 15	32768	AO	-2	Comparaison contre le champ d'essai, niveau 3
## 16	65536	AO	-3	Élément rejeté par la comparaison contre le champ d'essai (Background Check)
## 17	131072	AO	-4	Élément rejeté par le QC-Var
## 18	262144	DERIV	-5	Élément non-utilisé à cause de l'orographie
## 19	524288	DERIV	-6	Élément non-utilisé à cause du masque terre-mer
## 20	1048576	DERIV	-7	Erreur de position d'avion décelée par TrackQc
## 21	2097152	QC	-8	Inconsistance détectée par un processus de CQ

#TODO: BFAM desc list
#TODO: fst, use dict to provide var desc and units, tool rpy.dict

#TODO: stnid=^******* = regrouped data record
#TODO: stnid=^^****** = summary record
