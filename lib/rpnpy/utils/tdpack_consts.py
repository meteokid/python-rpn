#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
Module rpnpy.utils.tdpack_const defines constants used in RPN Physics
 
See Also:
    rpnpy.utils.tdpack

Details:
    AI     : pour fn htvocp
    AW     : pour fn htvocp
    BI     : pour fn htvocp
    BW     : pour fn htvocp
    CAPPA  : rgasd/cpd
    CHLF   : ch. lat. fusion [J kg-1]
    CHLC   : ch. lat. condens.(0C) [J kg-1]
    CONSOL : constante solaire (1367.0) [W m-2]
    CONSOL2: constante solaire (1361.0) (corrigee) [W m-2]
    CPD    : chal. spec. air sec [J kg-1 K-1]
    CPV    : chal. spec. vap eau [J kg-1 K-1]
    CPI    : chal. spec. glace [J kg-1 K-1]
    CPW    : chal. spec. eau liq. [J kg-1 K-1]
    DELTA  : 1/eps1 - 1
    EPS1   : rgasd/rgasv
    EPS2   : 1 - eps1
    GRAV        : acc. de gravite (9.80665) [m s-2]
    GRAV_LEGACY : acc. de gravite (9.80616) [m s-2]
    KARMAN : cte de von karman
    KNAMS  : passage kt a m/s
    OMEGA  : rotation terre [s-1]
    PI     : cte pi:acos(-1)
    RAUW   : densite eau liq [kg m-3]
    RAYT   : rayon moy. terre [m]
    RGASD  : cte gaz - air sec [J kg-1 K-1]
    RGASV  : cte gaz - vap eau [J kg-1 K-1]
    RIC    : cte richardson crit.
    SLP    : pour fn htvocp
    STEFAN        : cte stefan-boltzmann (5.670373e-08) [J m-2 s-1 K-4]
    STEFAN_LEGACY : cte stefan-boltzmann (5.669800e-08) [J m-2 s-1 K-4]
    STLO   : schuman-newell l.r. [K s2 m-2]
    T1S    : pour fn htvocp [K]
    T2S    : pour fn htvocp [K]
    TCDK   : passage k a c [C]
    TGL    : temp glace dans atm [K]
    TRPL   : point triple - eau [K]

    Tetens coefficients in saturation vapor pressure formulas
    TTNS1  :
    TTNS3W :
    TTNS3I :
    TTNS4W :
    TTNS4I :

    Alduchov and Eskridge (1995) coefficients in saturation vapor pressure formulas
    AERK1W : voir Alduchov and Eskridge (1995) -AERK
    AERK2W : voir Alduchov and Eskridge (1995) -AERK
    AERK3W : voir Alduchov and Eskridge (1995) -AERK
    AERK1I : voir Alduchov and Eskridge (1995) -AERKi
    AERK2I : voir Alduchov and Eskridge (1995) -AERKi
    AERK3I : voir Alduchov and Eskridge (1995) -AERKi

    Other constants:
    DEG2RAD      : Number of radian per degree [rad]
    KT2MS        : Number of m/s per knot (0.51444445) [m/s]
    KT2MS_LEGACY : Number of m/s per knot (0.51477333) [m/s]
    MB2PA        : Number of Pascal per millibar [Pa]
    MS2KT        : Number of knot per m/s [kt]
    MS2KT_LEGACY : Number of knot per m/s [kt]
    PA2MB        : Number of millibar per Pascal [mb]
    RAD2DEG      : Number of degree per radian [deg]

"""
from math import pi as _pi
import scipy.constants as _scipy_cst

AI = 0.2864887713087E+04
AW = 0.3135012829948E+04
BI = 0.1660931315020E+00
BW = 0.2367075766316E+01
CAPPA = 0.2854912179500E+00
CHLF = 0.3340000000000E+06
CHLC = 0.2501000000000E+07
CONSOL = 0.1367000000000E+04
CONSOL2 = 0.1361000000000E+04
CPD = 0.1005460000000E+04
CPV = 0.1869460000000E+04
CPI = 0.2115300000000E+04
CPW = 0.4218000000000E+04
DELTA = 0.6077686814144E+00
EPS1 = 0.6219800221014E+00
EPS2 = 0.3780199778986E+00
GRAV = 0.9806160000000E+01
KARMAN = 0.4000000000000E+00
KNAMS = 0.5147910000000E+00
OMEGA = 0.7292000000000E-04
PI = 0.3141592653590E+01
RAUW = 0.1000000000000E+04
RAYT = 0.6371220000000E+07
RGASD = 0.2870500000000E+03
RGASV = 0.4615100000000E+03
RIC = 0.2000000000000E+00
SLP = 0.6666666666667E-01
STEFAN = 0.5669800000000E-07
STLO = 0.6628486583943E-03
T1S = 0.2731600000000E+03
T2S = 0.2581600000000E+03
TCDK = 0.2731500000000E+03
TGL = 0.2731600000000E+03
TRPL = 0.2731600000000E+03
TTNS1 = 610.78
TTNS3W = 17.269
TTNS3I = 21.875
TTNS4W = 35.86
TTNS4I = 7.66
AERK1W = 610.94
AERK2W = 17.625
AERK3W = 30.11
AERK1I = 611.21
AERK2I = 22.587
AERK3I = -0.71

GRAV_LEGACY   = GRAV         # acc. de gravite       [m s-2]
GRAV          = _scipy_cst.g # acc. de gravite       [m s-2]
STEFAN_LEGACY = STEFAN
try:
    STEFAN = _scipy_cst.Stefan_Boltzmann
except:
    STEFAN = STEFAN_LEGACY

DEG2RAD = _pi/180.            # Number of radian per degree [rad]
KT2MS   = _scipy_cst.knot     # Number of m/s per knot (0.51444445) [m/s]
KT2MS_LEGACY  = 0.51477333    # Number of m/s per knot (0.51477333) [m/s]
MB2PA   = 100.                # Number of Pascal per millibar [Pa]
MS2KT   = 1./KT2MS            # Number of knot per m/s [kt]
MS2KT_LEGACY = 1./KT2MS_LEGACY # Number of knot per m/s [kt]
PA2MB   = 0.01                # Number of millibar per Pascal [mb]
RAD2DEG = 180./_pi            # Number of degree per radian [deg]

if __name__ == "__main__":
    import doctest
    doctest.testmod()


# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
