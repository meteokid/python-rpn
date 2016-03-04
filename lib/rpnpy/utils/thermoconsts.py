#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Copyright: LGPL 2.1

"""
 Module utils.thermoconsts defines constants used in RPN Physics
"""
from math import pi as _pi

AI     = 0.2864887713087e+04  #! pour fn htvocp
AW     = 0.3135012829948e+04  #! pour fn htvocp
BI     = 0.1660931315020e+00  #! pour fn htvocp
BW     = 0.2367075766316e+01  #! pour fn htvocp
CAPPA  = 0.2854912179500e+00  #! rgasd/cpd
CHLF   = 0.3340000000000e+06  #! ch. lat. fusion       J kg-1
CHLC   = 0.2501000000000e+07  #! ch. lat. condens.(0C) J kg-1
CONSOL = 0.1367000000000e+04  #! constante solaire     W m-2
CPD    = 0.1005460000000e+04  #! chal. spec. air sec   J kg-1 K-1
CPV    = 0.1869460000000e+04  #! chal. spec. vap eau   J kg-1 K-1
CPI    = 0.2115300000000e+04  #! chal. spec. glace     J kg-1 K-1
DELTA  = 0.6077686814144e+00  #! 1/eps1 - 1
EPS1   = 0.6219800221014e+00  #! rgasd/rgasv
EPS2   = 0.3780199778986e+00  #! 1 - eps1
GRAV   = 0.9806160000000e+01  #! acc. de gravite       m s-2
KARMAN = 0.4000000000000e+00  #! cte de von karman
KNAMS  = 0.5147910000000e+00  #! passage kt a m/s
OMEGA  = 0.7292000000000e-04  #! rotation terre        s-1
PI     = 0.3141592653590e+01  #! cte pi=acos(-1)
RAUW   = 0.1000000000000e+04  #! densite eau liq       kg m-3
RAYT   = 0.6371220000000e+07  #! rayon moy. terre      m
RGASD  = 0.2870500000000e+03  #! cte gaz - air sec     J kg-1 K-1
RGASV  = 0.4615100000000e+03  #! cte gaz - vap eau     J kg-1 K-1
RIC    = 0.2000000000000e+00  #! cte richardson crit.
SLP    = 0.6666666666667e-01  #! pour fn htvocp
STEFAN = 0.5669800000000e-07  #! cte stefan-boltzmann  J m-2 s-1 K-4
STLO   = 0.6628486583943e-03  #! schuman-newell l.r.   K s2 m-2
T1S    = 0.2731600000000e+03  #! pour fn htvocp        K
T2S    = 0.2581600000000e+03  #! pour fn htvocp        K
TCDK   = 0.2731500000000e+03  #! passage k a c         C
TGL    = 0.2731600000000e+03  #! temp glace dans atm   K
TRPL   = 0.2731600000000e+03  #! point triple - eau    K

TTNS1  = 610.78
TTNS3W = 17.269
TTNS3I = 21.875
TTNS4W = 35.86
TTNS4I =  7.66

DEG2RAD = _pi/180.  #! Number of radian per degree [rad]
KT2MS   = 0.514791  #! Number of m/s per knot (0.51477333) [m/s]
MB2PA   = 100.      #! Number of Pascal per millibar [Pa]
MS2KT   = 1./KT2MS  #! Number of knot per m/2 [kt]
PA2MB   = 0.01      #! Number of millibar per Pascal [mb]
RAD2DEG = 180./_pi  #! Number of degree per radian [deg]
