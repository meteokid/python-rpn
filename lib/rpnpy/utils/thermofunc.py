#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: N. Brunet - mai 90
# Author: Stephane Chamberland <stephane.chamberland@canada.ca>
# Revisions:
## !          *** n. brunet - mai 90 ***
## !          * revision 01 - mai 94 - n. brunet
## !                          nouvelle version pour faibles pressions
## !          * revision 02 - aout 2000 - j-p toviessi
## !                          calcul en real*8
## !          * revision 03 - sept 2000 - n. brunet
## !                          ajout de nouvelles fonctions
## !          * revision 04 - janv 2000 - j. mailhot
## !                          fonctions en phase mixte
## !          * revision 05 - dec 2001 - g. lemay
## !                          double precision pour phase mixte
## !          * revision 06 - avr 2002 - a. plante
## !                          ajout des nouvelles fonctions fottvh et fotvht
# Copyright: LGPL 2.1

"""
Module utils.thermofunc defines basic thermodynamic functions used in RPNPhy

Note: All functions uses SI units (e.g. ttt  [k], prs [pa], qqq [kg/kg])

* FOEW(ttt): calcule la tension de vapeur saturante (formule de Tetens); on obtient ew (tension de vapeur par rapport à l’eau) ou ei (tension de vapeur par rapport à la glace) selon la température.
* FODLE(ttt): calcule la dérivée selon la température du ln ew ou ln ei

* FOQST(ttt, prs): calcule l’humidité spécifique saturante (qst) à partir de la température et de la pression.
* FODQS(qst, ttt): calcule la dérivée de qst selon la température. qst est la sortie de FOQST
* FOEFQ(qqq, prs): calcule la tension de vapeur à partir de l’humidité spécifique et de la pression.
* FOQFE(eee, prs): calcule l’humidité spécifique à partir de la tension de vapeur et de la pression.
* FOTVT(ttt, qqq): calcule la température virtuelle (tvi) à partir de la température et de l’humidité spécifique.
* FOTVHT(ttt,qqq,qqh): calcule la temp virt. (tvi) de temp (ttt), hum sp (qqq) et masse sp des hydrometeores.
* FOTTV(tvi, qqq): calcule la température à partir de la température virtuelle et de l’humidité spécifique.
* FOTTVH(tvi,qqq,qqh): calcule ttt de temp virt. (tvi), hum sp (qqq) et masse sp des hydrometeores (qqh)
* FOHR(qqq, ttt, prs): calcule l’humidité relative à partir de l’humidité spécifique, de la température et de la pression. Le résultat est en fraction (et non en pourcentage).
* FOLV(ttt): calcule la chaleur latente de condensation, fonction de la température (J/kg).
* FOLS(ttt): calcule la chaleur latente de sublimation, fonction de la température (J/kg).
* FOPOIT(t0, p0, pf): résout l’équation de Poisson pour la température; si pf=100000 pa, on obtient le theta standard (K).
* FOPOIP(t0, tf, p0): résout l’équation de Poisson pour la pression (pa).

Les cinq fonctions internes suivantes sont valides dans le contexte où on ne désire pas tenir compte de la
phase glace dans les calculs de saturation, c’est-à-dire que la phase eau seulement est considérée quelque
soit la température.

* FOEWA(ttt): tension de vapeur saturante, ew.
* FODLA(ttt): dérivée du ln ew selon la température.
* FOQSA(ttt, prs): humidité spécifique saturante. 11 janvier, 2001 3
* FODQA(qst, prs): la dérivée de FOQSA selon la température.
* FOHRA(qqq, ttt, prs): l’humidité relative (résultat en fraction et non en pourcentage)

Definition of basic thermodynamic functions in mixed-phase mode
fff is the fraction of ice and ddff its derivative w/r to t
* FESI(ttt): saturation calculations in presence of liquid phase only function for saturation vapor pressure (tetens)
* ...

Dans toutes les fonctions internes ci-dessus, on a le symbolisme suivant:
* eee : la tension de vapeur en pa
* prs ou p0 (p zéro) ou pf : la pression en pa.
* qqq : l’humidité spécifique en kg/kg
* qst : l’humidité spécifique saturante en kg/kg
* tvi : la température virtuelle en deg K
* ttt ou t0 (t zér0) ou tf: la température en deg K

Reference:
https://wiki.cmc.ec.gc.ca/images/9/9e/RPNPhy_Thermodynamic_functions_brunet.pdf
https://wiki.cmc.ec.gc.ca/images/4/4c/RPNPhy_Thermodynamic_functions.pdf
"""
import math as _m
from rpnpy.utils.thermoconsts import *

DSIGN  = lambda V,S: V if S>=0. else -V
DABS   = lambda X: abs(X)
DMIN1  = lambda X,Y: min(X,Y)
DMAX1  = lambda X,Y: max(X,Y)
DLOG   = lambda X: _m.log(X)
DEXP   = lambda X: _m.exp(X)
DBLE   = lambda X: X

# fonction de tension de vapeur saturante (tetens) - ew ou ei selon tt
FOEWF  = lambda TTT: DMIN1(DSIGN(TTNS3W,DBLE(TTT)-DBLE(TRPL)),DSIGN(TTNS3I,DBLE(TTT)-DBLE(TRPL)))*DABS(DBLE(TTT)-DBLE(TRPL))/(DBLE(TTT)-TTNS4W+DMAX1(0.,DSIGN(TTNS4W-TTNS4I,DBLE(TRPL)-DBLE(TTT))))
FOMULT = lambda DDD: TTNS1*DDD
FOEW   = lambda TTT: FOMULT(DEXP(FOEWF(TTT)))

FODLE  = lambda TTT: (4097.93+DMAX1(0.,DSIGN(1709.88,DBLE(TRPL)-DBLE(TTT))))/((DBLE(TTT)-TTNS4W+DMAX1(0.,DSIGN(TTNS4W-TTNS4I,DBLE(TRPL)-DBLE(TTT))))*(DBLE(TTT)-TTNS4W+DMAX1(0.,DSIGN(TTNS4W-TTNS4I,DBLE(TRPL)-DBLE(TTT)))))
FOQST  = lambda TTT,PRS: DBLE(EPS1)/(DMAX1(1.,DBLE(PRS)/FOEW(TTT))-DBLE(EPS2))
FOQSTX = lambda PRS,DDD: DBLE(EPS1)/(DMAX1(1.,DBLE(PRS)/DDD)-DBLE(EPS2))
FODQS  = lambda QST,TTT: DBLE(QST)*(1.+DBLE(DELTA)*DBLE(QST))*FODLE(TTT)
FOEFQ  = lambda QQQ,PRS: DMIN1(DBLE(PRS),(DBLE(QQQ)*DBLE(PRS))/(DBLE(EPS1)+DBLE(EPS2)*DBLE(QQQ)))
FOQFE  = lambda EEE,PRS: DMIN1(1.,DBLE(EPS1)*DBLE(EEE)/(DBLE(PRS)-DBLE(EPS2)*DBLE(EEE)))
FOTVT  = lambda TTT,QQQ: DBLE(TTT)*(1.0+DBLE(DELTA)*DBLE(QQQ))
FOTVHT = lambda TTT,QQQ,QQH: DBLE(TTT)*(1.0+DBLE(DELTA)*DBLE(QQQ)-DBLE(QQH))
FOTTV  = lambda TVI,QQQ: DBLE(TVI)/(1.0+DBLE(DELTA)*DBLE(QQQ))
FOTTVH = lambda TVI,QQQ,QQH: DBLE(TVI)/(1.0+DBLE(DELTA)*DBLE(QQQ)-DBLE(QQH))
FOHR   = lambda QQQ,TTT,PRS: DMIN1(DBLE(PRS),FOEFQ(QQQ,PRS))/FOEW(TTT)
FOHRX  = lambda QQQ,PRS,DDD: DMIN1(DBLE(PRS),FOEFQ(QQQ,PRS))/DDD
FOLV   = lambda TTT: DBLE(CHLC)-2317.*(DBLE(TTT)-DBLE(TRPL))
FOLS   = lambda TTT: DBLE(CHLC)+DBLE(CHLF)+(DBLE(CPV)-(7.24*DBLE(TTT)+128.4))*(DBLE(TTT)-DBLE(TRPL))
FOPOIT = lambda T00,PR0,PF: DBLE(T00)*(DBLE(PR0)/DBLE(PF))**(-DBLE(CAPPA))
FOPOIP = lambda T00,TF,PR0: DBLE(PR0)*DEXP(-(DLOG(DBLE(T00)/DBLE(TF))/DBLE(CAPPA)))
FOEWAF = lambda TTT: TTNS3W*(DBLE(TTT)-DBLE(TRPL))/(DBLE(TTT)-TTNS4W)
FOEWA  = lambda TTT: FOMULT(DEXP(FOEWAF(TTT)))
FODLA  = lambda TTT: TTNS3W*(DBLE(TRPL)-TTNS4W)/(DBLE(TTT)-TTNS4W)**2
FOQSA  = lambda TTT,PRS: DBLE(EPS1)/(DMAX1(1.,DBLE(PRS)/FOEWA(TTT))-DBLE(EPS2))
FODQA  = lambda QST,TTT: DBLE(QST)*(1.+DBLE(DELTA)*DBLE(QST))*FODLA(TTT)
FOHRA  = lambda QQQ,TTT,PRS: DMIN1(DBLE(PRS),FOEFQ(QQQ,PRS))/FOEWA(TTT)
FESIF  = lambda TTT: TTNS3I*(DBLE(TTT)-DBLE(TRPL))/(DBLE(TTT)-TTNS4I)
FESI   = lambda TTT: FOMULT(DEXP(FESIF(TTT)))
FDLESI = lambda TTT: TTNS3I*(DBLE(TRPL)-TTNS4I)/(DBLE(TTT)-TTNS4I)**2
FESMX  = lambda TTT,FFF: (1.-DBLE(FFF))*FOEWA(TTT)+DBLE(FFF)*FESI(TTT)
FESMXX = lambda FFF,FESI8,FOEWA8: (1.-DBLE(FFF))*FOEWA8+DBLE(FFF)*FESI8
FDLESMX = lambda TTT,FFF,DDFF: ((1.-DBLE(FFF))*FOEWA(TTT)*FODLA(TTT)+DBLE(FFF)*FESI(TTT)*FDLESI(TTT)+DBLE(DDFF)*(FESI(TTT)-FOEWA(TTT)))/FESMX(TTT,FFF)
FDLESMXX = lambda TTT,FFF,DDFF,FOEWA8,FESI8,FESMX8: ((1.-DBLE(FFF))*FOEWA8*FODLA(TTT)+DBLE(FFF)*FESI8*FDLESI(TTT)+DBLE(DDFF)*(FESI8-FOEWA8))/FESMX8
FQSMX  = lambda TTT,PRS,FFF: DBLE(EPS1)/(DMAX1(1.,DBLE(PRS)/FESMX(TTT,FFF))-DBLE(EPS2))
FQSMXX = lambda FESMX8,PRS: DBLE(EPS1)/(DMAX1(1.,DBLE(PRS)/FESMX8)-DBLE(EPS2))
FDQSMX = lambda QSM,DLEMX: DBLE(QSM)*(1.+DBLE(DELTA)*DBLE(QSM))*DBLE(DLEMX)
