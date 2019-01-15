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
Module rpnpy.utils.tdpack defines basic thermodynamic functions used in RPNPhy

Note: All functions uses SI units (e.g. ttt  [k], prs [pa], qqq [kg/kg])

See Also:
    rpnpy.utils.tdpack_const

Details:

* FOEW(ttt):
    calcule la tension de vapeur saturante (formule de Tetens);
    on obtient ew (tension de vapeur par rapport à l’eau) ou ei
    (tension de vapeur par rapport à la glace) selon la température.
* FODLE(ttt):
    calcule la dérivée selon la température du ln ew ou ln ei
* FOQST(ttt, prs):
    calcule l’humidité spécifique saturante (qst) à partir de la
    température et de la pression.
* FODQS(qst, ttt):
    calcule la dérivée de qst selon la température. qst est la sortie de FOQST
* FOEFQ(qqq, prs):
    calcule la tension de vapeur à partir de l’humidité spécifique et
    de la pression.
* FOQFE(eee, prs):
    calcule l’humidité spécifique à partir de la tension de vapeur et
    de la pression.
* FOTVT(ttt, qqq):
    calcule la température virtuelle (tvi) à partir de la température et
    de l’humidité spécifique.
* FOTVHT(ttt,qqq,qqh):
    calcule la temp virt. (tvi) de temp (ttt), hum sp (qqq) et masse sp
    des hydrometeores.
* FOTTV(tvi, qqq):
    calcule la température à partir de la température virtuelle et
    de l’humidité spécifique.
* FOTTVH(tvi,qqq,qqh):
    calcule ttt de temp virt. (tvi), hum sp (qqq) et masse sp des
    hydrometeores (qqh)
* FOHR(qqq, ttt, prs):
    calcule l’humidité relative à partir de l’humidité spécifique,
    de la température et de la pression. Le résultat est en fraction
    (et non en pourcentage).
* FOLV(ttt):
    calcule la chaleur latente de condensation, fonction de la
    température (J/kg).
* FOLS(ttt):
    calcule la chaleur latente de sublimation, fonction de la température (J/kg)
* FOPOIT(t0, p0, pf):
    résout l’équation de Poisson pour la température; si pf=100000 pa,
    on obtient le theta standard (K).
* FOPOIP(t0, tf, p0):
    résout l’équation de Poisson pour la pression (pa).

Les cinq fonctions internes suivantes sont valides dans le contexte où
on ne désire pas tenir compte de la phase glace dans les calculs de saturation,
c’est-à-dire que la phase eau seulement est considérée quelque soit la température.

* FOEWA(ttt):
    tension de vapeur saturante, ew.
* FODLA(ttt):
    dérivée du ln ew selon la température.
* FOQSA(ttt, prs):
    humidité spécifique saturante. 11 janvier, 2001 3
* FODQA(qst, prs):
    la dérivée de FOQSA selon la température.
* FOHRA(qqq, ttt, prs):
    l’humidité relative (résultat en fraction et non en pourcentage)

Definition of basic thermodynamic functions in mixed-phase mode
fff is the fraction of ice and ddff its derivative w/r to t
* FESI(ttt):
    saturation calculations in presence of liquid phase only function for saturation vapor pressure (tetens)
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
import numpy as _np
from rpnpy.utils.tdpack_consts import *

_DSIGN  = lambda V,S: V * _np.sign(S)
_DABS   = lambda X: _np.absolute(X)
_DMIN1  = lambda X,Y: _np.minimum(X,Y)
_DMAX1  = lambda X,Y: _np.maximum(X,Y)
_DLOG   = lambda X: _np.log(X)
_DEXP   = lambda X: _np.exp(X)
_DBLE   = lambda X: _np.double(X)

# Fonction de tension de vapeur saturante (tetens) - ew ou ei selon tt
FOEWF  = lambda TTT: \
    _DMIN1(_DSIGN(TTNS3W,_DBLE(TTT)-_DBLE(TRPL)),
           _DSIGN(TTNS3I,_DBLE(TTT)-_DBLE(TRPL))) \
    * _DABS(_DBLE(TTT)-_DBLE(TRPL)) \
    / (_DBLE(TTT)-TTNS4W+_DMAX1(0.,_DSIGN(TTNS4W-TTNS4I,_DBLE(TRPL)-_DBLE(TTT))))
FOMULT = lambda DDD: \
    TTNS1*DDD
FOEW   = lambda TTT: \
    FOMULT(_DEXP(FOEWF(TTT)))

# Fonction calculant la derivee selon t de  ln ew (ou ln ei)
FODLE  = lambda TTT: \
    (4097.93+_DMAX1(0.,_DSIGN(1709.88,_DBLE(TRPL)-_DBLE(TTT)))) \
    / ((_DBLE(TTT)-TTNS4W+_DMAX1(0.,_DSIGN(TTNS4W-TTNS4I,_DBLE(TRPL)-_DBLE(TTT))))
       * (_DBLE(TTT)-TTNS4W+_DMAX1(0.,_DSIGN(TTNS4W-TTNS4I,_DBLE(TRPL)-_DBLE(TTT)))))

# Fonction calculant l'humidite specifique saturante (qsat)
FOQST  = lambda TTT,PRS: \
    _DBLE(EPS1)/(_DMAX1(1.,_DBLE(PRS)/FOEW(TTT))-_DBLE(EPS2))
FOQSTX = lambda PRS,DDD: \
    _DBLE(EPS1)/(_DMAX1(1.,_DBLE(PRS)/DDD)-_DBLE(EPS2))

# Fonction calculant la derivee de qsat selon t, qst est la sortie de foqst
FODQS  = lambda QST,TTT: \
    _DBLE(QST)*(1.+_DBLE(DELTA)*_DBLE(QST))*FODLE(TTT)

# Fonction calculant tension vap (eee) fn de hum sp (qqq) et prs
FOEFQ  = lambda QQQ,PRS: \
    _DMIN1(_DBLE(PRS),(_DBLE(QQQ)*_DBLE(PRS))/(_DBLE(EPS1)+_DBLE(EPS2)*_DBLE(QQQ)))

# Fonction calculant hum sp (qqq) de tens. vap (eee) et pres (prs)
FOQFE  = lambda EEE,PRS: \
    _DMIN1(1.,_DBLE(EPS1)*_DBLE(EEE)/(_DBLE(PRS)-_DBLE(EPS2)*_DBLE(EEE)))

# Fonction calculant temp virt. (tvi) de temp (ttt) et hum sp (qqq)
FOTVT  = lambda TTT,QQQ: \
    _DBLE(TTT)*(1.0+_DBLE(DELTA)*_DBLE(QQQ))

# Fonction calculant temp virt. (tvi) de temp (ttt), hum sp (qqq) et masse sp des hydrometeores.
FOTVHT = lambda TTT,QQQ,QQH: \
    _DBLE(TTT)*(1.0+_DBLE(DELTA)*_DBLE(QQQ)-_DBLE(QQH))

# Fonction calculant ttt de temp virt. (tvi) et hum sp (qqq)
FOTTV  = lambda TVI,QQQ: \
    _DBLE(TVI)/(1.0+_DBLE(DELTA)*_DBLE(QQQ))

# Fonction calculant ttt de temp virt. (tvi), hum sp (qqq) et masse sp des hydrometeores (qqh)
FOTTVH = lambda TVI,QQQ,QQH: \
    _DBLE(TVI)/(1.0+_DBLE(DELTA)*_DBLE(QQQ)-_DBLE(QQH))

# Fonction calculant hum rel de hum sp (qqq), temp (ttt) et pres (prs), hr = e/esat
FOHR   = lambda QQQ,TTT,PRS: \
    _DMIN1(_DBLE(PRS),FOEFQ(QQQ,PRS))/FOEW(TTT)
FOHRX  = lambda QQQ,PRS,DDD: \
    _DMIN1(_DBLE(PRS),FOEFQ(QQQ,PRS))/DDD

# Fonction calculant la chaleur latente de condensation
FOLV   = lambda TTT: \
    _DBLE(CHLC)-2317.*(_DBLE(TTT)-_DBLE(TRPL))

# Fonction calculant la chaleur latente de sublimation
FOLS   = lambda TTT: \
    _DBLE(CHLC)+_DBLE(CHLF)+(_DBLE(CPV)-(7.24*_DBLE(TTT)+128.4))*(_DBLE(TTT)-_DBLE(TRPL))

# Fonction resolvant l'eqn. de poisson pour la temperature; note: si pf=1000*100, "fopoit" donne le theta standard
FOPOIT = lambda T00,PR0,PF: \
    _DBLE(T00)*(_DBLE(PR0)/_DBLE(PF))**(-_DBLE(CAPPA))

# Fonction resolvant l'eqn. de poisson pour la pression
FOPOIP = lambda T00,TF,PR0: \
    _DBLE(PR0)*_DEXP(-(_DLOG(_DBLE(T00)/_DBLE(TF))/_DBLE(CAPPA)))

# Fonction de vapeur saturante (tetens) (No ice)
FOEWAF = lambda TTT: \
    TTNS3W*(_DBLE(TTT)-_DBLE(TRPL))/(_DBLE(TTT)-TTNS4W)
FOEWA  = lambda TTT: \
    FOMULT(_DEXP(FOEWAF(TTT)))

# Fonction calculant la derivee selon t de ln ew (No ice)
FODLA  = lambda TTT: \
    TTNS3W*(_DBLE(TRPL)-TTNS4W)/(_DBLE(TTT)-TTNS4W)**2

# Fonction calculant l'humidite specifique saturante (No ice)
FOQSA  = lambda TTT,PRS: \
    _DBLE(EPS1)/(_DMAX1(1.,_DBLE(PRS)/FOEWA(TTT))-_DBLE(EPS2))

# Fonction calculant la derivee de qsat selon t (No ice)
FODQA  = lambda QST,TTT: \
    _DBLE(QST)*(1.+_DBLE(DELTA)*_DBLE(QST))*FODLA(TTT)

# Fonction calculant l'humidite relative (No ice)
FOHRA  = lambda QQQ,TTT,PRS: \
    _DMIN1(_DBLE(PRS),FOEFQ(QQQ,PRS))/FOEWA(TTT)

# saturation calculations in presence of liquid phase only, function for saturation vapor pressure (tetens) (mixed-phase mode)
FESIF  = lambda TTT: \
    TTNS3I*(_DBLE(TTT)-_DBLE(TRPL))/(_DBLE(TTT)-TTNS4I)
FESI   = lambda TTT: \
    FOMULT(_DEXP(FESIF(TTT)))
FDLESI = lambda TTT: \
    TTNS3I*(_DBLE(TRPL)-TTNS4I)/(_DBLE(TTT)-TTNS4I)**2
FESMX  = lambda TTT,FFF: \
    (1.-_DBLE(FFF))*FOEWA(TTT)+_DBLE(FFF)*FESI(TTT)
FESMXX = lambda FFF,FESI8,FOEWA8:\
    (1.-_DBLE(FFF))*FOEWA8+_DBLE(FFF)*FESI8
FDLESMX  = lambda TTT,FFF,DDFF: \
    ((1.-_DBLE(FFF))*FOEWA(TTT)*FODLA(TTT)+_DBLE(FFF)*FESI(TTT)*FDLESI(TTT)+_DBLE(DDFF)*(FESI(TTT)-FOEWA(TTT)))/FESMX(TTT,FFF)
FDLESMXX = lambda TTT,FFF,DDFF,FOEWA8,FESI8,FESMX8: \
    ((1.-_DBLE(FFF))*FOEWA8*FODLA(TTT)+_DBLE(FFF)*FESI8*FDLESI(TTT)+_DBLE(DDFF)*(FESI8-FOEWA8))/FESMX8
FQSMX  = lambda TTT,PRS,FFF: \
    _DBLE(EPS1)/(_DMAX1(1.,_DBLE(PRS)/FESMX(TTT,FFF))-_DBLE(EPS2))
FQSMXX = lambda FESMX8,PRS: \
    _DBLE(EPS1)/(_DMAX1(1.,_DBLE(PRS)/FESMX8)-_DBLE(EPS2))
FDQSMX = lambda QSM,DLEMX: \
    _DBLE(QSM)*(1.+_DBLE(DELTA)*_DBLE(QSM))*_DBLE(DLEMX)


if __name__ == "__main__":
    import doctest
    doctest.testmod()


# -*- Mode: C; tab-width: 4; indent-tabs-mode: nil -*-
# vim: set expandtab ts=4 sw=4:
# kate: space-indent on; indent-mode cstyle; indent-width 4; mixedindent off;
