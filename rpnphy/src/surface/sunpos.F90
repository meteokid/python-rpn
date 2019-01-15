!-------------------------------------- LICENCE BEGIN ------------------------------------
!Environment Canada - Atmospheric Science and Technology License/Disclaimer, 
!                     version 3; Last Modified: May 7, 2008.
!This is free but copyrighted software; you can use/redistribute/modify it under the terms 
!of the Environment Canada - Atmospheric Science and Technology License/Disclaimer 
!version 3 or (at your option) any later version that should be found at: 
!http://collaboration.cmc.ec.gc.ca/science/rpn.comm/license.html 
!
!This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
!without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
!See the above mentioned License/Disclaimer for more details.
!You should have received a copy of the License/Disclaimer along with this software; 
!if not, you can write to: EC-RPN COMM Group, 2121 TransCanada, suite 500, Dorval (Quebec), 
!CANADA, H9P 1J3; or send e-mail to service.rpn@ec.gc.ca
!-------------------------------------- LICENCE END --------------------------------------
!     ####################################################################################
      SUBROUTINE SUNPOS (KYEAR, KMONTH, KDAY, PTIME, PLON, PLAT, PTSUN, PZENITH, PAZIMSOL)
!     ####################################################################################
!
!!****  *SUNPOS * - routine to compute the position of the sun
!!
!!    PURPOSE
!!    -------
!!      The purpose of this routine is to compute the cosine and sinus of the 
!!    solar zenithal angle (angle defined by the local vertical at the position
!!    XLAT, XLON and the direction of the sun) and the azimuthal solar
!!    angle (angle between an horizontal direction (south or north according
!!    to the terrestrial hemisphere) and the horizontal projection of the
!!    direction of the sun.
!!
!!**  METHOD
!!    ------
!!      The cosine and sinus of the zenithal solar angle  and the azimuthal 
!!    solar angle are computed from the true universal time, valid for the (XLAT,
!!    XLON) location, and from the solar declination angle of the day. There
!!    is a special convention to define the azimuthal solar angle.
!!     
!!    EXTERNAL
!!    --------
!!      NONE
!!
!!    IMPLICIT ARGUMENTS
!!    ------------------
!!
!!    REFERENCE
!!    ---------
!!      "Radiative Processes in Meteorology and Climatology"  
!!                          (1976)   Paltridge and Platt 
!!
!!    AUTHOR
!!    ------
!!	J.-P. Pinty      * Laboratoire d'Aerologie*
!!
!!    MODIFICATIONS
!!    -------------
!!      Original             16/10/94 
!!      Revised              12/09/95
!!      (J.Stein)            01:04/96  bug correction for ZZEANG     
!!      (K. Suhre)           14/02/97  bug correction for ZLON0     
!!      (V. Masson)          01/03/03  add zenithal angle output
!-------------------------------------------------------------------------------
!
!*       0.    DECLARATIONS
!              ------------
!
USE MODD_CSTS,          ONLY : XPI, XDAY
!
implicit none
#include <arch_specific.hf>
!
!*       0.1   Declarations of dummy arguments :
!
INTEGER,                      INTENT(IN)   :: KYEAR      ! current year                        
INTEGER,                      INTENT(IN)   :: KMONTH     ! current month                        
INTEGER,                      INTENT(IN)   :: KDAY       ! current day                        
REAL,                         INTENT(IN)   :: PTIME      ! current time                        
REAL, DIMENSION(:),           INTENT(IN)   :: PLON       ! longitude
REAL, DIMENSION(:),           INTENT(IN)   :: PLAT       ! latutude
!
REAL, DIMENSION(:),           INTENT(OUT)  :: PZENITH    ! Solar zenithal angle
REAL, DIMENSION(:),           INTENT(OUT)  :: PAZIMSOL   ! Solar azimuthal angle
REAL, DIMENSION(:),           INTENT(OUT)  :: PTSUN      ! Solar time
!
!*       0.2   declarations of local variables
!
!
REAL                                       :: ZTIME      ! Centered current time for radiation calculations
REAL                                       :: ZUT        ! Universal time
!
REAL, DIMENSION(SIZE(PLON))                :: ZTUT    ,&! True (absolute) Universal Time
                                              ZSOLANG ,&! Hourly solar angle
                                              ZSINAZI ,&! Sine of the solar azimuthal angle
                                              ZCOSAZI ,&! Cosine of the solar azimuthal angle
                                              ZLAT,    &
                                              ZLON,    &! Array of latitudes and longitudes
                                              ZSINZEN, &!Sine of zenithal angle
                                              ZCOSZEN, &!Cosine of zenithal angle
                                              ZAZIMSOL,&!azimuthal angle
                                              ZTSIDER, &!
                                              ZSINDEL, &!azimuthal angle
                                              ZCOSDEL  !azimuthal angle

INTEGER, DIMENSION(0:11)                   :: IBIS, INOBIS ! Cumulative number of days per month
                                                           ! for bissextile and regular years
REAL                                       :: ZDATE         ! Julian day of the year
REAL                                       :: ZAD           ! Angular Julian day of the year
REAL                                       :: ZDECSOL       ! Daily solar declination angle 
REAL                                       :: ZA1, ZA2      ! Ancillary variables

INTEGER                                    :: JI
!
!-------------------------------------------------------------------------------
!
!*       1.    LOADS THE ZLAT, ZLON ARRAYS
!              ---------------------------
!
ZLAT = PLAT*(XPI/180.)
ZLON = PLON*(XPI/180.)
!
!-------------------------------------------------------------------------------
!
!*       2.    COMPUTES THE TRUE SOLAR TIME
!              ----------------------------
!
ZUT  = AMOD( 24.0+AMOD(PTIME/3600.,24.0),24.0 )

INOBIS(:) = (/0,31,59,90,120,151,181,212,243,273,304,334/)
IBIS(0) = INOBIS(0)
DO JI=1,11
  IBIS(JI) = INOBIS(JI)+1
END DO
IF( MOD(KYEAR,4).EQ.0 ) THEN
  ZDATE = FLOAT(KDAY +   IBIS(KMONTH-1)) - 1
  ZAD = 2.0*XPI*ZDATE/366.0
ELSE
  ZDATE = FLOAT(KDAY + INOBIS(KMONTH-1)) - 1
  ZAD = 2.0*XPI*ZDATE/365.0
END IF

ZA1 = (1.00554*ZDATE- 6.28306)*(XPI/180.0)
ZA2 = (1.93946*ZDATE+23.35089)*(XPI/180.0)
ZTSIDER = (7.67825*SIN(ZA1)+10.09176*SIN(ZA2)) / 60.0
!
ZTUT = ZUT - ZTSIDER + ZLON(:)*((180./XPI)/15.0)
!
PTSUN = AMOD(PTIME -ZTSIDER*3600. +PLON*240., XDAY)
!-------------------------------------------------------------------------------
!
!*       3.     COMPUTES THE SOLAR DECLINATION ANGLE
!	        ------------------------------------
!
ZDECSOL = 0.006918-0.399912*COS(ZAD)   +0.070257*SIN(ZAD)    &
         -0.006758*COS(2.*ZAD)+0.000907*SIN(2.*ZAD) &
         -0.002697*COS(3.*ZAD)+0.00148 *SIN(3.*ZAD)
ZSINDEL = SIN(ZDECSOL)
ZCOSDEL = COS(ZDECSOL)
!-------------------------------------------------------------------------------
!
!*       3.    COMPUTES THE COSINE AND SINUS OF THE ZENITHAL SOLAR ANGLE
!              ---------------------------------------------------------
!
ZSOLANG = (ZTUT-12.0)*15.0*(XPI/180.)          ! hour angle in radians
!
ZCOSZEN = SIN(ZLAT)*ZSINDEL +                 &! Cosine of the zenithal
               COS(ZLAT)*ZCOSDEL*COS(ZSOLANG)  !       solar angle
!
ZSINZEN  = SQRT( 1. - ZCOSZEN*ZCOSZEN )
!
!-------------------------------------------------------------------------------
!
!*       4.    ZENITHAL SOLAR ANGLE
!              --------------------
!
PZENITH = ACOS(ZCOSZEN)
!
!-------------------------------------------------------------------------------
!
!*       5.    COMPUTE THE AZINUTHAL SOLAR ANGLE (PAZIMSOL)
!              --------------------------------------------
!
WHERE (ZSINZEN/=0.)
  ZSINAZI  = - ZCOSDEL * SIN(ZSOLANG) / ZSINZEN
  ZCOSAZI  = (-SIN(ZLAT)*ZCOSDEL*COS(ZSOLANG)     & 
                   +COS(ZLAT)*ZSINDEL                       &
                  ) / ZSINZEN
  PAZIMSOL = XPI - ATAN2(ZSINAZI,ZCOSAZI)
ELSEWHERE
  PAZIMSOL = 0.
END WHERE
!
!-------------------------------------------------------------------------------
!
END SUBROUTINE SUNPOS
