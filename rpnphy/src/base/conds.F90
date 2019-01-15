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
!**S/P CONDS
!
      SUBROUTINE CONDS(TE,QE,SRR,SSR,FN, &
                     TP1,QP1,PSP1,KBL, &
                     SIGMA, TAU, N, NI, NK, DBGCOND, SATUCO)
      implicit none
#include <arch_specific.hf>
!
      LOGICAL LO,DBGCOND
      LOGICAL SATUCO
      INTEGER IERR
      REAL TAU
!
!
      INTEGER N,NI,NK
      REAL TEMP1, TEMP2, TEMP3
!
      REAL TP1(N,NK),QP1(N,NK), &
                PSP1(N),KBL(N),SIGMA(NI,NK)
      REAL TE(NI,NK),QE(NI,NK),SRR(NI),SSR(NI)
      REAL FN(NI,NK)
      INTEGER NIP,NIKP,NKP1,NKM1,JL,JK,IS
!
!Author
!          J.Mailhot 11/03/85.  Adapted from E.C.M.W.F.
!
!Revision
! 001      J. Mailhot (Mar 1987) base of stable condensation
! 002      G.Pellerin(Nov87)Adaptation to revised code
!                     (Mar88)Standard documentations
! 003      J. Mailhot (Mar 1988) threshold of evaporation
! 004      J P Toviessi(May1990)Conversion in CFT77
! 005      G.Pellerin(August90)Standardization of thermo functions
! 006      N. Brunet  (May91)
!                 New version of thermodynamic functions
!                 and file of constants
! 007      B. Bilodeau  (August 1991)- Adaptation to UNIX
!
! 008      J. Mailhot (Dec.1992) - Newton Method Bug Correction
!            (ref. Revision 005)
! 009      C. Girard (Nov.1992) - Small clean-up, more correction
!          to the thickness of the 1st layer and "implicit"
!          evaporation of the precipitation
! 010      A. Methot (Aug 93) - L/Cp added in calculation of evap.
! 011      G. Lemay (Oct 93) - Dynamic memory allocation with stkmemw
! 012      R. Benoit (Aug 93) - Local Sigma
! 013      B. Bilodeau (June 94) - New physics interface
! 014      B. Bilodeau (Jan  01) - Automatic arrays
! 015      M. Lepine  (March 2003) -  CVMG... Replacements
! 016      L. Spacek (June 2003) - IBM conversion
!                                - boucle 331 defectuosite
!
!Object
!          to calculate the T and Q tendencies due to large scale
!          precipitation
!
!Arguments
!
!          - Output -
! TE       temperature tendency due to stratiform processes
! QE       specific humidity tendency due to stratiform processes
!
!          - Output -
! SRR      rate of liquid precipitation
! SSR      rate of solid precipitation
! FN       cloud fraction
!
!          - Input -
! TP1      temperature
! QP1      specific humidity
! PSP1     surface pressure
! KBL      index of first level in boundary layer
! SIGMA    sigma levels
! TAU      FACTDT * timestep (see common bloc OPTIONS)
! N        dimension of some arrays
! NI       1st horizontal dimension
! NK       vertical dimension
! DBGCOND  .TRUE. to have debugging for condensation
!          .FALSE. to have no debugging
! SATUCO   .TRUE. to have water/ice phase for saturation
!          .FALSE. to have water phase only for saturation
!
!Notes
!          During the process, the variables T and Q at (T+DT) are
!          adjusted instantly. There is no storage of water or snow
!          in the cloud. In super-saturated layers, Q is restored
!          to Q* and the difference between Q and Q* is added to
!          the fluxes of rain and snow from the top to the bottom
!          of the layer. The evaporation, condensation or melting
!          can affect the divergence of the precipitation fluxes.
!          Reference in ECMWF Research Manual (Vol.3)
!          Physical Parameterization Chapter 5)
!
!*
!
!***********************************************************************
!     AUTOMATIC ARRAYS
!***********************************************************************
!
      INTEGER, dimension(NI   ) :: IQCD
!
      LOGICAL, dimension(NI   ) :: LO1
!
      REAL, dimension(NI,NK) :: ZPP1
      REAL, dimension(NI,NK) :: ZDSG
      REAL, dimension(NI,NK) :: ZDPP1
      REAL, dimension(NI,NK) :: ZTP1
      REAL, dimension(NI,NK) :: ZQP1
      REAL, dimension(NI,NK) :: ZQSATE
      REAL, dimension(NI,NK) :: ZTC
      REAL, dimension(NI,NK) :: ZQC
      REAL, dimension(NI   ) :: ZRFL
      REAL, dimension(NI   ) :: ZSFL
      REAL, dimension(NI   ) :: ZRFLN
      REAL, dimension(NI   ) :: ZSFLN
      REAL, dimension(NI   ) :: ZFLN
!
!***********************************************************************
!
      REAL ZSTPRO,ZDIP,ZEVAP,ZMELT,ZSQFLN,ZNIMP
      REAL ZEPFLM,ZEPFLS,ZCONS1,ZCONS2
      REAL ZTMST,ZLDCPE,ZQCD
      REAL ZQSATC,ZCOR,ZRITO,ZLVDCP,ZLSDCP,ZRIT
!
!
!*    PHYSICAL CONSTANTS.
!     -------- ----------
#include "comphy.cdk"
!
      REAL CHLS
include "thermoconsts.inc"
include "dintern.inc"
include "fintern.inc"
!
!     -------------
!
!     *ZEVAP*IS A CONSTANT FOR THE EVAPORATION OF
!     TOTAL PRECIPITATION, *ZMELT* IS THE CONSTANT OF THE FORMULA FOR
!     THE RATE OF CHANGE OF THE LIQUID WATER/ICE COMPOSITION OF THESE
!     PRECIPITATIONS.
!
      NKP1=NK+1
      NKM1=NK-1
      ZEVAP=CEVAP
      ZMELT=CMELT
!
!*    SECURITY PARAMETERS.
!     --------------------
!
!         *ZEPFLM* IS A MINIMUM FLUX TO AVOID DIVIDING BY ZERO IN THE IC
!     PROPORTION CALCULATIONS.
!
      ZEPFLM=1.E-24
      ZEPFLS=SQRT(ZEPFLM)
!
!*    COMPUTATIONAL CONSTANTS.
!     ------------- ----------
!
      ZTMST= TAU
      CHLS = CHLC + CHLF
!
      ZCONS1=CPV - CPD
      ZCONS2 = 1./(ZTMST*GRAV)
!
!
!     ------------------------------------------------------------------
!
!
!*         1.     PRELIMINARY COMPUTATIONS.
!                 ----------- -------------
!
  200 CONTINUE
!
!*         1.1     INITIAL VALUES FOR ACCUMULATION
!
  210 CONTINUE
!
      DO 211 JL=1,NI
      ZRFL(JL)=0.
      ZSFL(JL)=0.
      ZFLN(JL)=0.
  211 CONTINUE
!
!
!
!     ------------------------------------------------------------------
!
!*         2.     CLOUD VARIABLES, RAIN/SNOW FLUXES.
!                 ----- ---------- --------- -------
!
  300 CONTINUE
!
!*         2.1     T+1 T,Q VARIABLES AND SATURATION MIXING RATIO.
!
  310 CONTINUE
      DO 3150 JL=1,NI
      ZDSG(JL,1)=0.5*(SIGMA(JL,2)-SIGMA(JL,1))
      DO 315 JK=2,NKM1
      ZDSG(JL,JK)=0.5*(SIGMA(JL,JK+1)-SIGMA(JL,JK-1))
  315 CONTINUE
      ZDSG(JL,NK)=0.5*(1.-SIGMA(JL,NKM1))+0.5*(1.-SIGMA(JL,NK))
 3150 continue
!
      DO 311 JK=1,NK
      DO 311 JL=1,NI
      ZPP1(JL,JK)=SIGMA(JL,JK)*PSP1(JL)
      ZTP1(JL,JK)=TP1(JL,JK)
      ZQP1(JL,JK)=QP1(JL,JK)
      ZDPP1(JL,JK)=ZDSG(JL,JK)*PSP1(JL)
  311 CONTINUE
      IF(SATUCO)THEN
      DO 312 JK=1,NK
      DO 312 JL=1,NI
      TEMP1 = ZTP1(JL,JK)
      TEMP2 = ZPP1(JL,JK)
  312 ZQSATE(JL,JK)=FOQST(TEMP1,TEMP2)
      ELSE
      DO 314 JK=1,NK
      DO 314 JL=1,NI
      TEMP1= ZTP1(JL,JK)
      TEMP2= ZPP1(JL,JK)
  314 ZQSATE(JL,JK)=FOQSA(TEMP1,TEMP2)
      ENDIF
!
!
!*         2.2     CALCULATE TC AND QC IN SUPERSATURATED LAYERS. THE
!*                 CONDENSATION CALCULATIONS ARE DONE WITH TWO ITERATION
!
  320 CONTINUE
!
!***
      DO 329 JK=1,NK
!***
      DO 322 JL=1,NI
      ZTC(JL,JK)=ZTP1(JL,JK)
      ZQC(JL,JK)=ZQP1(JL,JK)
  322 CONTINUE
      IF(SATUCO)THEN
      DO 323 JL=1,NI
      ZQSATC=ZQSATE(JL,JK)
!      ZLDCPE = CVMGT(CHLC,CHLS,ZTC(JL,JK)-TCDK .GT. 0.)
!     *      /(CPD+ZCONS1*ZQC(JL,JK))
      if (ZTC(JL,JK)-TCDK .GT. 0.) then
         ZLDCPE = CHLC /(CPD+ZCONS1*ZQC(JL,JK))
      else
         ZLDCPE = CHLS /(CPD+ZCONS1*ZQC(JL,JK))
      endif
      TEMP1 = ZTC(JL,JK)
      ZCOR=ZLDCPE*FODQS(ZQSATC,TEMP1)
      ZQCD=AMAX1(0.,(ZQC(JL,JK)-ZQSATC)/(1.+ZCOR))
      LO=ZQCD.EQ.0.
!      IQCD(JL) = CVMGT(0.,1.,LO)
      if (ZQCD.EQ.0.) then
         IQCD(JL) = 0.
      else
         IQCD(JL) = 1.
      endif
      ZQC(JL,JK)=ZQC(JL,JK)-ZQCD
      ZTC(JL,JK)=ZTC(JL,JK)+ZQCD*ZLDCPE
  323 CONTINUE
      ELSE
      DO 327 JL=1,NI
      ZQSATC=ZQSATE(JL,JK)
!      ZLDCPE= CVMGT(CHLC, CHLS,  ZTC(JL,JK)-TCDK .GT. 0.)
!     *      /(CPD+ZCONS1*ZQC(JL,JK))
      if (ZTC(JL,JK)-TCDK .GT. 0.) then
         ZLDCPE = CHLC /(CPD+ZCONS1*ZQC(JL,JK))
      else
         ZLDCPE = CHLS /(CPD+ZCONS1*ZQC(JL,JK))
      endif
      TEMP1 = ZTC(JL,JK)
      ZCOR=ZLDCPE*FODQA(ZQSATC,TEMP1)
      ZQCD=AMAX1(0.,(ZQC(JL,JK)-ZQSATC)/(1.+ZCOR))
      LO=ZQCD.EQ.0.
!      IQCD(JL) = CVMGT(0.,1.,LO)
      if (ZQCD.EQ.0.) then
        IQCD(JL) = 0.
      else
        IQCD(JL) = 1.
      endif
      ZQC(JL,JK)=ZQC(JL,JK)-ZQCD
      ZTC(JL,JK)=ZTC(JL,JK)+ZQCD*ZLDCPE
  327 CONTINUE
      ENDIF
      IS=0
      DO 324 JL=1,NI
      IS=IS+IQCD(JL)
  324 CONTINUE
      IF (IS.NE.0) THEN
      IF(SATUCO)THEN
      DO 325 JL=1,NI
      TEMP1 = ZTC (JL,JK)
      TEMP2 = ZPP1(JL,JK)
      ZQSATC= FOQST(TEMP1,TEMP2)
!      ZLDCPE = CVMGT(CHLC,CHLS,ZTC(JL,JK)-TCDK .GT. 0.)
!     *      /(CPD+ZCONS1*ZQC(JL,JK))
      if (ZTC(JL,JK)-TCDK .GT. 0.) then
         ZLDCPE = CHLC /(CPD+ZCONS1*ZQC(JL,JK))
      else
         ZLDCPE = CHLS /(CPD+ZCONS1*ZQC(JL,JK))
      endif
      ZCOR=ZLDCPE*FODQS(ZQSATC,TEMP1)
      ZQCD=(ZQC(JL,JK)-ZQSATC)/(1.+ZCOR)
      LO1(JL)=IQCD(JL).NE.0
!      ZQCD = CVMGT(ZQCD,0.,LO1(JL))
      if (IQCD(JL) .eq. 0) ZQCD = 0.
      ZQC(JL,JK)=ZQC(JL,JK)-ZQCD
      ZTC(JL,JK)=ZTC(JL,JK)+ZQCD*ZLDCPE
  325 CONTINUE
      ELSE
      DO 328 JL=1,NI
      TEMP1 = ZTC (JL,JK)
      TEMP2 = ZPP1(JL,JK)
      ZQSATC= FOQSA(TEMP1,TEMP2)
!      ZLDCPE= CVMGT(CHLC, CHLS,  ZTC(JL,JK)-TCDK .GT. 0.)
!     *      /(CPD+ZCONS1*ZQC(JL,JK))
      if (ZTC(JL,JK)-TCDK .GT. 0.) then
         ZLDCPE = CHLC /(CPD+ZCONS1*ZQC(JL,JK))
      else
         ZLDCPE = CHLS /(CPD+ZCONS1*ZQC(JL,JK))
      endif
      ZCOR=ZLDCPE*FODQA(ZQSATC,TEMP1)
      ZQCD=(ZQC(JL,JK)-ZQSATC)/(1.+ZCOR)
      LO1(JL)=IQCD(JL).NE.0
!      ZQCD = CVMGT(ZQCD,0.,LO1(JL))
      if (IQCD(JL) .eq. 0) ZQCD = 0.
      ZQC(JL,JK)=ZQC(JL,JK)-ZQCD
      ZTC(JL,JK)=ZTC(JL,JK)+ZQCD*ZLDCPE
  328 CONTINUE
      ENDIF
      ENDIF
      DO 326 JL=1,NI
      LO1(JL)=ZQP1(JL,JK).LE.ZQSATE(JL,JK)
!      ZTC(JL,JK) = CVMGT(ZTP1(JL,JK),ZTC(JL,JK),LO1(JL))
      if (LO1(JL)) ZTC(JL,JK) = ZTP1(JL,JK)
!      ZQC(JL,JK) = CVMGT(ZQP1(JL,JK),ZQC(JL,JK),LO1(JL))
      if (LO1(JL)) ZQC(JL,JK) = ZQP1(JL,JK)
  326 CONTINUE
!
!
  329 CONTINUE
!
!
!
!***
      DO 645 JK=1,NK
!
!
!*         3.3     CALCULATE RAIN/SNOW FLUX IN SUPERSATURATED LAYERS.
!
  330 CONTINUE
!
!***
      DO 331 JL=1,NI
      LO = ZTC(JL,JK) .GT. TGL
      ZSTPRO    =AMAX1((ZQP1(JL,JK)-ZQC(JL,JK)),0.)
      TEMP1=ZSTPRO*ZDPP1(JL,JK)*ZCONS2
!      ZRFLN(JL)=ZRFL(JL)+CVMGT(ZSTPRO    *ZDPP1(JL,JK)*ZCONS2,0.,LO)
!      ZSFLN(JL)=ZSFL(JL)+CVMGT(0.,ZSTPRO    *ZDPP1(JL,JK)*ZCONS2,LO)
      ZRFLN(JL)=ZRFL(JL)
      ZSFLN(JL)=ZSFL(JL)
      IF (LO) THEN
         ZRFLN(JL) = ZRFLN(JL) + TEMP1
      ELSE
         ZSFLN(JL) = ZSFLN(JL) + TEMP1
      ENDIF
  331 CONTINUE
!
!
!***
      IF (JK.GT.1) THEN
!
!     ------------------------------------------------------------------
!
!*         3.     EVAPORATION OF PRECIPITATIONS.
!                 ----------- -- ---------------
!
  400 CONTINUE
!
      DO 521 JL=1,NI
!***
!
!*         3.2     EVAPORATION OF PRECIPITATIONS.
!

  420 CONTINUE
      ZSQFLN = SQRT( ZRFLN(JL)+ZSFLN(JL) )
      TEMP1 = ZQSATE(JL,JK)
      TEMP2 = ZTP1(JL,JK)
!      ZLDCPE= CVMGT(CHLC, CHLS, TEMP2-TCDK .GT. 0.)
!     *      /(CPD+ZCONS1*ZQC(JL,JK))
      if (TEMP2-TCDK .GT. 0.) then
         ZLDCPE = CHLC /(CPD+ZCONS1*ZQC(JL,JK))
      else
         ZLDCPE = CHLS /(CPD+ZCONS1*ZQC(JL,JK))
      endif

!
      ZNIMP = 1. + 2.*(1.+ ZLDCPE*FODQS(TEMP1,TEMP2)) &
                    *ZEVAP*ZSQFLN/ZCONS2
!
      ZFLN(JL) = (AMAX1(0.,ZSQFLN-ZEVAP*ZDPP1(JL,JK) &
                 *AMAX1(0.,ZQSATE(JL,JK)-ZQP1(JL,JK))/ZNIMP ))**2
!
!
!     ------------------------------------------------------------------
!
!*         4.     MELTING/FREEZING, OUTGOING RAIN/SNOW FLUXES.
!                 ----------------- -------- --------- ------
!
  500 CONTINUE
!
!
!
!*         5.1     MELTING/FREEZING OF PRECIPITATIONS.
!*         5.2     OUTGOING FLUXES AT THE BOTTOM OF THE LAYER.
!
  520 CONTINUE
      ZDIP    =ZDPP1(JL,JK)/ZPP1(JL,JK)**2
      ZRITO=(ZSFLN(JL)/AMAX1(ZSFLN(JL)+ZRFLN(JL),ZEPFLM))
      ZRIT=ZRITO-ZMELT*ZDIP    *(ZTC(JL,JK)-TGL)/AMAX1(ZEPFLS,0.5 &
           *(SQRT(ZFLN(JL))+SQRT(ZRFL(JL)+ZSFL(JL))))
      ZRIT=AMIN1(1.,AMAX1(0.,ZRIT))
      ZSFLN(JL)=ZRIT*ZFLN(JL)
      ZRFLN(JL)=ZFLN(JL)-ZSFLN(JL)
  521 CONTINUE
!
!
!
!     ------------------------------------------------------------------
!
!*         5.     TENDENCIES DUE TO CONDENSATION, SURFACE FLUXES.
!                 ---------- --- -- ------------- ------- -------
!
  600 CONTINUE
!
!
!*         5.2     UPDATE FLAG IN ACTIVE LAYERS.
!
  620 CONTINUE
!***
      ENDIF
!***
      DO 621 JL=1,NI
      ZLVDCP=CHLC/(CPD+ZCONS1*ZQC(JL,JK))
      ZLSDCP=CHLS/(CPD+ZCONS1*ZQC(JL,JK))
      QE(JL,JK)= -((ZRFLN(JL)-ZRFL(JL))+(ZSFLN(JL) &
                 - ZSFL(JL)))*(GRAV/ZDPP1(JL,JK))
      TE(JL,JK)=(ZLVDCP*(ZRFLN(JL)-ZRFL(JL))+ZLSDCP*(ZSFLN(JL) &
                - ZSFL(JL)))*(GRAV/ZDPP1(JL,JK))
  621 CONTINUE
!
!*         5.3     DO ZONAL MEAN AND BOX DIAGNOSTICS.
!
  630 CONTINUE
!
!*         5.4     SWAP OF FLUXES, END OF VERTICAL LOOP AND STABLE
!*                 RAIN AND SNOW RATES.
!
  640 CONTINUE
      DO 641 JL=1,NI
      ZRFL(JL)=ZRFLN(JL)
      ZSFL(JL)=ZSFLN(JL)
  641 CONTINUE
!***
  645 CONTINUE
!***
      DO 647 JL=1,NI
      SRR(JL)=ZRFL(JL)
      SSR(JL)=ZSFL(JL)
  647 CONTINUE
!
!
!
      RETURN
      END
