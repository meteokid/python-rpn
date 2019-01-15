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
!** S/P INIFCP
      SUBROUTINE INIFCP ( PSB,PS,PSM,RAINCV,RC, &
                          FCPFLG,NCA,SCR3,OMEGAP, &
                          AVERT,SIGMA,S, &
                          PTOP,NI,NK,DT)
      implicit none
#include <arch_specific.hf>
!
      INTEGER NI,NK
      INTEGER NCA(NI)
      REAL PSB(NI),PS(NI),PSM(NI)
      REAL RAINCV(NI),RC(NI),FCPFLG(NI)
      REAL SCR3(NI,NK),OMEGAP(NI,NK)
      REAL AVERT(NI,NK)
      REAL SIGMA(NI,NK+1),S(NI,NK)
      REAL PTOP,DT
!
!Author
!          Stephane Belair (July 5,1991)
!
!Revision
!
!Object
!          inteface between the RFE model and the
!          Fritsch-Chappell scheme (FCPARA)
!          and the Kain-Fritsch scheme (KFCP)
!
!Arguments
!
!          - Input -
! NI       X dimension of the model grid
! NJ       Y dimension of the model grid
! NK       Z dimension of the model grid
!
!          - Output -
! PSB      pressure at the bottom of the atmosphere
!
!          - Input -
! PS       surface pressure at time (T+1)
! PSM      surface pressure at time (T-1) (not used)

!          - Output -
! RAINCV   accumulated convective precip. during last timestep
! RC       rate of convective precipitation for previous timestep
!
!          - Input -
! FCPFLG   flag for Fritsch-Chappell scheme (real    value)
!
!          - Output -
! NCA      flag for Fritsch-Chappell scheme (integer value)
! SCR3     dp/dt (vertical velocity) (kPa/s)
!
!          - Input -
! OMEGAP   dp/dt (vertical velocity) (Pa/s)
!
!          - Output -
! AVERT    a subset of the model's sigma levels
! SIGMA    staggered sigma levels
!
!          - Input -
! S        sigma levels of the model
!
!          - Output -
! PTOP     pressure at the top of the atmosphere
! DT       length of timestep
!
!
!Notes
!
!*
      INTEGER I,K,KZ
!
!
!     SIGMA COORDINATES
!     -----------------
!
!     THE TWO VERTICAL ARRAYS ARE :
!
!     AVERT(K)  : WHERE ALL THE VARIABLES ARE DEFINED
!     SIGMA (K) : STAGGERED LEVELS
!
!     AT THE BOTTOM
!
!VDIR NODEP
      DO I=1,NI
         SIGMA(I,NK+1) = S(I,NK)
         AVERT(I,NK  ) = S(I,NK-1)+0.75*(S(I,NK)-S(I,NK-1))
      END DO
!
!
!     IN BETWEEN
!
      DO K=1,NK-1
!VDIR NODEP
         DO I=1,NI
            AVERT(I,K) = S(I,K)
         END DO
      END DO
!
      DO K=1,NK-1
         KZ = NK-K+1
!VDIR NODEP
         DO I=1,NI
            SIGMA(I,KZ) = 2.0*AVERT(I,KZ)-SIGMA(I,KZ+1)
            IF (SIGMA(I,KZ).LT.AVERT(I,KZ-1))                    THEN
               SIGMA(I,KZ) = (AVERT(I,KZ) + AVERT(I,KZ-1)) * 0.5
            ENDIF
         END DO
      END DO
!
!
!     AT THE TOP
!
!VDIR NODEP
      DO I=1,NI
         SIGMA(I,1) = 2.0*AVERT(I,1)-SIGMA(I,2)
         IF (SIGMA(I,1).LT.0.0 ) SIGMA(I,1)= AVERT(I,1) * 0.5
      END DO
!
!
!     PRESSURE AT THE TOP OF THE ATMOSPHERE
!     -------------------------------------
!
      PTOP=0.0
!
!
!     UNIT CONVERSIONS
!     ----------------
!
!VDIR NODEP
      DO I=1,NI
         PSB   (I) = PS(I)*1.E-3
         RAINCV(I) = RC(I)*DT*100.
         NCA   (I) = INT(FCPFLG(I))
      END DO
!
!
      DO K=1,NK
!VDIR NODEP
         DO I=1,NI
!           CONVERSION D'UNITES (VOIR S/P FCPARA ET KFCP)
            SCR3(I,K) = OMEGAP(I,K)*1.E-3
         END DO
      END DO
!
!
!VDIR NODEP
      DO I=1,NI
        SCR3(I,NK) = SCR3(I,NK-1) + 0.75* &
                    ( SCR3(I,NK) - SCR3(I,NK-1) )
      END DO
!
!
!
      RETURN
      END
