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
!** S/P BLWEIGHT2
!
      SUBROUTINE BLWEIGHT2 (W, S, PS, N, NK)
!
      implicit none
#include <arch_specific.hf>
!
!
      INTEGER N, NK
      REAL W(N,NK), S(N,NK)
      REAL PS(N)
!
!Author
!          J. Mailhot and B. Bilodeau (Dec 2001)
!
!
!Revision
!001       A-M. Leduc and S. Belair (Jun 2003) - ps as argument
!                   blweight ---> blweight2. Change weighting
!                   profile from sigma to pressure dependent.
!
!Object
!          Compute a weighting profile to be used with the moist
!          turbulence scheme.
!
!Arguments
!
!          - Output -
! W        weighting profile
!
!          - Input -
! S        sigma levels
! PS       surface pressure (in Pa)
! N        horizontal dimension
! NK       vertical dimension
!
!
!Notes
!          The profile is set to:
!            1 in the lower part of the atmosphere (S .ge. SMAX) if pres .ge. pmax
!            0 in the upper part of the atmosphere (S .le. SMIN) if pres .le. pmin
!            (with a linear interpolation in-between)
!
!
!
!******************************************************
!     AUTOMATIC ARRAYS
!******************************************************
!
      REAL, dimension(N,NK   ) :: PRES
!
!******************************************************


      INTEGER J, K
!
!      REAL SMIN, SMAX
!      SAVE SMIN, SMAX
       REAL PMIN, PMAX
       SAVE PMIN, PMAX
!
!***********************************************************
!
!      DATA SMIN , SMAX / 0.45 , 0.55 /
       DATA PMIN , PMAX / 45000, 55000 /
!
!
      DO K=1,NK
      DO J=1,N
!        W(J,K) = 1.0
!*
!        IF (S(J,K).LE.SMIN) THEN
!           W(J,K) = 0.0
!        ELSE IF (S(J,K).LE.SMAX.AND.S(J,K).GT.SMIN) THEN
!           W(J,K) = (1. - (SMAX - S(J,K)) / (SMAX-SMIN) )
!        ENDIF

         PRES(J,K)=S(J,K)*PS(J)
         W(J,K)=(1- (PMAX-PRES(J,K))/(PMAX-PMIN) )
         W(J,K)=MIN ( MAX ( W(J,K), 0.), 1.)


      END DO
      END DO
!
!
      RETURN
      END
