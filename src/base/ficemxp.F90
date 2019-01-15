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
!** S/P FICEMXP
!
      SUBROUTINE FICEMXP (FICE, TF, DF, T, N, M, NK)
!
      implicit none
#include <arch_specific.hf>
!
!
      INTEGER N, M, NK
      REAL FICE(N,NK), TF(N,NK), DF(N,NK)
      REAL T(M,NK)
!
!Author
!          J. Mailhot (Jan 2000)
!
!Object
!          Calculate the fraction of ice and set the values of threshold
!          and derivative w/r to T for computation of saturation values
!          in the presence of mixed phases.
!
!Arguments
!
!          - Output -
! FICE     fraction of ice
! TF       threshold value for saturation w/r to ice or liquid
! DF       value of derivative w/r to T for saturation w/r to ice or liquid
!
!          - Input -
! T        temperature
!
!          - Input -
! N        horizontal dimension
! M        first dimension of T
! NK       vertical dimension
!
!
!Notes
!          Based on the definition in:
!          - Burk et al. 1997, JGR 102, 16529-16544
!          and the observations of:
!          - Curry et al. 1990, Int. J. Climatol. 10, 749-764.
!          - Curry et al. 1997, JGR 102, 13851-13860.
!
!          For F (fraction of ice), linear variation between Tmin and Tmax
!          For TF and DF, values are set such that saturation is w/r to liquid for T > Tmin
!                 "         "             "        saturation is w/r to ice    for T < Tmin
!
!*
!
      INTEGER J, K
!
      REAL DT
      REAL TMIN, TMAX
!
      SAVE TMIN, TMAX
      DATA TMIN, TMAX / 248.16, 258.16 /
!
      DT= 1.0/(TMAX-TMIN)
!
      DO K=1,NK
      DO J=1,N
        FICE(J,K)= MAX( 0.0 , MIN( 1.0 , (TMAX-T(J,K))*DT ) )
        TF(J,K)= FICE(J,K)
        DF(J,K)= -DT
        IF( T(J,K).LT.TMIN .OR. T(J,K).GT.TMAX) DF(J,K) = 0.0
      END DO
      END DO
!
!
      RETURN
      END
