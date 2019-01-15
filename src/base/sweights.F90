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
!**S/P SWEIGHTS
!
      SUBROUTINE SWEIGHTS (AT2E,SIGM,SIGT,NI,NJ,NK,LSTAG)
!
      implicit none
#include <arch_specific.hf>
      INTEGER NI,NJ,NK
      LOGICAL LSTAG
      REAL AT2E(NI,NJ),SIGM(NI,NJ),SIGT(NI,NJ)
!
!Author
!          L. Spacek (Dec 2007)
!
!Revision
! 001      L. Spacek (Sep 2008)- add coefficients for extrapolation in gwd5
!
!Object
!          to calculate the linear coefficients for interpolation
!          from temperature levels to 'E' levels
!
!Arguments
!
!          - Output -
! AT2E     coefficients for interpolation of T,Q to 'E' levels
!
!          - Input -
! SIGM     sigma momentum levels
! SIGT     sigma thermo levels
! NI       horizontal dimension
! NJ       horizontal dimension
! NK       number of levels
! LSTAG    .true.  model is staggered
!          .false. model is non-staggered
!
!*
      INTEGER I, K
!
!
      IF(LSTAG) THEN
         DO K=1,NK+1
            DO I=1,NI
               AT2E(I,K)=0.0
            ENDDO
         ENDDO
      ELSE
         DO K=1,NK-1
            DO I=1,NI
               AT2E(I,K)=0.5
            ENDDO
         ENDDO
         AT2E(:,NK)=0.0
!
! Coefficients for extrapolation in gwd5
!
      DO I=1,NI
         AT2E(I,NK+1)=.5*(1.-SIGM(I,NK))/(SIGM(I,NK)-SIGM(I,NK-1))
      ENDDO

      ENDIF
!
      RETURN
      END
