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
!**S/P MWEIGHTS
!
      SUBROUTINE MWEIGHTS (ATQ2M,SIGM,SIGT,NI,NJ,NK,LSTAG)
!
      implicit none
#include <arch_specific.hf>
      INTEGER NI,NJ,NK
      LOGICAL LSTAG
      REAL ATQ2M(NI,NJ),SIGM(NI,NJ),SIGT(NI,NJ)
!
!Author
!          L. Spacek (Dec 2007)
!
!Revision
!
!Object
!          to calculate the linear coefficients for interpolation
!          of temperature/humidity to momentum levels
!
!Arguments
!
!          - Output -
! ATQ2T    coefficients for interpolation of T/Q to momentum levels
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
!***********************************************************************
!     AUTOMATIC ARRAYS
!***********************************************************************
!
      REAL, dimension(NI,NK-1) :: WRK
!
!*
      IF(LSTAG) THEN
         WRK(:,1:NK-1)=SIGT(:,2:NK)-SIGT(:,1:NK-1)
!
         CALL VSREC(WRK,WRK,NI*(NK-1))
!
         ATQ2M(:,1)=0.0
         DO K=2,NK
            DO I=1,NI
               ATQ2M(I,K)=(SIGT(I,K)-SIGM(I,K))*WRK(I,K-1)
            ENDDO
         ENDDO
!
      ELSE
         ATQ2M=0.0
      ENDIF
!
      RETURN
      END

