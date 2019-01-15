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
!**S/P TOHERMO
!
      SUBROUTINE TOTHERMO (FMOM,FTHE,AT2T,AT2M,NI,NJ,NK,MO2TH)
!
      implicit none
#include <arch_specific.hf>
      INTEGER NI,NJ,NK
      LOGICAL MO2TH
      REAL FMOM(NI,NK), FTHE(NI,NK)
      REAL AT2T(NI,NJ), AT2M(NI,NJ)
!
!Author
!          L. Spacek (Dec 2007)
!
!Revision
!
!Object
!          vertical interpolation of T/Q to momentum or
!          ttermo levels
!
!Arguments
!
!          - Input/Output -
! FMOM     variable at sigma momentum levels
! FTHE     variable at sigma thermo levels
!
!          - Input -
! AT2T     coefficients for interpolation of T,Q to thermo levels
! AT2M     coefficients for interpolation of T,Q to momentum levels
! NI       horizontal dimension
! NJ       vertical dimension
! NK       loop length
! MO2TH    .true.  mommentum -> thermo
!          .false. thermo    -> momentum
!
!*
      INTEGER K, I, KP1, KM1
!
!
      IF(MO2TH)THEN
         DO K=1,NK
            KP1=MIN(NK,K+1)
            DO I=1,NI
               FTHE(I,K)=FMOM(I,K)+AT2T(I,K)*(FMOM(I,KP1)-FMOM(I,K))
            ENDDO
         ENDDO
      ELSE
         DO K=NK,1,-1
            KM1=MAX(1,K-1)
            DO I=1,NI
               FMOM(I,K)=FTHE(I,K)-AT2M(I,K)*(FTHE(I,K)-FTHE(I,KM1))
            ENDDO
         ENDDO
      ENDIF
!
      RETURN
      END
