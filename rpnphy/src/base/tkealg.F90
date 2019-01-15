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
!**S/P  TKEALG
!
      SUBROUTINE TKEALG(ESTAR,EN,ZN,ZE,DVDZ2,RI,TAU,N,NK)
!
      implicit none
#include <arch_specific.hf>
!
      INTEGER N,NK
      REAL ESTAR(N,NK),EN(N,NK),ZN(N,NK)
      REAL ZE(N,NK),DVDZ2(N,NK),RI(N,NK)
      REAL TAU
!
!Author
!          J. Mailhot (August 1999)
!Revision
! 001      A.-M. Leduc (Oct 2001) Automatic arrays
! 002      B. Bilodeau (Mar 2003) Include machcon.cdk
! 003      A. Plante   (May 2003) IBM conversion
!             - calls to vsqrt routine (from massvp4 library)
!             - calls to vstan routine (from massvp4 library)
!             - calls to vslog routine (from massvp4 library)
! 004      a. Plante   (juillet 2003) correction of bugs in IBM conversion
!             - optimization of tan removed
!             - optimization of log removed
!
!Object
!          Solve the algebraic part of the TKE equation
!
!Arguments
!
!          - Output -
! ESTAR    turbulent kinetic energy (at time *)
!
!          - Input -
! EN       turbulent kinetic energy (at time n)
! ZN       mixing length of the turbulence
!
!          - Input/Output -
! ZE       dissipation length of the turbulence (on input)
!          viscous dissipation term (on output - for time series ED)
! DVDZ2    mechanical (shear) component (on input)
!          mechanical production term (on output - for time series EM)
! RI       buoyancy flux term - in flux form - (on input)
!          thermal production term (on output - for time series EB)
!
!          - Input -
! TAU      timestep
! N        horizontal dimension
! NK       vertical dimension
!
!
!IMPLICITS
#include "machcon.cdk"
#include "clefcon.cdk"
!
      INTEGER J,K
      INTEGER ITOTAL
!
!
!
!*
!
!*********************************************************
!     AUTOMATIC ARRAYS
!*********************************************************
!
      REAL, dimension(N,NK) :: B
      REAL, dimension(N,NK) :: C
      REAL, dimension(N,NK) :: D
      REAL, dimension(N,NK) :: ETA
      REAL, dimension(N,NK) :: WORK
      REAL*8, dimension(N,NK) :: WORK_8
!
!*********************************************************
!
!
!
!
!       1.     Preliminaries
!       --------------------
!
!
!
!
      DO K=1,NK-1
      DO J=1,N
!                                              mechanical and thermal part (either + or -)
         B(J,K)=BLCONST_CK*ZN(J,K)*(DVDZ2(J,K)-RI(J,K))
!                                              dissipation part (always +)
         C(J,K)=BLCONST_CE/ZE(J,K)
         D(J,K)=(ABS(B(J,K)/C(J,K)))
!
      END DO
      END DO
!
!     IBM CONV : Plus de chiffre dans stat en real*8
      WORK_8(:,1:nk-1)=D(:,1:nk-1)
      CALL VSQRT(WORK_8,WORK_8,N*(NK-1))
      D(:,1:nk-1)=WORK_8(:,1:nk-1)
      WORK_8=EN
      CALL VSQRT(WORK_8,WORK_8,N*(NK-1))
      ETA=WORK_8
!
!     IBM CONV : Seulement 3 chiffres dans stat en real
!      CALL VSSQRT(D,D,N*(NK-1))
!      CALL VSSQRT(ETA,EN,N*(NK-1))
!
!
!       2.     Solve the algebraic TKE equation
!       ---------------------------------------
!
!
      DO K=1,NK-1
      DO J=1,N
!
        IF( abs(B(J,K)) < epsilon(B) ) THEN
          ESTAR(J,K) = ETA(J,K)/ &
                    (1.+0.5*ETA(J,K)*C(J,K)*TAU)
        ELSEIF( B(J,K) > 0. ) THEN
!         Note : il n'est pas avantageux de changer ce exp pour IBM
!                du moins sur la slab testee.
          ESTAR(J,K)=D(J,K)*( -1.0+2.0/(1.0+EXP(-D(J,K)*C(J,K)*TAU) &
                        *(-1.0+2.0/ &
                         (1.0+ETA(J,K)/D(J,K)))) )
        ELSE
!         Note : il n'est pas avantageux de changer cette TAN pour IBM
!                du moins sur la slab testee.
          ESTAR(J,K)=D(J,K)*TAN(MIN(TANLIM,MAX(ATAN(ETA(J,K)/D(J,K)) &
                         -0.5*D(J,K)*C(J,K)*TAU , 0.0 ) ) )
        ENDIF
!
        ESTAR(J,K)=MAX( ETRMIN , ESTAR(J,K)**2 )
!
      END DO
      END DO
!
!
!       3.     For output purposes (time series)
!       ----------------------------------------
!
!
      DO K=1,NK-1
      DO J=1,N
!
         D(J,K)=B(J,K)/C(J,K)
         B(J,K)=0.0
         IF( EN(J,K).NE.D(J,K).AND.ESTAR(J,K).NE.D(J,K) ) THEN
           B(J,K)=ALOG( ABS( (ESTAR(J,K)-D(J,K)) / (EN(J,K)-D(J,K)) ) )
         ENDIF
         ETA(J,K)=BLCONST_CK*ZN(J,K)*B(J,K)/(C(J,K)*TAU)
!                                              DVDZ2 contains EM
         DVDZ2(J,K)=-ETA(J,K)*DVDZ2(J,K)
!                                              RI contains EB
         RI(J,K)=ETA(J,K)*RI(J,K)
!                                              ZE contains ED
         ZE(J,K)=(ESTAR(J,K)-EN(J,K))/TAU-(DVDZ2(J,K)+RI(J,K))
!
      END DO
      END DO
!
      DO J=1,N
         EN(J,NK)=0.0
         ESTAR(J,NK)=0.0
         DVDZ2(J,NK)=0.0
         RI(J,NK)=0.0
         ZE(J,NK)=0.0
      END DO
!
!
!
      RETURN
      END
