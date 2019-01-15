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
!** S/P THERMCO2
!
      SUBROUTINE THERMCO2 (T, QV, QC, S, PS, TIF, FICE, FNN, &
                           THL, QW, A, B, C, ALPHA, BETA, &
                           TYPE, INMODE, N, M, NK)
!
      implicit none
#include <arch_specific.hf>
!
!
      INTEGER N, M, NK
      REAL T(M,NK), QV(M,NK), QC(N,NK)
      REAL S(N,NK), PS(N), TIF(M,NK), FICE(N,NK), FNN(N,NK)
      REAL THL(N,NK), QW(N,NK), A(N,NK), B(N,NK), C(N,NK)
      REAL ALPHA(N,NK), BETA(N,NK)
      INTEGER TYPE
      LOGICAL INMODE
!
!Author
!          J. Mailhot (Nov 1999)
!
!Revision
! 001      J. Mailhot  (Jan 2000) - Changes to add mixed-phase mode
! 002      A.-M. Leduc (Oct 2001) Automatic arrays
! 003      J. Mailhot  (Jun 2002) - Add cloud type and input mode
!                       Change calling sequence and rename THERMCO2
! 004      A. Plante   (May 2003) - IBM conversion
!                         - calls to exponen4 (to calculate power function '**')
!                         - divisions replaced by reciprocals
!                         - calls to optimized routine mfdlesmx
! 005      B. Bilodeau (Aug 2003) - exponen4 replaced by vspown1
!
!Object
!          Calculate the thermodynamic coefficients used in the presence of clouds
!          and the conservative variables.
!
!Arguments
!
!          - Input -
! T        temperature
! QV       specific humidity
! QC       total cloud water content
! S        sigma levels
! PS       surface pressure (in Pa)
! TIF      temperature used to compute ice fraction
! FICE     ice fraction
! FNN      flux enhancement factor
!
!          - Input/Output -
! THL      ice-liquid potential temperature (thetal)
! QW       total water content (QW = QV + QC )
!
!          - Output -
! A        thermodynamic coefficient
! B        thermodynamic coefficient
! C        thermodynamic coefficient
! ALPHA    thermodynamic coefficient
! BETA     thermodynamic coefficient
!
!          - Input -
! TYPE     integer switch for cloud type: 0 = implicit only
!                                         1 = explicit only
!                                         2 = implicit/explicit
! INMODE   logical switch for input mode: .TRUE. = standard mode
!                                         .FALSE. = THL,QW are inputs
! N        horizontal dimension
! M        first dimension of T and QV
! NK       vertical dimension
!
!
!Notes
!          See definitions in:
!          - Bechtold and Siebesma 1998, JAS 55, 888-895
!
!
!IMPLICITS
!
include "thermoconsts.inc"
!
!*
!
      INTEGER J, K, ITOTAL
!
!
!
!*********************************************************
!     AUTOMATIC ARRAYS
!*********************************************************
!
      REAL, dimension(N,NK) :: PRES
      REAL, dimension(N,NK) :: EXNER
      REAL*8, dimension(N,NK) :: EXNERR
      REAL, dimension(N,NK) :: QSAT
      REAL, dimension(N,NK) :: DQSAT
      REAL, dimension(N,NK) :: TH
      REAL, dimension(N,NK) :: TL
      REAL, dimension(N,NK) :: FFICE
      REAL, dimension(N,NK) :: TFICE
      REAL, dimension(N,NK) :: DFICE
      REAL, dimension(N,NK) :: WORK
      REAL*8, dimension(N,NK) :: WORK8
!
!*********************************************************
!
!
include "dintern.inc"
include "fintern.inc"
!
!
!MODULES
      EXTERNAL FICEMXP,MFDLESMX
!------------------------------------------------------------------------
!
!
!
!      1.     Preliminaries
!      --------------------
!
      DO K=1,NK
      DO J=1,N
        PRES(J,K) = S(J,K)*PS(J)
        FFICE(J,K) = FICE(J,K)
      END DO
      END DO
      CALL VSPOWN1 (EXNER,S,CAPPA,N*NK)
      WORK8=EXNER
      CALL VREC(EXNERR,WORK8,N*NK)
!
      IF ( INMODE ) THEN
        IF ( TYPE .EQ. 0 ) &
          CALL FICEMXP(FFICE,TFICE,DFICE,TIF,N,M,NK)
        DO K=1,NK
        DO J=1,N
          TH(J,K) = T(J,K)*EXNERR(J,K)
          THL(J,K) = TH(J,K)*( 1.0 - ((CHLC+FFICE(J,K)*CHLF)/CPD) &
                              *( QC(J,K)/T(J,K) ) )
          QW(J,K) = QV(J,K)+QC(J,K)
        END DO
        END DO
      ENDIF
!
      DO K=1,NK
      DO J=1,N
        TL(J,K) = EXNER(J,K)*THL(J,K)
      END DO
      END DO
!
!
!      2.     Saturation specific humidity
!      -----------------------------------
!
      IF ( TYPE .EQ. 0 ) THEN
        CALL FICEMXP(A,TFICE,DFICE,TIF,N,M,NK)
        DO K=1,NK
        DO J=1,N
          QSAT(J,K) = FQSMX( TL(J,K), PRES(J,K), TFICE(J,K) )
        END DO
        END DO
        CALL MFDLESMX(C,TL,TFICE,DFICE,N,NK)
        DO K=1,NK
        DO J=1,N
          DQSAT(J,K) = FDQSMX( QSAT(J,K), C(J,K) )
        END DO
        END DO
!
      ELSEIF( TYPE .EQ. 1 ) THEN
        DO K=1,NK
        DO J=1,N
          QSAT(J,K) = FOQSA( TL(J,K), PRES(J,K) )
          DQSAT(J,K) = FODQA( QSAT(J,K), TL(J,K) )
        END DO
        END DO
!
      ENDIF
!
      IF ( TYPE .EQ. 2 ) THEN
        CALL FICEMXP(A,TFICE,DFICE,TIF,N,M,NK)
        CALL MFDLESMX(WORK,TL,TFICE,DFICE,N,NK)
        DO K=1,NK
        DO J=1,N
          IF ( FNN(J,K) .LT. 1.0 ) THEN
            QSAT(J,K) = FQSMX( TL(J,K), PRES(J,K), TFICE(J,K) )
            C(J,K) = WORK(J,K)
            DQSAT(J,K) = FDQSMX( QSAT(J,K), C(J,K) )
          ELSE
            QSAT(J,K) = FOQSA( TL(J,K), PRES(J,K) )
            DQSAT(J,K) = FODQA( QSAT(J,K), TL(J,K) )
          ENDIF
        END DO
        END DO
      ENDIF
!
!       3.     Thermodynamic coefficients
!       ---------------------------------
!                                              (cf. BS 1998 Appendix A)
      DO K=1,NK
      DO J=1,N
        A(J,K) = 1.0/( 1.0 + ((CHLC+FFICE(J,K)*CHLF)/CPD)*DQSAT(J,K) )
        B(J,K) = A(J,K)*EXNER(J,K)*DQSAT(J,K)
        C(J,K) = A(J,K)*( QW(J,K)-QSAT(J,K) )
!
        IF ( INMODE ) THEN
          ALPHA(J,K) = DELTA*TH(J,K)
          BETA(J,K) = ((CHLC+FFICE(J,K)*CHLF)/CPD)/EXNER(J,K) &
                      - (1.0+DELTA)*TH(J,K)
        ELSE
          ALPHA(J,K) = 0.0
          BETA(J,K) = 0.0
        ENDIF
!
      END DO
      END DO
!
!
!
      RETURN
      END
