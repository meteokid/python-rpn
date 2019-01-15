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
!** S/P BAKTOTQ4
!
      SUBROUTINE BAKTOTQ4 (T, QV, QC, TM, S, SW, PS, TIF, FICE, &
                          DT, DQV, DQC, &
                          TVE, QCBL, FNN, FN, ZN, ZE, MG, &
                          AT2T,AT2M,AT2E,TAU, N, M, NK)
!
!
      implicit none
#include <arch_specific.hf>
!
!
      INTEGER N, M, NK
      REAL TAU
      REAL T(M,NK), QV(M,NK), QC(N,NK), TM(N,NK)
      REAL S(N,NK), SW(N,NK), PS(N), TIF(N,NK), FICE(N,NK)
      REAL DT(N,NK), DQV(N,NK), DQC(N,NK)
      REAL TVE(N,NK), QCBL(N,NK)
      REAL FNN(N,NK), FN(N,NK), ZN(N,NK), ZE(N,NK)
      REAL AT2T(N,NK),AT2M(N,NK),AT2E(N,NK)
      REAL MG(N)
!
!Author
!          J. Mailhot (Nov 2000)
!
!Revision
! 001      A.-M. Leduc (Oct 2001) Automatic arrays
! 002      B. Bilodeau and J. Mailhot (Dec 2001) Add a test to
!                      check the presence of advected explicit cloud water.
! 003      J. Mailhot (Nov 2000) Cleanup of routine
! 004      J. Mailhot (Feb 2003) - MOISTKE option based on implicit clouds only
! 005      A-M. Leduc (Jun 2003) - pass ps to clsgs---> clsgs2
! 006      J. P. Toviessi ( Oct. 2003) - IBM conversion
!               - calls to exponen4 (to calculate power function '**')
!               - etc.
! 007      B. Bilodeau (Dec 2003)   More optimizations for IBM
!                                   - Call to vspown1
!                                   - Replace divisions by multiplications
! 008      L. Spacek (Dec 2007) - add "vertical staggering" option
!                                 change the name to baktotq3
! 009      A. Zadra (Oct 2015) -- add land-water mask (MG) to input, which
!                                 is then passed on to CLSGS4
!
!
!Object
!          Transform conservative variables and their tendencies
!          back to non-conservative variables and tendencies.
!          Calculate the boundary layer cloud properties (cloud fraction, cloud
!          water content, flux enhancement factor).
!
!Arguments
!
!          - Input/Output -
! T        thetal on input (temperature on output)
! QV       qw (total water content = QV + QC) on input (specific humidity on output)
!
!          - Input -
! QC       cloud water content
! TM       temperature at current time
! S        sigma levels
! SW       sigma levels of T, Q
! PS       surface pressure (in Pa)
! TIF      temperature to compute ice fraction
! FICE     ice fraction
! MG       land-water mask
!
!          - Input/Output -
! DT       thetal tendency on input (temperature tendency on output)
! DQV      qw tendency on input (specific humidity tendency on output)
!
!          - Output -
! DQC      cloud water content tendency
!
!          - Input -
! TVE      virtual temperature on 'E' levels
! QCBL     cloud water content of BL clouds (subgrid-scale)
! FNN      flux enhancement factor (fN) * cloud fraction (N)
!
!          - Input/Output -
! FN       constant C1 in second-order moment closure (on input)
!          cloud fraction (on output)
!
!          - Input -
! ZN       length scale for turbulent mixing (on 'E' levels)
! ZE       length scale for turbulent dissipation (on 'E' levels)
! AT2T     coefficients for interpolation of T,Q to thermo levels
! AT2M     coefficients for interpolation of T,Q to momentum levels
! AT2E     coefficients for interpolation of T,Q to energy levels
! TAU      timestep
! N        horizontal dimension
! M        first dimension of T and QV
! NK       vertical dimension
!
!
!Notes
!          Retrieval of cloud water content is done by
!          a sub-grid-scale parameterization (implicit clouds)
!
!IMPLICITS
!
include "thermoconsts.inc"
!
!*
!
      INTEGER J, K
!
      REAL CPDINV, TAUINV
!
!
!*********************************************************
!     AUTOMATIC ARRAYS
!*********************************************************
!
      REAL, dimension(N,NK) :: EXNER
      REAL, dimension(N,NK) :: THL
      REAL, dimension(N,NK) :: QW
      REAL, dimension(N,NK) :: A
      REAL, dimension(N,NK) :: B
      REAL, dimension(N,NK) :: C
      REAL, dimension(N,NK) :: ALPHA
      REAL, dimension(N,NK) :: BETA
      REAL, dimension(N,NK) :: QCP
!
!*********************************************************
!
!
! MODULES
      EXTERNAL THERMCO2, CLSGS4


!
!
!------------------------------------------------------------------------
!
      CPDINV = 1./CPD
      TAUINV = 1./TAU
!
!       1. Retrieval of implicit cloud water content
!       --------------------------------------------
!
      CALL VSPOWN1(EXNER,SW,CAPPA,NK*N)
!
      DO K=1,NK
      DO J=1,N
        THL(J,K) = T(J,K) + TAU*DT(J,K)
        QW(J,K) = QV(J,K) + TAU*DQV(J,K)
      END DO
      END DO
!
      CALL THERMCO2 (T, QV, QC, SW, PS, TIF, FICE, FNN, &
                     THL, QW, A, B, C, ALPHA, BETA, &
                     0, .FALSE., N, M, NK)
!
!                                              retrieve QC from QW and THL (put in QCP)
      CALL CLSGS4 (THL, TVE, QW, QCP, FN, FNN, FN, &
                  ZN, ZE, S, PS, MG, A, B, C, AT2T, AT2M, AT2E, N, NK)
!
!
!       2.     Back to non-conservative variables (T and QV) and tendencies
!       -------------------------------------------------------------------
!
      DO K=1,NK
      DO J=1,N
!                                              back to T- and QV-
        T(J,K) = TM(J,K)
        QV(J,K) = QV(J,K) - MAX( 0.0 , QC(J,K) )
!
!                                              update QC and QCBL
        DQC(J,K) = ( MAX(0.0 , QCP(J,K)) - &
                     MAX(0.0 , QC(J,K)   ) )*TAUINV
!                                              prevent negative values for new QCBL
        DQC(J,K) = MAX( DQC(J,K) , -MAX( 0.0 ,QC(J,K) )*TAUINV )
        QCBL(J,K) =  MAX( 0.0 , QC(J,K) ) + DQC(J,K) * TAU
!                                              retrieve T, and QV tendencies
!                                              (T and QV updates are made elsewhere)
        DT(J,K) = EXNER(J,K)*DT(J,K) &
                  + ((CHLC+FICE(J,K)*CHLF)*CPDINV)*DQC(J,K)
        DQV(J,K) = DQV(J,K) - DQC(J,K)
!                                              prevent negative values for QV
        DQV(J,K) = MAX( DQV(J,K) , -MAX( 0.0 ,QV(J,K) )*TAUINV )
!                                              set cloud water content tendency to zero
        DQC(J,K) = 0.0
!
      END DO
      END DO
!
!
      RETURN
      END
