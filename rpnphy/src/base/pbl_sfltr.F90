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
!**S/R  pbl_sfltr    - FILTERING OF A FIELD IN THE S DIRECTION
!                      WITH A GENERAL SYMMETRIC THREE POINT OPERATOR
!
      SUBROUTINE pbl_sfltr ( R , F , CS , N , NK )
!
      implicit none
#include <arch_specific.hf>
      INTEGER N,NK
      REAL R(N,NK),F(N,NK),CS
!
!Author
!          J. Mailhot   (Sept 1984)
!
!Revision
! 001      J. Cote(Nov. 1984), Vectorization, Documentation
! 002      M. Lepine  -  RFE model code revision project (Feb 87)
!                          -  Delete the test [IF (LOC(R).NE.LOC(F))]
!                             for transportability
!
!Object
!          to filter a field in the S direction with a general
!          symmetric three point operator
!
!Arguments
!
!          - Output -
! R        result (can share memory location with F)
!
!          - Input -
! F        field to be filtered
! CS       filter coefficient along S
! N        horizontal dimension
! NK       vertical dimension
!
!Notes
!          R = F + (CS/2) * Fss (in the interior)
!          with Fss = F(J,K+1) + F(J,K-1) - 2 * F(J,K)
!          An average preserving correction is applied at the
!          boundaries
!
!*
!
      REAL CSO2
      INTEGER J,K,LL,LM,LO,JK
      REAL W(N,2)
!
      REAL EPS
      SAVE EPS
      DATA EPS / 1.E-37 /
!
      IF (ABS(CS).GT.EPS) THEN
!
!     FILTER ALONGS
!
         CSO2=CS/2.0
         LM=1
         LO=2
!
         DO 1 J=1,N
            W(J,1)=CSO2*(F(J,2)-F(J,1))
    1       R(J,1)=F(J,1)+W(J,1)
!
         DO 3 K=2,NK-1
            DO 2 J=1,N
               W(J,LO)=CSO2*(F(J,K+1)-F(J,K))
    2          R(J,K)=F(J,K)+W(J,LO)-W(J,LM)
            LL=LO
            LO=LM
    3       LM=LL
!
         DO 4 J=1,N
    4       R(J,NK)=F(J,NK)-W(J,LM)
!
      ELSE
          DO 5 JK=1,N*NK
    5        R(JK,1) = F(JK,1)
      ENDIF
!
      RETURN
      END
