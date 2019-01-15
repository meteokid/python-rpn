!     ################################################################
      SUBROUTINE CONVECT_SATMIXRATIO( KLON,                          &
                                    & PPRES, PT, PEW, PLV, PLS, PCPH )
!     ################################################################

!!**** Compute vapor saturation mixing ratio over liquid water
!!
!!
!!    PDRPOSE
!!    -------
!!     The purpose of this routine is to determine saturation mixing ratio
!!     and to return values for L_v L_s and C_ph
!!
!!
!!**  METHOD
!!    ------
!!
!!    EXTERNAL
!!    --------
!!     None
!!
!!
!!    IMPLICIT ARGUMENTS
!!    ------------------
!!      Module YOMCST
!!          RALPW, RBETW, RGAMW ! constants for water saturation pressure
!!          RD, RV             ! gaz  constants for dry air and water vapor
!!          RCPD, RCPV           ! specific heat for dry air and water vapor
!!          RCW, RCS             ! specific heat for liquid water and ice
!!          RTT                  ! triple point temperature
!!          RLVTT, RLSTT         ! vaporization, sublimation heat constant
!!
!!
!!    REFERENCE
!!    ---------
!!
!!      Book_ONE__TWO_of documentation ( routine CONVECT_SATMIXRATIO)
!!
!!    AUTHOR
!!    ------
!!      P. BECHTOLD       * Laboratoire d'Aerologie *
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    _ZERO_/_ONE_/_HALF_
!!   Last modified  _ZERO_/_ONE_/97
!------------------------- ------------------------------------------------------


!*       _ZERO_    DECLARATIONS
!              ------------

USE YOMCST
#include "tsmbkind.cdk"


IMPLICIT NONE
#include <arch_specific.hf>

!*       1.1  Declarations of dummy arguments :


INTEGER_M,                  INTENT(IN) :: KLON    ! horizontal loop index
REAL_B, DIMENSION(KLON),  INTENT(IN) :: PPRES   ! pressure
REAL_B, DIMENSION(KLON),  INTENT(IN) :: PT      ! temperature

REAL_B, DIMENSION(KLON),  INTENT(OUT):: PEW     ! vapor saturation mixing ratio
REAL_B, DIMENSION(KLON),  INTENT(OUT):: PLV     ! latent heat L_v
REAL_B, DIMENSION(KLON),  INTENT(OUT):: PLS     ! latent heat L_s
REAL_B, DIMENSION(KLON),  INTENT(OUT):: PCPH    ! specific heat C_ph

!*       1.2  Declarations of local variables :

REAL_B, DIMENSION(KLON)              :: ZT      ! temperature
REAL_B                               :: ZEPS           ! R_d / R_v


!-------------------------------------------------------------------------------

    ZEPS      = RD / RV

    ZT(:)     = MIN( 400._JPRB, MAX( PT(:), 10._JPRB ) ) ! overflow bound
    PEW(:)    = EXP( RALPW - RBETW / ZT(:) - RGAMW * LOG( ZT(:) ) )
    PEW(:)    = ZEPS * PEW(:) / ( PPRES(:) - PEW(:) )

    PLV(:)    = RLVTT + ( RCPV - RCW ) * ( ZT(:) - RTT ) ! compute L_v
    PLS(:)    = RLSTT + ( RCPV - RCS ) * ( ZT(:) - RTT ) ! compute L_i

    PCPH(:)   = RCPD + RCPV * PEW(:)                     ! compute C_ph

END SUBROUTINE CONVECT_SATMIXRATIO

