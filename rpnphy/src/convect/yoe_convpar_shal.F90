!     ########################
      MODULE YOE_CONVPAR_SHAL
!     ########################

!!****  *YOE_CONVPAR_SHAL* - Declaration of convection constants
!!
!!    PURPOSE
!!    -------
!!      The purpose of this declarative module is to declare  the
!!      constants in the deep convection parameterization.
!!
!!
!!**  IMPLICIT ARGUMENTS
!!    ------------------
!!      None
!!
!!    REFERENCE
!!    ---------
!!      Book2 of documentation of Meso-NH (YOE_CONVPAR_SHAL)
!!
!!    AUTHOR
!!    ------
!!      P. Bechtold   *Laboratoire d'Aerologie*
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    26/03/96
!!   Last modified  04/10/98
!-------------------------------------------------------------------------------

#include "tsmbkind.cdk"

!*       0.   DECLARATIONS
!             ------------

IMPLICIT NONE

REAL_B, SAVE :: XA25        ! 25 km x 25 km reference grid area

REAL_B, SAVE :: XCRAD       ! cloud radius
REAL_B, SAVE :: XCTIME_SHAL ! convective adjustment time
REAL_B, SAVE :: XCDEPTH     ! minimum necessary cloud depth
REAL_B, SAVE :: XCDEPTH_D   ! maximum allowed cloud thickness
REAL_B, SAVE :: XDTPERT     ! add small Temp perturb. at LCL
REAL_B, SAVE :: XENTR       ! entrainment constant (m/Pa) = 0.2 (m)

REAL_B, SAVE :: XZLCL       ! maximum allowed allowed height
                            ! difference between departure level and surface
REAL_B, SAVE :: XZPBL       ! minimum mixed layer depth to sustain convection
REAL_B, SAVE :: XWTRIG      ! constant in vertical velocity trigger


REAL_B, SAVE :: XNHGAM      ! accounts for non-hydrost. pressure
                            ! in buoyancy term of w equation
                            ! = 2 / (1+gamma)
REAL_B, SAVE :: XTFRZ1      ! begin of freezing interval
REAL_B, SAVE :: XTFRZ2      ! end of freezing interval


REAL_B, SAVE :: XSTABT      ! factor to assure stability in  fractional time
                            ! integration, routine CONVECT_CLOSURE
REAL_B, SAVE :: XSTABC      ! factor to assure stability in CAPE adjustment,
                            !  routine CONVECT_CLOSURE

END MODULE YOE_CONVPAR_SHAL

