!-------------------------------------------------------------------------------

!     ###################
      MODULE YOE_CONVPAR
!     ###################

!!****  *YOE_CONVPAR* - Declaration of convection constants
!!
!!    PURPOSE
!!    -------
!      The purpose of this declarative module is to declare  the
!      constants in the deep convection parameterization.

!!
!!**  IMPLICIT ARGUMENTS
!!    ------------------
!!      None
!!
!!    REFERENCE
!!    ---------
!!      Book2 of documentation of Meso-NH (YOE_CONVPAR)
!!
!!    AUTHOR
!!    ------
!!      P. Bechtold   *Laboratoire d'Aerologie*
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    26/03/96
!!   Last modified  15/11/96
!-------------------------------------------------------------------------------

#include "tsmbkind.cdk"

!*       0.   DECLARATIONS
!             ------------

IMPLICIT NONE

REAL_B, SAVE :: XA25        ! 25 km x 25 km reference grid area

REAL_B, SAVE :: XCRAD       ! cloud radius
REAL_B, SAVE :: XCDEPTH     ! minimum necessary cloud depth
REAL_B, SAVE :: XENTR       ! entrainment constant (m/Pa) = 0.2 (m)

REAL_B, SAVE :: XZLCL       ! maximum allowed allowed height
                            ! difference between departure level and surface
REAL_B, SAVE :: XZPBL       ! minimum mixed layer depth to sustain convection
REAL_B, SAVE :: XWTRIG      ! constant in vertical velocity trigger
REAL_B, SAVE :: XDTHPBL     ! temperature perturbation in PBL for trigger
REAL_B, SAVE :: XDRVPBL     ! moisture perturbation in PBL for trigger


REAL_B, SAVE :: XNHGAM      ! accounts for non-hydrost. pressure
                            ! in buoyancy term of w equation
                            ! = 2 / (1+gamma)
REAL_B, SAVE :: XTFRZ1      ! begin of freezing interval
REAL_B, SAVE :: XTFRZ2      ! end of freezing interval

REAL_B, SAVE :: XRHDBC      ! relative humidity below cloud in downdraft

REAL_B, SAVE :: XRCONV      ! constant in precipitation conversion
REAL_B, SAVE :: XSTABT      ! factor to assure stability in  fractional time
                            ! integration, routine CONVECT_CLOSURE
REAL_B, SAVE :: XSTABC      ! factor to assure stability in CAPE adjustment,
                            !  routine CONVECT_CLOSURE
REAL_B, SAVE :: XUSRDPTH    ! pressure thickness used to compute updraft
                            ! moisture supply rate for downdraft
REAL_B, SAVE :: XMELDPTH    ! layer (Pa) through which precipitation melt is
                            ! allowed below  melting level
REAL_B, SAVE :: XUVDP       ! constant for pressure perturb in momentum transport

END MODULE YOE_CONVPAR

!-------------------------------------------------------------------------------

