!     ###########################
      SUBROUTINE SU_CONVPAR_SHAL
!     ###########################

!!****  *SU_CONVPAR * - routine to initialize the constants modules
!!
!!    PURPOSE
!!    -------
!!       The purpose of this routine is to initialize  the constants
!!     stored in  modules YOE_CONVPAR_SHAL
!!
!!
!!**  METHOD
!!    ------
!!      The shallow convection constants are set to their numerical values
!!
!!
!!    EXTERNAL
!!    --------
!!
!!    IMPLICIT ARGUMENTS
!!    ------------------
!!      Module YOE_CONVPAR_SHAL   : contains deep convection constants
!!
!!    REFERENCE
!!    ---------
!!      Book2 of the documentation (module YOE_CONVPAR_SHAL, routine SU_CONVPAR)
!!
!!
!!    AUTHOR
!!    ------
!!      P. BECHTOLD       * Laboratoire d'Aerologie *
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    26/03/96
!!   Last modified  15/04/98 adapted for ARPEGE
!-------------------------------------------------------------------------------


!*       0.    DECLARATIONS
!              ------------

USE YOE_CONVPAR_SHAL
#include "tsmbkind.cdk"

IMPLICIT NONE
#include <arch_specific.hf>

!-------------------------------------------------------------------------------

!*       1.    Set the thermodynamical and numerical constants for
!              the deep convection parameterization
!              ---------------------------------------------------


XA25        = 625.E6_JPRB ! 25 km x 25 km reference grid area

XCRAD       =  50._JPRB   ! cloud radius (m)
XCTIME_SHAL = 10800._JPRB ! convective adjustment time (s)
XCDEPTH     = 0.5E3_JPRB  ! minimum necessary shallow cloud depth
XCDEPTH_D   = 3.0E3_JPRB  ! maximum allowed shallow cloud depth (m)
XDTPERT     = .2_JPRB     ! add small Temp perturbation at LCL (K)
XENTR    = 0.03_JPRB      ! entrainment constant (m/Pa) = 0.2 (m)

XZLCL    = 0.5E3_JPRB     ! maximum allowed height (m)
                          ! difference between the DPL and the surface
XZPBL    = 40.E2_JPRB     ! minimum mixed layer depth to sustain convection


XNHGAM   = 1.3333_JPRB    ! accounts for non-hydrost. pressure
                          ! in buoyancy term of w equation
                          ! = 2 / (1+gamma)
XTFRZ1   = 273.16_JPRB    ! begin of freezing interval (K)
XTFRZ2   = 250.16_JPRB    ! end of freezing interval (K)


XSTABT   = 0.95_JPRB      ! factor to assure stability in  fractional time
                          ! integration, routine CONVECT_CLOSURE
XSTABC   = 0.95_JPRB      ! factor to assure stability in CAPE adjustment,
                          !  routine CONVECT_CLOSURE


END SUBROUTINE SU_CONVPAR_SHAL

