!     #######################
      SUBROUTINE SU_CONVPAR1
!     #######################

!!****  *SU_CONVPAR * - routine to initialize the constants modules
!!
!!    PURPOSE
!!    -------
!       The purpose of this routine is to initialize  the constants
!     stored in  modules YOE_CONVPAR


!!**  METHOD
!!    ------
!!      The deep convection constants are set to their numerical values
!!
!!
!!    EXTERNAL
!!    --------
!!
!!    IMPLICIT ARGUMENTS
!!    ------------------
!!      Module YOE_CONVPAR   : contains deep convection constants
!!
!!    REFERENCE
!!    ---------
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

USE YOE_CONVPAR
#include "tsmbkind.cdk"

IMPLICIT NONE
#include <arch_specific.hf>

!-------------------------------------------------------------------------------

!*       1.    Set the thermodynamical and numerical constants for
!              the deep convection parameterization
!              ---------------------------------------------------


XA25     = 625.E6_JPRB    ! 25 km x 25 km reference grid area

XCRAD    =  500._JPRB     ! cloud radius (m)
!XCRAD    = 1500._JPRB     ! cloud radius
XCDEPTH  = 3.E3_JPRB      ! minimum necessary cloud depth
XENTR    = 0.03_JPRB      ! entrainment constant (m/Pa) = 0.2 (m)

XZLCL    = 3.5E3_JPRB     ! maximum allowed allowed height
                          ! difference between the surface and the DPL (m)
XZPBL    = 60.E2_JPRB     ! minimum mixed layer depth to sustain convection
XWTRIG   = 6.00_JPRB      ! constant in vertical velocity trigger
XDTHPBL  = .1_JPRB        ! Temp. perturbation in PBL for trigger (K)
XDRVPBL  = 1.e-4_JPRB     ! moisture  perturbation in PBL for trigger (kg/kg)


XNHGAM   = 1.3333_JPRB    ! accounts for non-hydrost. pressure
                          ! in buoyancy term of w equation
                          ! = 2 / (1+gamma)
XTFRZ1   = 273.16_JPRB    ! begin of freezing interval (K)
XTFRZ2   = 250.16_JPRB    ! end of freezing interval (K)

XRHDBC   = 0.9_JPRB       ! relative humidity below cloud in downdraft

XRCONV   = 0.015_JPRB     ! constant in precipitation conversion
XRCONV   = 0.01_JPRB     ! constant in precipitation conversion
XSTABT   = 0.75_JPRB      ! factor to assure stability in  fractional time
                          ! integration, routine CONVECT_CLOSURE
XSTABC   = 0.95_JPRB      ! factor to assure stability in CAPE adjustment,
                          !  routine CONVECT_CLOSURE
XUSRDPTH = 165.E2_JPRB    ! pressure thickness used to compute updraft
                          ! moisture supply rate for downdraft
XMELDPTH = 200.E2_JPRB    ! layer (Pa) through which precipitation melt is
                          ! allowed below downdraft
XUVDP    = 0.7_JPRB       ! constant for pressure perturb in momentum transport
!
!
END SUBROUTINE SU_CONVPAR1

