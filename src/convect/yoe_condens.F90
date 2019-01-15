!     ##################
      MODULE YOE_CONDENS
!     ##################

!!****  *YOE_CONDENS * -  constants module
!!
!!    PURPOSE
!!    -------
!       The module contains constants used in routine CONDENS.
!!
!!**  METHOD
!!    ------
!!      The "condensation" constants are set to their numerical values
!!
!!
!!    EXTERNAL
!!    --------
!!
!!    IMPLICIT ARGUMENTS
!!    ------------------
!!
!!
!!    REFERENCE
!!    ---------
!!      Book2 of the documentation (module YOE_CONDENS)
!!
!!
!!    AUTHOR
!!    ------
!!      P. Marquet   *Meteo-France CNRM/GMGEC/EAC*
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    07/06/2002
!!   Last modified  07/06/2002
!-------------------------------------------------------------------------------

#include "tsmbkind.cdk"

!*       0.    DECLARATIONS
!              ------------


IMPLICIT NONE

!-------------------------------------------------------------------------------

!*       1.    Set the thermodynamical and numerical constants for
!              the deep convection parameterization
!              ---------------------------------------------------


REAL_B :: XCTFRZ1    ! begin of freezing interval
REAL_B :: XCTFRZ2    ! end of freezing interval

REAL_B :: XCLSGMAX   ! Max-tropospheric length for Sigma_s 
REAL_B :: XCLSGMIN   ! Lower-height Min-length for Sigma_s

REAL_B :: XCSIGMA    ! constant in sigma_s parameterization

REAL_B :: XCSIG_CONV ! scaling factor for ZSIG_CONV as
                     ! function of mass flux

END MODULE YOE_CONDENS
