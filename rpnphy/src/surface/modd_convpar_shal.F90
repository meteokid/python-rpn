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
!
!     ########################
      MODULE MODD_CONVPAR_SHAL
!     ########################
!
!!****  *MODD_CONVPAR_SHAL* - Declaration of convection constants 
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
!!      Book2 of documentation of Meso-NH (MODD_CONVPAR_SHAL)
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
!
!*       0.   DECLARATIONS
!             ------------
!
implicit none
!
REAL, SAVE :: XA25        ! 25 km x 25 km reference grid area
!
REAL, SAVE :: XCRAD       ! cloud radius 
REAL, SAVE :: XCTIME_SHAL ! convective adjustment time
REAL, SAVE :: XCDEPTH     ! minimum necessary cloud depth
REAL, SAVE :: XCDEPTH_D   ! maximum allowed cloud thickness
REAL, SAVE :: XDTPERT     ! add small Temp perturb. at LCL
REAL, SAVE :: XENTR       ! entrainment constant (m/Pa) = 0.2 (m)  
!
REAL, SAVE :: XZLCL       ! maximum allowed allowed height 
                          ! difference between departure level and surface
REAL, SAVE :: XZPBL       ! minimum mixed layer depth to sustain convection
REAL, SAVE :: XWTRIG      ! constant in vertical velocity trigger
!
!
REAL, SAVE :: XNHGAM      ! accounts for non-hydrost. pressure 
			  ! in buoyancy term of w equation
                          ! = 2 / (1+gamma)
REAL, SAVE :: XTFRZ1      ! begin of freezing interval
REAL, SAVE :: XTFRZ2      ! end of freezing interval
!
!
REAL, SAVE :: XSTABT      ! factor to assure stability in  fractional time
                          ! integration, routine CONVECT_CLOSURE
REAL, SAVE :: XSTABC      ! factor to assure stability in CAPE adjustment,
                          !  routine CONVECT_CLOSURE
!
END MODULE MODD_CONVPAR_SHAL 
!
