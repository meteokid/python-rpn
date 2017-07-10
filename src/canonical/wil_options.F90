!---------------------------------- LICENCE BEGIN -------------------------------
! GEM - Library of kernel routines for the GEM numerical atmospheric model
! Copyright (C) 1990-2010 - Division de Recherche en Prevision Numerique
!                       Environnement Canada
! This library is free software; you can redistribute it and/or modify it
! under the terms of the GNU Lesser General Public License as published by
! the Free Software Foundation, version 2.1 of the License. This library is
! distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
! without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
! PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
! You should have received a copy of the GNU Lesser General Public License
! along with this library; if not, write to the Free Software Foundation, Inc.,
! 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
!---------------------------------- LICENCE END ---------------------------------
module wil_options
   implicit none
   public
   save

   !# Williamson case selector
   !# * Williamson_case=0 (none)
   !# * Williamson_case=1 (Advection 2D - Williamson_NAIR/Terminator_L)
   !# * Williamson_case=2 (Steady-state nonlinear zonal geostrophic flow - Williamson et al.,1992,JCP,102,211-224)
   !# * Williamson_case=5 (Zonal flow over an isolated mountain - Williamson et al.,1992,JCP,102,211-224)
   !# * Williamson_case=6 (Rossby-Haurwitz wave - Williamson et al.,1992,JCP,102,211-224)
   !# * Williamson_case=7 (The 21 December 1978 case - Williamson et al.,1992,JCP,102,211-224)
   !# * Williamson_case=8 (Galewsky's barotropic wave - Galewsky et al.,2004,Tellus,56A,429-440)
   integer :: Williamson_case = 0
   namelist /williamson/ Williamson_case

   !# Used when Williamson_case=1
   !# * Williamson_NAIR=0 (Solid body rotation of a cosine bell - Williamson et al.,1992,JCP,102,211-224)
   !# * Williamson_NAIR=1 (Deformational Non-divergent winds - Lauritzen et al.,2012,GMD,5,887-901)
   !# * Williamson_NAIR=2 (Deformational divergent winds - Lauritzen et al.,2012,GMD,5,887-901)
   !# * Williamson_NAIR=3 (Deformational Flow for Circular vortex - Nair and Machenhauer,2002,MWR,130,649-667)
   integer :: Williamson_NAIR = 0
   namelist /williamson/ Williamson_NAIR

   !# Set lower value of Tracer in Terminator
   !# * Williamson_lower_value=0 (free)
   !# * Williamson_lower_value=1 (0)
   !# * Williamson_lower_value=2 (1.0e-15)
   integer :: Williamson_lower_value = 0
   namelist /williamson/ Williamson_lower_value

   !# Do Terminator chemistry if T
   logical :: Williamson_Terminator_L = .false.
   namelist /williamson/ Williamson_Terminator_L

   !# Rotation angle  in DEGREE - W_NAIR=0)
   real :: Williamson_alpha = 0.
   namelist /williamson/ Williamson_alpha

   !# LON cosine Bell  in DEGREE - W_NAIR=0)
   real :: Williamson_clon0 = 270. ! 3*pi/2 rad in the paper
   namelist /williamson/ Williamson_clon0

   !# LAT cosine Bell  in DEGREE - W_NAIR=0)
   real :: Williamson_clat0 = 0.
   namelist /williamson/ Williamson_clat0

   !# rotation period in SECS - W_NAIR=0)
   real :: Williamson_period = 12.*24.*3600.
   namelist /williamson/ Williamson_period

   !# LON rot.POLE  in DEGREE - W_NAIR=0/3)
   real :: Williamson_rlon0 = 0.
   namelist /williamson/ Williamson_rlon0

   !# LAT rot.POLE  in DEGREE - W_NAIR=3)
   real :: Williamson_rlat0 = 0.
   namelist /williamson/ Williamson_rlat0

   real*8 :: Williamson_lon_pole_r_8, Williamson_lat_pole_r_8, Williamson_rho_i_8, &
             Williamson_v0_8, Williamson_ubar_8

contains

   function wil_options_init() result(F_istat)
      implicit none
      integer :: F_istat
#include <rmnlib_basics.hf>
      logical, save :: init_L = .false.
      F_istat = RMN_OK
      if (init_L) return
      init_L = .true.

      return
   end function wil_options_init

end module wil_options
