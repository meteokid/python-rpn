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
module dcmip_options
   implicit none
   public
   save
   
   !# Dcmip case selector
   !# * Dcmip_case=0 (none)
   !# * Dcmip_case=11 (3D deformational flow)
   !# * Dcmip_case=12 (3D Hadley-like meridional circulation)
   !# * Dcmip_case=13 (2D solid-body rotation of thin cloud-like tracer in the presence of orography)
   !# * Dcmip_case=20 (Steady-state at rest in presence of oro.)
   !# * Dcmip_case=21 (Mountain waves over a Schaer-type mountain)
   !# * Dcmip_case=22 (As 21 but with wind shear)
   !# * Dcmip_case=31 (Gravity wave along the equator)
   !# * Dcmip_case=41 (Dry Baroclinic Instability Small Planet)
   !# * Dcmip_case=43 (Moist Baroclinic Instability Simple physics)
   !# * Dcmip_case=161 (Baroclinic wave with Toy Terminal Chemistry)
   !# * Dcmip_case=162 (Tropical cyclone)
   !# * Dcmip_case=163 (Supercell -Small Planet-)
   integer :: Dcmip_case = 0
   namelist /dcmip/ Dcmip_case
   
   !# Type of precipitation/microphysics
   !# * Dcmip_prec_type=-1 (none)
   !# * Dcmip_prec_type=0 (Large-scale precipitation -Kessler-)
   !# * Dcmip_prec_type=1 (Large-scale precipitation -Reed-Jablonowski-)
   integer :: Dcmip_prec_type = -1
   namelist /dcmip/ Dcmip_prec_type

   !# Type of planetary boundary layer
   !# * Dcmip_pbl_type=-1 (none)
   !# * Dcmip_pbl_type=0 (Reed-Jablonowski Boundary layer)
   !# * Dcmip_pbl_type=1 (Georges Bryan Planetary Boundary Layer)
   integer :: Dcmip_pbl_type = -1
   namelist /dcmip/ Dcmip_pbl_type

   !# Account for moisture
   !# * Dcmip_moist=0 (dry)
   !# * Dcmip_moist=1 (moist)
   integer :: Dcmip_moist = 1
   namelist /dcmip/ Dcmip_moist

   !# Set lower value of Tracer in Terminator 
   !# * Dcmip_lower_value=0 (free)
   !# * Dcmip_lower_value=1 (0)
   !# * Dcmip_lower_value=2 (1.0e-15)
   integer :: Dcmip_lower_value = 0
   namelist /dcmip/ Dcmip_lower_value

   !# Do Terminator chemistry if T
   logical :: Dcmip_Terminator_L = .false.
   namelist /dcmip/ Dcmip_Terminator_L

   !# Do Rayleigh friction if T
   logical :: Dcmip_Rayleigh_friction_L = .false.
   namelist /dcmip/ Dcmip_Rayleigh_friction_L

   !# Vertical Diffusion Winds (if <0,we remove REF)
   real :: Dcmip_nuZ_wd = 0.
   namelist /dcmip/ Dcmip_nuZ_wd

   !# Vertical Diffusion Theta (if <0,we remove REF)
   real :: Dcmip_nuZ_th = 0.
   namelist /dcmip/ Dcmip_nuZ_th

   !# Vertical Diffusion Tracers (if <0,we remove REF)
   real :: Dcmip_nuZ_tr = 0.
   namelist /dcmip/ Dcmip_nuZ_tr
   
   !# Earth's radius reduction factor
   real*8  :: Dcmip_X = 1.d0
   namelist /dcmip/ Dcmip_X

   logical :: Dcmip_vrd_L
contains

   function dcmip_options_init() result(F_istat)
      implicit none 
      integer :: F_istat
#include <rmnlib_basics.hf>
      logical, save :: init_L = .false.
      F_istat = RMN_OK
      if (init_L) return
      init_L = .true.
      
      return
   end function dcmip_options_init

end module dcmip_options
