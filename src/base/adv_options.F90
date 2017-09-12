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
module adv_options
   implicit none
   public
   save

   !# switch:.T.:MONO(CLIPPING) of RHS
   logical :: adv_rhst_mono_L = .false.
   namelist /adv_cfgs/ adv_rhst_mono_L

   !# switch:.T.:
   logical :: adv_catmullrom_L = .false.
   namelist /adv_cfgs/ adv_catmullrom_L

   !# switch:.T.:MONO(CLIPPING) after Bermejo-Conde
   !# (MASS-CONSERVATION for Chemical Tracers)
   logical :: adv_BC_min_max_L = .true.
   namelist /adv_cfgs/ adv_BC_min_max_L

   !# Number of neighborhood zones in ILMC
   integer :: adv_ILMC_sweep_max = 2
   !# (MASS-CONSERVATION for Chemical Tracers)
   namelist /adv_cfgs/ adv_ILMC_sweep_max

   !# switch:.T.:MONO(CLIPPING) after ILMC
   logical :: adv_ILMC_min_max_L = .true.
   !# (MASS-CONSERVATION for Chemical Tracers)
   namelist /adv_cfgs/ adv_ILMC_min_max_L

   !# Type of rebuild in SLICE
   !# * SLICE_rebuild=1 (LP)
   !# * SLICE_rebuild=2 (CW)
   !# (MASS-CONSERVATION for Chemical Tracers)
   integer :: adv_SLICE_rebuild = 2
   namelist /adv_cfgs/ adv_SLICE_rebuild

   !# Activate conservation diagnostics if /=0
   !# (MASS-CONSERVATION for Chemical Tracers)
   integer :: adv_verbose = 0
   namelist /adv_cfgs/ adv_verbose

   !# South boundary in GY for an embedded LAM
   !# (MASS-CONSERVATION for Chemical Tracers)
   integer :: adv_pil_sub_s = -1
   namelist /adv_cfgs/ adv_pil_sub_s

   !# North boundary in GY for an embedded LAM
   !# (MASS-CONSERVATION for Chemical Tracers)
   integer :: adv_pil_sub_n = -1
   namelist /adv_cfgs/ adv_pil_sub_n

   !# West boundary in GY for an embedded LAM
   !# (MASS-CONSERVATION for Chemical Tracers)
   integer :: adv_pil_sub_w = -1
   namelist /adv_cfgs/ adv_pil_sub_w

   !# East boundary in GY for an embedded LAM
   !# (MASS-CONSERVATION for Chemical Tracers)
   integer :: adv_pil_sub_e = -1
   namelist /adv_cfgs/ adv_pil_sub_e

   !# Scaling for mass of tracer
   !# * adv_scaling=0 (none)
   !# * adv_scaling=1 (CO2)
   !# * adv_scaling=2 (O3)
   integer :: adv_scaling = 1
   namelist /adv_cfgs/ adv_scaling

contains

   function adv_options_init() result(F_istat)
      implicit none
      integer :: F_istat
#include <rmnlib_basics.hf>
      logical, save :: init_L = .false.
      F_istat = RMN_OK
      if (init_L) return
      init_L = .true.

      return
   end function adv_options_init

end module adv_options
