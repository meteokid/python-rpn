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

!**s/p adv_check_tracers: Check if extended advection operations required 

      subroutine adv_check_tracers ()

      implicit none

#include <arch_specific.hf>

      !@author Monique Tanguay

      !@revisions
      ! v4_80 - Tanguay M.        - GEM4 Mass-Conservation

#include "tr3d.cdk"
#include "tracers.cdk"
#include "adv_nml.cdk"
#include "glb_ld.cdk"
#include "grd.cdk"
#include "schm.cdk"
#include "lun.cdk"

      !---------------------------------------------------------------------
      integer n
      logical qw_L
      !---------------------------------------------------------------------

      Tr_SLICE_rebuild  = Adv_SLICE_rebuild

      Tr_BC_min_max_L   = Adv_BC_min_max_L

      Tr_ILMC_min_max_L = Adv_ILMC_min_max_L
      Tr_ILMC_sweep_max = Adv_ILMC_sweep_max

      Tr_verbose = Adv_verbose

      Tr_dry_mixing_ratio_L = .false.   

      Tr_flux_L  = .false.
      Tr_slice_L = .false.

      if (Schm_psadj_L.and.G_lam.and..not.Grd_yinyang_L) Tr_flux_L = .true. !PSADJ (FLUX) 

      do n=1,Tr3d_ntr

         qw_L= Tr3d_wload(n) .or. Tr3d_name_S(n)(1:2).eq.'HU'

         if (qw_L) cycle

         if (G_lam.and..not.Grd_yinyang_L) then !LAM

            if (Tr3d_mass(n)==1) Tr_flux_L  = .true. !BC (FLUX)   
            if (Tr3d_mass(n)==2) Tr_slice_L = .true. !SLICE 

         endif

      end do

      Tr_extension_L = Tr_flux_L.or.Tr_slice_L

      if (Tr_extension_L.and.Lun_out>0) then
         write(Lun_out,*) ''
         write(Lun_out,*) 'ADV_CHECK_EXT: EXTENDED ADVECTION OPERATIONS REQUIRED'
         write(Lun_out,*) ''
      endif
 
      !If not initialized by namelist adv_cfgs
      !---------------------------------------
      if (adv_pil_sub_s == -1) then
         Tr_pil_sub_s = pil_s
         Tr_pil_sub_n = pil_n
         Tr_pil_sub_w = pil_w
         Tr_pil_sub_e = pil_e
      else
         Tr_pil_sub_s = adv_pil_sub_s
         Tr_pil_sub_n = adv_pil_sub_n
         Tr_pil_sub_w = adv_pil_sub_w
         Tr_pil_sub_e = adv_pil_sub_e
         if (.NOT.l_north) Tr_pil_sub_n = 0
         if (.NOT.l_south) Tr_pil_sub_s = 0
         if (.NOT.l_west ) Tr_pil_sub_w = 0
         if (.NOT.l_east ) Tr_pil_sub_e = 0
      endif

      !---------------------------------------------------------------------

      return
end subroutine adv_check_tracers
