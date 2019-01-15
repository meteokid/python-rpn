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

!**s/p adx_check_tracers: Check if extended advection operations required 

      subroutine adx_check_tracers ()

      implicit none

#include <arch_specific.hf>

      !@author Monique Tanguay

      !@revisions
      ! v4_80 - Tanguay M.        - GEM4 Mass-Conservation
      ! v4_87 - Tanguay M.        - Add Tr_scaling and Schm_dry_mixing_ratio

#include "tr3d.cdk"
#include "tracers.cdk"
#include "adx_nml.cdk"
#include "glb_ld.cdk"
#include "grd.cdk"
#include "schm.cdk"
#include "lun.cdk"

      !---------------------------------------------------------------------
      integer n
      !---------------------------------------------------------------------

      Tr_BC_min_max_L   = Adw_BC_min_max_L

      Tr_ILMC_min_max_L = Adw_ILMC_min_max_L
      Tr_ILMC_sweep_max = Adw_ILMC_sweep_max

      Tr_verbose = Adw_verbose

      Tr_flux_L = .false.
      Tr_core_L = .false.

      if (Schm_psadj>0.and.G_lam.and..not.Grd_yinyang_L) Tr_flux_L = .true. !PSADJ (FLUX) 

      Tr_scaling = Adw_scaling

      if (Tr_scaling<0.or.Tr_scaling>2) call handle_error(-1,'ADX_CHECK_TRACERS','TR_SCALING not defined')

      do n=1,Tr3d_ntr

         if (G_lam.and..not.Grd_yinyang_L) then !LAM

            if (Tr3d_mass(n)==1) Tr_flux_L = .true. !BC (FLUX)   

            if (Tr3d_mass(n)/=0) Tr_core_L = .true. 
            if (Tr3d_mono(n)/=0) Tr_core_L = .true. 

         endif

      end do

      Tr_extension_L = Tr_flux_L

      if (Tr_extension_L.and.Grd_yinyang_L) call handle_error(-1,'ADX_CHECK_TRACERS','TR_EXTENSION YIN-YANG not defined')

      if (Tr_extension_L.and.Lun_out>0) then
         write(Lun_out,*) ''
         write(Lun_out,*) 'ADX_CHECK_EXT: EXTENDED ADVECTION OPERATIONS REQUIRED'
         write(Lun_out,*) ''
      endif

      !If not initialized by namelist adw_cfgs
      !---------------------------------------
      if (adw_pil_sub_s == -1) then
         Tr_pil_sub_s = pil_s
         Tr_pil_sub_n = pil_n
         Tr_pil_sub_w = pil_w
         Tr_pil_sub_e = pil_e
      else
         Tr_pil_sub_s = adw_pil_sub_s
         Tr_pil_sub_n = adw_pil_sub_n
         Tr_pil_sub_w = adw_pil_sub_w
         Tr_pil_sub_e = adw_pil_sub_e
      endif

      !---------------------------------------------------------------------

      return
end subroutine adx_check_tracers
