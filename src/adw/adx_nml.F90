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
!/@*
function adx_nml(F_nmlFileName_S) result(F_stat)
   implicit none
#include <arch_specific.hf>
#include "msg.h"
#include "adx_nml.cdk"
   !@objective Advection default configuration and reading namelists adx_cfgs
   !@arguments
   character(len=*) :: F_nmlFileName_S
   !@returns
   integer :: F_stat
   !@author Stephane Chamberland, Nov 2009
   !@revisions
   ! v4_80 - Tanguay M.        - GEM4 Mass-Conservation 
!*@/
   integer :: fileUnit
   integer, external :: file_open_existing
   !---------------------------------------------------------------------
   adw_nkbz_L = .true.
   adw_exdg_L = .false.
   adw_ckbd_L = .false.
   adw_mono_L = .true.
   adw_catmullrom_L = .false.
   adw_positive_L = .false.
   adw_trunc_traj_L = .false.
   adw_halox  = -1
   adw_haloy  = -1
   adw_stats_L = .false.
   adw_BC_min_max_L   = .true.
   adw_ILMC_sweep_max = 2
   adw_ILMC_min_max_L = .true.
   adw_verbose = 0 
   adw_scaling = 1 
   adw_pil_sub_s = -1
   adw_pil_sub_n = -1
   adw_pil_sub_w = -1
   adw_pil_sub_e = -1

   F_stat = -1
   fileUnit = file_open_existing(F_nmlFileName_S,'SEQ')
   if (fileUnit>=0) then
      read(fileUnit, nml=adw_cfgs, iostat=F_stat)
      call fclos(fileUnit)
      F_stat = -1 * F_stat
      if (F_stat<0) then
         call msg(MSG_ERROR,'adw_nml - Probleme reading nml adw_cfgs in file: '//trim(F_nmlFileName_S))
      endif
   endif
   !TODO: fix adw_stat_L then remove the following warning/override
   if (adw_stats_L) then
      adw_stats_L = .false.
      call msg(MSG_WARNING,'adw_nml - adw_stats_L not supported -- ignored')
   endif
   !---------------------------------------------------------------------
   return
end function adx_nml


!/@*
subroutine adx_nml_print()
   implicit none
#include <arch_specific.hf>
#include "msg.h"
#include "adx_nml.cdk"
!*@/
   integer :: msgUnit
   integer, external :: msg_getUnit !TODO: get full interface from <msg.hf>
   !---------------------------------------------------------------------
   msgUnit = msg_getUnit(MSG_INFO)
   if (msgUnit>=0) write(msgUnit,nml=adw_cfgs)
   !---------------------------------------------------------------------
   return
end subroutine adx_nml_print


!/@*
function adx_config() result(F_stat)
   implicit none
#include <arch_specific.hf>
   !@objective Establish final Advection configuration
   !@returns
   integer :: F_stat
   !@author Stephane Chamberland, Nov 2009
#include "glb_ld.cdk"
#include "grd.cdk"
#include "adx_nml.cdk"
#include "adx_dims.cdk"
!*@/
   !---------------------------------------------------------------------
   F_stat = 0

   adw_maxcfl= max(1,Grd_maxcfl)
   adw_halox = max(1,adw_maxcfl + 1)
   adw_haloy = adw_halox

   if (.not.G_lam) then
      adw_halox = max(3,adw_halox)
      adw_haloy = max(2,adw_haloy)
   endif

   if (adw_catmullrom_L .and. .not.G_lam) then
      call msg(MSG_ERROR,'adw_nml: adw_catmullrom_L is supported only in a LAM')
      F_stat = -1
   end if

   if (adw_positive_L .and. adw_mono_L) then
      call msg(MSG_ERROR,'adw_nml: adw_positive_L and adw_mono_L cannot be both .true.')
      F_stat = -1
   endif
   !---------------------------------------------------------------------
   return
end function adx_config
