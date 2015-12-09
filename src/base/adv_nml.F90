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

      function adv_nml (F_nmlFileName_S) result (F_stat)
      implicit none
#include <arch_specific.hf>

!@objective Advection default configuration and reading namelists adv_cfgs
!@arguments
      character(len=*) :: F_nmlFileName_S
!@returns
      integer :: F_stat
!@author Stephane Chamberland, Nov 2009
!@revisions
! v4_XX - Tanguay M.        - GEM4 Mass-Conservation 

#include "msg.h"
#include "adv_nml.cdk"
#include "grd.cdk"

      integer :: fileUnit
      integer, external :: file_open_existing
!
!---------------------------------------------------------------------
!
      adw_mono_L       = .true.
      adw_catmullrom_L = .false.
      adw_BC_min_max_L = .true.
      adw_halox  = -1
      adw_haloy  = -1
      adw_ILMC_sweep_max = 2
      adw_ILMC_min_max_L = .true.
      Adw_reconstruction = 1
      Adw_PPM_mono       = 0
      Adw_verbose        = 0
      pil_sub_s = -1
      pil_sub_n = -1
      pil_sub_w = -1
      pil_sub_e = -1

      F_stat = -1
      fileUnit = file_open_existing(F_nmlFileName_S,'SEQ')
      if (fileUnit >= 0) then
         read(fileUnit, nml=adw_cfgs, iostat=F_stat)
         call fclos(fileUnit)
         F_stat = -1 * F_stat
         if (F_stat<0) then
            call msg(MSG_ERROR,'adw_nml - Probleme reading nml adw_cfgs in file: '//trim(F_nmlFileName_S))
         endif
      endif

      adw_maxcfl= max(1,Grd_maxcfl)
      adw_halox = max(1,adw_maxcfl + 1)
      adw_haloy = adw_halox
!
!---------------------------------------------------------------------
!
      return
      end function adv_nml

      subroutine adv_nml_print()
      implicit none
#include <arch_specific.hf>

#include "msg.h"
#include "adv_nml.cdk"

      integer :: msgUnit
      integer, external :: msg_getUnit
      namelist /adw_cfgs/ adw_ILMC_min_max_L

!---------------------------------------------------------------------
      msgUnit = msg_getUnit(MSG_INFO)
      if (msgUnit>=0) write(msgUnit,nml=adw_cfgs)
!---------------------------------------------------------------------
      return
      end subroutine adv_nml_print
