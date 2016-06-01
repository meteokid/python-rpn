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
! v4_80 - Tanguay M.        - GEM4 Mass-Conservation 

#include "msg.h"
#include "adv_nml.cdk"
#include "grd.cdk"
#include "adv_grid.cdk"

      integer :: fileUnit
      integer, external :: file_open_existing
!
!---------------------------------------------------------------------
!
      
      adv_rhst_mono_L  = .false.
      adv_catmullrom_L = .false.
      adv_BC_min_max_L = .true.
      adv_halox  = -1
      adv_haloy  = -1
      adv_ILMC_sweep_max = 2
      adv_ILMC_min_max_L = .true.
      adv_SLICE_rebuild  = 2 
      adv_verbose        = 0
      adv_pil_sub_s = -1
      adv_pil_sub_n = -1
      adv_pil_sub_w = -1
      adv_pil_sub_e = -1

      F_stat = -1
      fileUnit = file_open_existing(F_nmlFileName_S,'SEQ')
      if (fileUnit >= 0) then
         read(fileUnit, nml=adv_cfgs, iostat=F_stat)
         call fclos(fileUnit)
         F_stat = -1 * F_stat
         if (F_stat<0) then
            call msg(MSG_ERROR,'adv_nml - Probleme reading nml adv_cfgs in file: '//trim(F_nmlFileName_S))
         endif
      endif

      adv_maxcfl= max(1,Grd_maxcfl)
      adv_halox = max(1,adv_maxcfl + 1)
      adv_haloy = adv_halox
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
      namelist /adv_cfgs/ adv_ILMC_min_max_L

!---------------------------------------------------------------------
      msgUnit = msg_getUnit(MSG_INFO)
      if (msgUnit>=0) write(msgUnit,nml=adv_cfgs)
!---------------------------------------------------------------------
      return
      end subroutine adv_nml_print
