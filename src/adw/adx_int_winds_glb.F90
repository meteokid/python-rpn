!COMP_ARCH=intel13sp1u2 ; -suppress=-C
!COMP_ARCH=intel-2016.1.156; -suppress=-C
!COMP_ARCH=PrgEnv-intel-5.2.82 ; -suppress=-C

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
#include "constants.h"
#include "msg.h"

!/@*
subroutine adx_int_winds_glb(F_wrkx1,F_wrky1,F_u1,F_u2,F_xth,F_yth,F_zth, &
     F_dth, F_has_u2_L, F_i0,F_in,F_j0,F_jn, &
     F_ni,F_nj,F_aminx, F_amaxx, F_aminy, F_amaxy,F_k0,F_nk, F_nk_super)
   implicit none
#include <arch_specific.hf>
   !@objective
   !@arguments
   logical :: F_has_u2_L           !I, .T. if F_u2 needs to be treated
   real    :: F_dth                !I, factor (1. or timestep)
   integer :: F_nk, F_nk_super     !I, number of vertical levels
   integer :: F_aminx, F_amaxx, F_aminy, F_amaxy !I, wind fields array bounds
   integer :: F_ni, F_nj           !I, dims of position fields
   integer :: F_i0,F_in,F_j0,F_jn,F_k0 !I, operator scope
   real, dimension(F_ni,F_nj,F_nk) :: F_xth,F_yth !I/O, x,y positions
   real, dimension(F_ni,F_nj,F_nk) :: F_zth       !I, z positions
   real, dimension(F_aminx:F_amaxx,F_aminy:F_amaxy,F_nk_super) :: &
        F_u1,F_u2   !I, field to interpol
   real, dimension(F_ni,F_nj,F_nk) :: F_wrkx1,F_wrky1  !O, F_dt * result of interp
   !@author Stephane Chamberland
   !@revisions
   ! v4_40 - Tanguay M.        - Revision TL/AD
   ! v4_XX - Tanguay M.        - GEM4 Mass-Conservation
   !*@/
#include "adx_nml.cdk"
#include "adx_poles.cdk"
#include "ptopo.cdk"
#include "lun.cdk"
#include "lctl.cdk"
#include "orh.cdk"
#include "schm.cdk"
#include "step.cdk"

   integer, external :: adx_ckbd3

   integer :: nb_flds, istat,outside,sum_outside,ier
   real    :: dummy
   integer, dimension(F_ni,F_nj,F_nk) :: wrkc1
   real,    dimension(F_ni,F_nj,F_nk) :: wrkz1, wrky
   real,    dimension(:,:,:), allocatable :: xpos2,ypos2,zpos2
   !---------------------------------------------------------------------
   call msg(MSG_DEBUG,'adx_int_winds_glb [begin]')

   call adx_exch_1c(F_wrkx1,F_wrky1,wrkz1,wrkc1,F_xth,F_yth,F_zth,F_ni,F_nj,F_k0,F_nk)

   allocate( &
        xpos2(max(1,adx_fro_a),1,1), &
        ypos2(max(1,adx_fro_a),1,1), &
        zpos2(max(1,adx_fro_a),1,1), &
        stat=istat)
   call handle_error_l(istat==0,'adx_main_2_pos/adx_pos','Problem allocating mem')

   call adx_exch_2(xpos2, ypos2, zpos2, dummy, dummy, F_wrkx1, F_wrky1, wrkz1, dummy, dummy, &
        adx_fro_n, adx_fro_s, adx_fro_a, &
        adx_for_n, adx_for_s, adx_for_a, 3)

   call adx_trilin5(F_wrkx1,F_wrky1,F_u1,F_u2,F_xth,F_yth,F_zth, &
        F_dth, F_has_u2_L, F_i0,F_in,F_j0,F_jn, &
        F_ni,F_nj,F_aminx, F_amaxx, F_aminy, F_amaxy,F_k0,F_nk, F_nk_super)

   istat = 1
   if (adx_fro_a > 0 .and. adw_ckbd_L) istat = adx_ckbd3(ypos2,adx_fro_n,adx_fro_s)
   call handle_error(istat,'adx_int_winds_glb','Error raised in adx_ckbd')

   if (adx_fro_a > 0) then
      call adx_trilin5(xpos2,ypos2,F_u1,F_u2,xpos2,ypos2,zpos2, &
           F_dth,F_has_u2_L, 1,adx_fro_a,1,1,&
           adx_fro_a,1,F_aminx, F_amaxx, F_aminy, F_amaxy,1,1, F_nk_super)
   endif

   nb_flds = 1
   if (F_has_u2_L) nb_flds = 2

   call adx_exch_2(wrkz1, wrky, dummy, dummy, dummy, xpos2, ypos2, dummy, dummy, dummy, &
        adx_for_n, adx_for_s, adx_for_a, &
        adx_fro_n, adx_fro_s, adx_fro_a, nb_flds)

   if (adx_for_a > 0) &
        call adx_exch_3b(F_wrkx1,F_wrky1,dummy,dummy,dummy,wrkz1,wrky,dummy,dummy,dummy,wrkc1,nb_flds,F_ni,F_nj,F_nk)

   deallocate(xpos2,ypos2,zpos2,stat=istat)

   call msg(MSG_DEBUG,'adx_int_winds_glb [end]')
   !---------------------------------------------------------------------
   return
end subroutine adx_int_winds_glb

