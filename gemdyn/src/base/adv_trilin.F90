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
#include "msg.h"

subroutine adv_trilin(F_xo,F_yo,F_u1,F_u2,F_xth,F_yth,F_zth, &
     F_dth,F_has_u2_L, F_i0,F_in,F_j0,F_jn, &
     F_ni,F_nj,F_aminx, F_amaxx, F_aminy, F_amaxy,F_k0,F_nk)
   implicit none
#include <arch_specific.hf>
   !@objective switcher to call adx_setint/adx_trilin or adx_trilin_turbo
   !@arguments
   logical :: F_has_u2_L               !I, .T. if F_u2 needs to be treated
   real    :: F_dth
   integer :: F_nk         !I, number of vertical levels (may be super grid for winds)
   integer :: F_aminx, F_amaxx, F_aminy, F_amaxy !I, wind fields array bounds
   integer :: F_ni, F_nj               !I, dims of position fields
   integer :: F_i0,F_in,F_j0,F_jn,F_k0 !I, operator scope
   real,dimension(F_ni,F_nj,F_nk) :: F_xth,F_yth,F_zth !I, x,y,z positions
   real,dimension(F_aminx:F_amaxx,F_aminy:F_amaxy,F_nk) :: &
        F_u1, F_u2  !I, field to interpol
   real,dimension(F_ni,F_nj,F_nk) :: F_xo,F_yo !O, result of interp
   !@author Stephane Chamberland 
   !@revisions
   ! v4_40 - Tanguay M.        - Revision TL/AD
   ! v4_70 - PLante A.         - remove adw_nosetint_L

#include "glb_ld.cdk"
#include "adv_grid.cdk"
#include "adv_interp.cdk"
#include "ver.cdk"

   integer :: num,iimax
   integer, dimension(F_ni,F_nj,F_nk) :: loci,locj,lock
   real,    dimension(F_ni,F_nj,F_nk) :: capz1
   real*8 p_z00_8
   real*8,  dimension(:), pointer :: p_bsz_8
   integer, dimension(:), pointer :: p_lcz
   !---------------------------------------------------------------------
   call msg(MSG_DEBUG,'adv_trilin [begin]')
   num = F_ni*F_nj*F_nk
   
   p_z00_8 =Ver_z_8%m(0)
   p_lcz   => adv_lcz%m
   p_bsz_8 => adv_bsz_8%m      
   
   iimax   = adv_iimax+1

   call adv_trilin_ijk (                                              &
        F_xth, F_yth, F_zth, capz1, loci, locj, lock,                   &
        adv_lcx, adv_lcy, p_lcz, adv_bsx_8, adv_bsy_8, p_bsz_8, &
        adv_diz_8, p_z00_8, iimax,                                      &
        num, F_i0, F_in, F_j0, F_jn, F_k0, F_nk, l_nk)

   call adv_trilin_turbo3 (F_xo, F_u1, F_dth,      &
        F_xth, F_yth, capz1, loci, locj, lock,      &
        adv_bsx_8, adv_bsy_8, adv_xbc_8, adv_ybc_8, &
        num, F_i0, F_in, F_j0, F_jn, F_k0, F_nk)
  
   if (F_has_u2_L) then
      call adv_trilin_turbo3 (F_yo, F_u2, F_dth,      &
           F_xth, F_yth, capz1, loci, locj, lock,      &
           adv_bsx_8, adv_bsy_8, adv_xbc_8, adv_ybc_8, &
           num, F_i0, F_in, F_j0, F_jn, F_k0, F_nk)
   endif
   
   call msg(MSG_DEBUG,'adv_trilin [end]')
   !---------------------------------------------------------------------
   return
end subroutine adv_trilin
