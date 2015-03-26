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
#include "stop_mpi.h"
#include "msg.h"

subroutine adx_pole0s()
   implicit none
#include <arch_specific.hf>
   !---------------------------------------------------------------------
   call stop_mpi(STOP_ERROR,'adx_pole0s','called a stub')
   !---------------------------------------------------------------------
   return
end subroutine adx_pole0s

!/@*
subroutine adx_pole0s2(F_fld_adw, F_fld_model, &
     F_aminx,F_amaxx,F_aminy,F_amaxy,F_minx,F_maxx,F_miny,F_maxy,F_nk,&
     F_pol0_L, F_extend_L, F_is_south_L)
   implicit none
#include <arch_specific.hf>
#include "adx_dims.cdk"
#include "adx_grid.cdk"
   !@objective Extend the grid from model to adw with filled halos
   !@arguments
   logical :: F_is_south_L !I, .true. if south pole
   logical :: F_extend_L   !I, Extend field beyond poles
   logical :: F_pol0_L     !I, Set values=0 around poles (e.g. 4 winds)
   integer :: F_aminx,F_amaxx,F_aminy,F_amaxy !I, adw local array bounds
   integer :: F_minx,F_maxx,F_miny,F_maxy     !I, model's local array bounds
   integer :: F_nk         !I, number of levels
   real, dimension(F_minx:F_maxx,F_miny:F_maxy,F_nk) :: &
        F_fld_model        !I, fld on model-grid
   real, dimension(F_aminx:F_amaxx,F_aminy:F_amaxy,F_nk) :: &
        F_fld_adw          !O, fld on adw-grid
   !*@/
   integer :: j1, j2, i, k
   real*8  :: ww_8
   !---------------------------------------------------------------------
   call msg(MSG_DEBUG,'adx_pole0s')

!$omp parallel private(j1,j2,ww_8)
   if (F_is_south_L) then
      j1 = 0
      j2 = 1
   else
      j1 = adx_lnj+1
      j2 = adx_lnj
   endif

   if  (F_pol0_L) then
      !set values at the pole = 0.0
!$omp do
      do k = 1, F_nk
         do i = F_aminx, F_amaxx
            F_fld_adw(i,j1,k) = 0.0
         enddo
      enddo
!$omp enddo
   else
      !compute weighted average around the pole
!$omp do
      do k = 1, F_nk
         ww_8 = 0.D0
         do i = 1, adx_lni
            ww_8 = ww_8 + adx_wx_8(i) * dble(F_fld_adw(i,j2,k))
         enddo
         do i = F_aminx, F_amaxx
            F_fld_adw(i,j1,k) = sngl(ww_8)
         enddo
      enddo
!$omp enddo
   endif
   if (F_extend_L) then
      !extension of a scalar field beyond the poles
      call adx_polx3(F_fld_adw, F_is_south_L, &
           F_aminx,F_amaxx,F_aminy,F_amaxy, &
           adx_lni,adx_lnj, adx_halox,adx_haloy, F_nk)
   endif
!$omp end parallel

   call msg(MSG_DEBUG,'adx_pole0s [end]')
   !---------------------------------------------------------------------
   return
end subroutine adx_pole0s2
