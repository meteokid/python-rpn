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
subroutine adx_set_grid()
   implicit none
#include <arch_specific.hf>
   !@objective Compute derived grid parameters
   !@author Stephane Chamberland, 2010-01
   !@revisions
   !@ v4_40 Lee/Qaddouri - add precalculation of vsec, vtan variables for GCCSA
   !*@/
#include "adx_dims.cdk"
#include "adx_grid.cdk"
   integer :: istat, i, j
   real*8 :: prhxmn, prhymn



   !---------------------------------------------------------------------
   allocate( &
        adx_xx_8(adx_lminx:adx_lmaxx), &
        adx_cx_8(adx_lni), &
        adx_sx_8(adx_lni), &
        adx_wx_8(adx_lni), &
        adx_yy_8(adx_lminy:adx_lmaxy), &
        adx_vsec_8(adx_lminy:adx_lmaxy), &
        adx_vtan_8(adx_lminy:adx_lmaxy), &
        adx_cy_8(adx_lnj), &
        adx_sy_8(adx_lnj), &
        stat = istat)
   call handle_error_l(istat==0,'adx_set_grid','problem allocating mem')

   if (.not.adx_lam_L) then

      do i = adx_gminx,0
         adx_xg_8(i) = adx_xg_8(i+adx_gni) - CONST_2PI_8
      enddo
      do i = adx_gni+1,adx_gmaxx
         adx_xg_8(i) = adx_xg_8(i-adx_gni) + CONST_2PI_8
      enddo

      j = -1
      adx_yg_8(j) = -1.D0 * (CONST_PI_8 + adx_yg_8(j+2))
      j = 0
      adx_yg_8(j) = -1.D0* CONST_HALF_PI_8
      do j = -2,adx_gminy,-1
         adx_yg_8(j) = 2.D0*adx_yg_8(j+1) - adx_yg_8(j+2)
      enddo
      j = adx_gnj+1
      adx_yg_8(j) = CONST_HALF_PI_8
      j = adx_gnj+2
      adx_yg_8(j) = CONST_PI_8 - adx_yg_8(j-2)
      do j = adx_gnj+3,adx_gmaxy
         adx_yg_8(j) = 2.D0*adx_yg_8(j-1) - adx_yg_8(j-2)
      enddo

   else

      prhxmn =  adx_xg_8(2)-adx_xg_8(1)
      do i = 0,adx_gminx,-1
         adx_xg_8(i) = adx_xg_8(i+1) - prhxmn
      enddo
      do i = adx_gni+1,adx_gmaxx
         adx_xg_8(i) = adx_xg_8(i-1) + prhxmn
      enddo

      prhymn =  adx_yg_8(2)-adx_yg_8(1)
      do j = 0,adx_gminy,-1
         adx_yg_8(j) = adx_yg_8(j+1) - prhymn
      enddo
      do j = adx_gnj+1,adx_gmaxy
         adx_yg_8(j) = adx_yg_8(j-1) + prhymn
      enddo

   endif

   !- advection grid
   do i = adx_lminx,adx_lmaxx
      adx_xx_8(i) = adx_xg_8(adx_li0-1+i)
   enddo
   do j = adx_lminy,adx_lmaxy
      adx_yy_8(j) = adx_yg_8(adx_lj0-1+j)
   enddo

   do j = adx_lminy,adx_lmaxy
   !- precalculation vsec, vtan for grand circle computation
      adx_vsec_8(j) = 1.0D0/(cos(adx_yy_8(j)))
      adx_vtan_8(j) = tan(adx_yy_8(j))
   enddo

   if (.not.adx_lam_L) then
      do i = 1,adx_lni
         adx_wx_8(i) = 0.5D0*(adx_xx_8(i+1) - adx_xx_8(i-1)) / CONST_2PI_8
      enddo
   endif

   do i = 1,adx_lni
      adx_cx_8(i) = cos(adx_xx_8(i))
      adx_sx_8(i) = sin(adx_xx_8(i))
   enddo

   do j = 1,adx_lnj
      adx_cy_8(j) = cos(adx_yy_8(j))
      adx_sy_8(j) = sin(adx_yy_8(j))
   enddo
   !---------------------------------------------------------------------
   return
end subroutine adx_set_grid
