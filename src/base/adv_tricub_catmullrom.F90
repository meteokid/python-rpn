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

      subroutine adv_tricub_catmullrom (F_out, F_in, F_x, F_y, F_z, &
                  F_num, F_mono_L, i0, in, j0, jn, k0, F_nk, F_lev_S)
      implicit none
#include <arch_specific.hf>

      character(len=*), intent(in) :: F_lev_S ! m/t : Momemtum/thermo level
      integer, intent(in) :: F_num ! number points
      integer, intent(in) :: F_nk ! number of vertical levels
      integer, intent(in) :: i0,in,j0,jn,k0 ! scope of operator
      logical, intent(in) :: F_mono_L ! .true. monotonic interpolation
      real,dimension(F_num), intent(in) :: F_x, F_y, F_z ! interpolation target x,y,z coordinates
      real,dimension(*), intent(in) :: F_in ! field to interpolate
      real,dimension(F_num), intent(out) :: F_out ! result of interpolation
   !
   !@revisions
   !  2012-05,  Stephane Gaudreault: code optimization
   !@objective Tri-cubic interp: Lagrange vertical / Catmull-Rom horiz.

#include "adv_grid.cdk"
#include "adv_interp.cdk"
#include "glb_ld.cdk"
#include "ver.cdk"

      logical :: zcubic_L

      integer,dimension(:),pointer :: p_lcz
      real*8, dimension(:),pointer :: p_bsz_8, p_zbc_8, p_zabcd_8
      real*8, dimension(:),pointer :: p_zbacd_8, p_zcabd_8, p_zdabc_8

      integer :: n,i,j,k,ii_a,jj_a,kk_a,kkmax,idxk,idxjk,o1,o2,o3,o4
      real :: prmin, prmax, a1, a2, a3, a4  , &
           b1, b2, b3, b4, c1, c2, c3, c4, &
           d1, d2, d3, d4, p1, p2, p3, p4, ii_d, jj_d
      real*8 :: rri,rrj,rrk,ra,rb,rc,rd,p_z00_8,triprd,za,zb,zc,zd

      triprd(za,zb,zc,zd)=(za-zb)*(za-zc)*(za-zd)
!     
!---------------------------------------------------------------------
!     
      call timing_start2 (34, 'ADV_CATMUL', 31)

      p_z00_8 = Ver_z_8%m(0)
      if (F_lev_S == 'm') then
         kkmax   = F_nk - 1
         p_lcz     => adv_lcz%m
         p_bsz_8   => adv_bsz_8%m
         p_zabcd_8 => adv_zabcd_8%m
         p_zbacd_8 => adv_zbacd_8%m
         p_zcabd_8 => adv_zcabd_8%m
         p_zdabc_8 => adv_zdabc_8%m
         p_zbc_8   => adv_zbc_8%m
      else if (F_lev_S == 't') then
         kkmax   =  F_nk-1
         p_lcz     => adv_lcz%t
         p_bsz_8   => adv_bsz_8%t
         p_zabcd_8 => adv_zabcd_8%t
         p_zbacd_8 => adv_zbacd_8%t
         p_zcabd_8 => adv_zcabd_8%t
         p_zdabc_8 => adv_zdabc_8%t
         p_zbc_8   => adv_zbc_8%t
      else if (F_lev_S == 'x') then
         p_lcz     => adv_lcz%x
         p_bsz_8   => adv_bsz_8%x
         p_zabcd_8 => adv_zabcd_8%x
         p_zbacd_8 => adv_zbacd_8%x
         p_zcabd_8 => adv_zcabd_8%x
         p_zdabc_8 => adv_zdabc_8%x
         p_zbc_8   => adv_zbc_8%x
      endif

      if (F_mono_L) then

!$omp parallel private(o1, o2, o3, o4, a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4,  &
!$omp                  d1, d2, d3, d4, p1, p2, p3, p4, rri,rrj,rrk,ra,rb,rc,rd, n,i,j,k,&
!$omp                  ii_a,jj_a,kk_a,idxk, idxjk,zcubic_L)                                   &
!$omp          shared (F_out, F_in, F_x, F_y, F_z, adv_y00_8,adv_x00_8,adv_ovdx_8,adv_ovdy_8,   &
!$omp                  adv_ovdz_8, adv_bsx_8,adv_bsy_8,p_bsz_8, adv_xabcd_8,   &
!$omp                  adv_xbacd_8,adv_xcabd_8,adv_xdabc_8, adv_yabcd_8,adv_ybacd_8,adv_ycabd_8,&
!$omp                  adv_ydabc_8, p_zabcd_8,p_zbacd_8, p_zcabd_8, p_zdabc_8, p_zbc_8, kkmax,  &
!$omp                  F_num, i0, in, j0, jn, k0, F_nk)
!$omp do private(prmin, prmax)

#define ADV_MONO
#include "adv_tricub_catmullrom_loop.cdk"

!$omp enddo
!$omp end parallel

      else

!$omp parallel private(o1, o2, o3, o4, a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4,  &
!$omp                  d1, d2, d3, d4, p1, p2, p3, p4, rri,rrj,rrk,ra,rb,rc,rd, n,i,j,k,&
!$omp                  ii_a,jj_a,kk_a,idxk, idxjk,zcubic_L)                                   &
!$omp          shared (F_out, F_in, F_x, F_y, F_z, adv_y00_8,adv_x00_8,adv_ovdx_8,adv_ovdy_8,   &
!$omp                  adv_ovdz_8,adv_bsx_8,adv_bsy_8,p_bsz_8, adv_xabcd_8,    &
!$omp                  adv_xbacd_8,adv_xcabd_8,adv_xdabc_8, adv_yabcd_8,adv_ybacd_8,adv_ycabd_8,&
!$omp                  adv_ydabc_8, p_zabcd_8,p_zbacd_8, p_zcabd_8, p_zdabc_8, p_zbc_8, kkmax,  &
!$omp                  F_num, i0, in, j0, jn, k0, F_nk)
!$omp do

#undef ADV_MONO
#include "adv_tricub_catmullrom_loop.cdk"

!$omp enddo
!$omp end parallel

      endif

      call timing_stop (34)
!     
!---------------------------------------------------------------------
!     
      return
      end subroutine adv_tricub_catmullrom
