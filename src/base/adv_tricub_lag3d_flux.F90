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

!**s/p adv_tricub_lag3d_flux: Estimate FLUX_out/FLUX_in when LAM using Flux calculations based on Aranami et al. (2015)

      subroutine adv_tricub_lag3d_flux (F_cub_o,F_in_o,F_cub_i,F_in_i, & 
                                         F_x,F_y,F_z,F_num,k0,F_nk,F_lev_S) 

      implicit none

#include <arch_specific.hf>

      character(len=*), intent(in) :: F_lev_S ! m/t : Momemtum/thermo level
      integer, intent(in) :: F_num ! number points
      integer, intent(in) :: F_nk ! number of vertical levels
      integer, intent(in) :: k0   ! scope of operator
      real,dimension(F_num), intent(in)    :: F_x, F_y, F_z ! interpolation target x,y,z coordinates
      real,dimension(*),     intent(in)    :: F_in_o,F_in_i ! field to interpolate (F_in_o for FLUX_out;F_in_i for FLUX_in)
      real,dimension(F_num), intent(inout) :: F_cub_o       ! High-order SL solution FLUX_out
      real,dimension(F_num), intent(inout) :: F_cub_i       ! High-order SL solution FLUX_in

      !@author Monique Tanguay

      !@revisions
      ! v4_XX - Tanguay M.        - GEM4 Mass-Conservation


#include "adv_grid.cdk"
#include "adv_interp.cdk"
#include "glb_ld.cdk"
#include "grd.cdk"
#include "ver.cdk"

      logical :: zcubic_L

      integer,dimension(:),pointer :: p_lcz
      real*8, dimension(:),pointer :: p_bsz_8, p_zbc_8, p_zabcd_8
      real*8, dimension(:),pointer :: p_zbacd_8, p_zcabd_8, p_zdabc_8

      integer, parameter :: CFL = 5.0

      integer :: n,i,j,k,ii,jj,kk,kkmax,idxk,idxjk,o1,o2,o3,o4, &
                 i0_e,in_e,j0_e,jn_e,i0_o,in_o,j0_o,jn_o,i0_i,in_i,j0_i,jn_i
      real*8 :: a1, a2, a3, a4, &
              b1, b2, b3, b4, c1, c2, c3, c4, &
              d1, d2, d3, d4, p1, p2, p3, p4, &
              rri,rrj,rrk,ra,rb,rc,rd, &
              triprd,za,zb,zc,zd,p_z00_8

      triprd(za,zb,zc,zd)=(za-zb)*(za-zc)*(za-zd)
!     
!---------------------------------------------------------------------
!     
      call timing_start2 (33, 'ADV_LAG3D', 31)
      
      kkmax   = l_nk - 1
      p_z00_8 = Ver_z_8%m(0)
      if (F_lev_S == 'm') then       
         p_lcz     => adv_lcz%m
         p_bsz_8   => adv_bsz_8%m
         p_zabcd_8 => adv_zabcd_8%m
         p_zbacd_8 => adv_zbacd_8%m
         p_zcabd_8 => adv_zcabd_8%m
         p_zdabc_8 => adv_zdabc_8%m
         p_zbc_8   => adv_zbc_8%m
      else if (F_lev_S == 't') then
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




      !Establish scope of extended advection operations
      !------------------------------------------------
      call adv_get_ij0n_ext (i0_e,in_e,j0_e,jn_e)

      !-----------------
      !Estimate FLUX_out
      !-----------------
      F_cub_o = 0.0

      if (l_west) then
         i0_o = i0_e
         in_o = pil_w
         j0_o = j0_e
         jn_o = jn_e
#include "adv_tricub_lag3d_flux_loop_o.cdk"
      endif
      if (l_south) then
         i0_o = i0_e
         in_o = in_e
         j0_o = j0_e
         jn_o = pil_s
#include "adv_tricub_lag3d_flux_loop_o.cdk"
      endif
      if (l_east) then
         i0_o = l_ni-pil_e+1
         in_o = in_e
         j0_o = j0_e
         jn_o = jn_e
#include "adv_tricub_lag3d_flux_loop_o.cdk"
      endif
      if (l_north) then
         i0_o = i0_e
         in_o = in_e
         j0_o = l_nj-pil_n+1
         jn_o = jn_e
#include "adv_tricub_lag3d_flux_loop_o.cdk"
      endif

      !----------------
      !Estimate FLUX_in
      !----------------
      F_cub_i = 0.0

      if (l_west) then
         i0_i = 1+pil_w
         in_i = 1+pil_w+CFL
         j0_i = 1+pil_s
         jn_i = l_nj-pil_n
#include "adv_tricub_lag3d_flux_loop_i.cdk"
      endif
      if (l_south) then
         i0_i = 1+pil_w
         in_i = l_ni-pil_e
         j0_i = 1+pil_s
         jn_i = 1+pil_s+CFL
#include "adv_tricub_lag3d_flux_loop_i.cdk"
      endif
      if (l_east) then
         i0_i = l_ni-pil_e-CFL
         in_i = l_ni-pil_e
         j0_i = 1+pil_s
         jn_i = l_nj-pil_n
#include "adv_tricub_lag3d_flux_loop_i.cdk"
      endif
      if (l_north) then
         i0_i = 1+pil_w
         in_i = l_ni-pil_e
         j0_i = l_nj-pil_n-CFL
         jn_i = l_nj-pil_n
#include "adv_tricub_lag3d_flux_loop_i.cdk"
      endif

      call timing_stop (33)
!     
!---------------------------------------------------------------------
!     
      return

      end subroutine adv_tricub_lag3d_flux
