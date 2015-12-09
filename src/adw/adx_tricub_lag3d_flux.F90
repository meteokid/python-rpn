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

!**s/p adx_tricub_lag3d_flux: Estimate FLUX_out/FLUX_in when LAM using Flux calculations based on Aranami et al. (2015)

      subroutine adx_tricub_lag3d_flux (F_cub_o,F_in_o,F_cub_i,F_in_i, & 
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
      ! v4_80 - Tanguay M.        - GEM4 Mass-Conservation and FLUX calculations

#include "adx_dims.cdk"
#include "adx_grid.cdk"
#include "adx_interp.cdk"

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
      call timing_start2 (33, 'ADX_LAG3D', 31)

      p_z00_8 = adx_verZ_8%m(0)
      if (F_lev_S == 'm') then
         kkmax   = adx_lnk - 1
         p_lcz     => adx_lcz%m
         p_bsz_8   => adx_bsz_8%m
         p_zabcd_8 => adx_zabcd_8%m
         p_zbacd_8 => adx_zbacd_8%m
         p_zcabd_8 => adx_zcabd_8%m
         p_zdabc_8 => adx_zdabc_8%m
         p_zbc_8   => adx_zbc_8%m
      else if (F_lev_S == 't') then
         kkmax   = adx_lnk-1
         p_lcz     => adx_lcz%t
         p_bsz_8   => adx_bsz_8%t
         p_zabcd_8 => adx_zabcd_8%t
         p_zbacd_8 => adx_zbacd_8%t
         p_zcabd_8 => adx_zcabd_8%t
         p_zdabc_8 => adx_zdabc_8%t
         p_zbc_8   => adx_zbc_8%t
      else if (F_lev_S == 's') then
         kkmax   = adx_lnks - 1
         p_lcz     => adx_lcz%s
         p_bsz_8   => adx_bsz_8%s
         p_zabcd_8 => adx_zabcd_8%s
         p_zbacd_8 => adx_zbacd_8%s
         p_zcabd_8 => adx_zcabd_8%s
         p_zdabc_8 => adx_zdabc_8%s
         p_zbc_8   => adx_zbc_8%s
      else
         !TO DO catch this error (since notr all cpu pass here we cannot call handle_error)
         ! This s/r should be a function with a return status.
         ! print*,'ERROR IN ADX_TRICUB_LAG3D'
      endif

      !Establish scope of extended advection operations
      !------------------------------------------------
      call adx_get_ij0n_ext (i0_e,in_e,j0_e,jn_e)

      !-----------------
      !Estimate FLUX_out
      !-----------------
      F_cub_o = 0.0

      if (adx_is_west) then
         i0_o = i0_e
         in_o = adx_pil_w
         j0_o = j0_e
         jn_o = jn_e
#include "adx_tricub_lag3d_flux_loop_o.cdk"
      endif
      if (adx_is_south) then
         i0_o = i0_e
         in_o = in_e
         j0_o = j0_e
         jn_o = adx_pil_s
#include "adx_tricub_lag3d_flux_loop_o.cdk"
      endif
      if (adx_is_east) then
         i0_o = adx_mlni-adx_pil_e+1
         in_o = in_e
         j0_o = j0_e
         jn_o = jn_e
#include "adx_tricub_lag3d_flux_loop_o.cdk"
      endif
      if (adx_is_north) then
         i0_o = i0_e
         in_o = in_e
         j0_o = adx_mlnj-adx_pil_n+1
         jn_o = jn_e
#include "adx_tricub_lag3d_flux_loop_o.cdk"
      endif

      !----------------
      !Estimate FLUX_in
      !----------------
      F_cub_i = 0.0

      if (adx_is_west) then
         i0_i = 1+adx_pil_w
         in_i = 1+adx_pil_w+CFL
         j0_i = 1+adx_pil_s
         jn_i = adx_mlnj-adx_pil_n
#include "adx_tricub_lag3d_flux_loop_i.cdk"
      endif
      if (adx_is_south) then
         i0_i = 1+adx_pil_w
         in_i = adx_mlni-adx_pil_e
         j0_i = 1+adx_pil_s
         jn_i = 1+adx_pil_s+CFL
#include "adx_tricub_lag3d_flux_loop_i.cdk"
      endif
      if (adx_is_east) then
         i0_i = adx_mlni-adx_pil_e-CFL
         in_i = adx_mlni-adx_pil_e
         j0_i = 1+adx_pil_s
         jn_i = adx_mlnj-adx_pil_n
#include "adx_tricub_lag3d_flux_loop_i.cdk"
      endif
      if (adx_is_north) then
         i0_i = 1+adx_pil_w
         in_i = adx_mlni-adx_pil_e
         j0_i = adx_mlnj-adx_pil_n-CFL
         jn_i = adx_mlnj-adx_pil_n
#include "adx_tricub_lag3d_flux_loop_i.cdk"
      endif

      call timing_stop (33)
!     
!---------------------------------------------------------------------
!     
      return

      end subroutine adx_tricub_lag3d_flux
