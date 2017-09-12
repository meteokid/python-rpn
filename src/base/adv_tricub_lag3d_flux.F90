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
                                         F_x,F_y,F_z,F_num,k0,F_lev_S)

      use grid_options
      use glb_ld
      use ver
      use adv
      use adv_grid
      use adv_interp
      use outgrid
      implicit none

#include <arch_specific.hf>

      character(len=*), intent(in) :: F_lev_S ! m/t : Momemtum/thermo level
      integer, intent(in) :: F_num ! number points
      integer, intent(in) :: k0   ! scope of operator
      real,dimension(F_num), intent(in)    :: F_x, F_y, F_z ! interpolation target x,y,z coordinates
      real,dimension(*),     intent(in)    :: F_in_o,F_in_i ! field to interpolate (F_in_o for FLUX_out;F_in_i for FLUX_in)
      real,dimension(F_num), intent(inout) :: F_cub_o       ! High-order SL solution FLUX_out
      real,dimension(F_num), intent(inout) :: F_cub_i       ! High-order SL solution FLUX_in

      !@author Monique Tanguay

      !@revisions
      ! v4_80 - Tanguay M.        - GEM4 Mass-Conservation
      ! v5_00 - Tanguay M.        - Adjust extension


#include "adv_precompute_flux.cdk"

      logical :: zcubic_L

      integer,dimension(:),pointer :: p_lcz
      real*8, dimension(:),pointer :: p_bsz_8, p_zbc_8, p_zabcd_8
      real*8, dimension(:),pointer :: p_zbacd_8, p_zcabd_8, p_zdabc_8

      integer jext

      integer :: n0,nx,ny,nz,m1,id,n,kkmax,o1,o2,o3,o4, &
                 i0_e,in_e,j0_e,jn_e
      real   :: za
      real*8 :: a1, a2, a3, a4, &
              b1, b2, b3, b4, c1, c2, c3, c4, &
              d1, d2, d3, d4, p1, p2, p3, p4, &
              ra,rb,rc,rd, &
              triprd,zb,zc,zd,p_z00_8

      triprd(za,zb,zc,zd)=(dble(za)-zb)*(za-zc)*(za-zd)

      logical,save :: done_L=.FALSE.

      integer i0_w_i,in_w_i,j0_w_i,jn_w_i,i0_w_o,in_w_o,j0_w_o,jn_w_o, &
              i0_s_i,in_s_i,j0_s_i,jn_s_i,i0_s_o,in_s_o,j0_s_o,jn_s_o, &
              i0_e_i,in_e_i,j0_e_i,jn_e_i,i0_e_o,in_e_o,j0_e_o,jn_e_o, &
              i0_n_i,in_n_i,j0_n_i,jn_n_i,i0_n_o,in_n_o,j0_n_o,jn_n_o

      common/storage/ i0_w_i,in_w_i,j0_w_i,jn_w_i,i0_w_o,in_w_o,j0_w_o,jn_w_o, &
                      i0_s_i,in_s_i,j0_s_i,jn_s_i,i0_s_o,in_s_o,j0_s_o,jn_s_o, &
                      i0_e_i,in_e_i,j0_e_i,jn_e_i,i0_e_o,in_e_o,j0_e_o,jn_e_o, &
                      i0_n_i,in_n_i,j0_n_i,jn_n_i,i0_n_o,in_n_o,j0_n_o,jn_n_o

!
!---------------------------------------------------------------------
!
      call timing_start2 (77, 'ADV_FLUX_', 39)

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

      jext = Grd_maxcfl

      if (.NOT.done_L) then

         if (l_west) then

            i0_w_o = i0_e
            in_w_o = pil_w
            j0_w_o = j0_e
            jn_w_o = jn_e

            nind_w_o = (in_w_o-i0_w_o+1)*(jn_w_o-j0_w_o+1)*(l_nk-k0+1)

            allocate (ii_w_o(4*nind_w_o))

            i0_w_i = 1+pil_w
            in_w_i = 1+pil_w+jext
            j0_w_i = 1+pil_s
            jn_w_i = l_nj-pil_n

            nind_w_i = (in_w_i-i0_w_i+1)*(jn_w_i-j0_w_i+1)*(l_nk-k0+1)

            allocate (ii_w_i(4*nind_w_i))

         endif

         if (l_south) then

           i0_s_o = i0_e
           in_s_o = in_e
           j0_s_o = j0_e
           jn_s_o = pil_s

           nind_s_o = (in_s_o-i0_s_o+1)*(jn_s_o-j0_s_o+1)*(l_nk-k0+1)

           allocate (ii_s_o(4*nind_s_o))

           i0_s_i = 1+pil_w
           in_s_i = l_ni-pil_e
           j0_s_i = 1+pil_s
           jn_s_i = 1+pil_s+jext

           nind_s_i = (in_s_i-i0_s_i+1)*(jn_s_i-j0_s_i+1)*(l_nk-k0+1)

           allocate (ii_s_i(4*nind_s_i))

         endif

         if (l_east) then

            i0_e_o = l_ni-pil_e+1
            in_e_o = in_e
            j0_e_o = j0_e
            jn_e_o = jn_e

            nind_e_o = (in_e_o-i0_e_o+1)*(jn_e_o-j0_e_o+1)*(l_nk-k0+1)

            allocate (ii_e_o(4*nind_e_o))

            i0_e_i = l_ni-pil_e-jext
            in_e_i = l_ni-pil_e
            j0_e_i = 1+pil_s
            jn_e_i = l_nj-pil_n

            nind_e_i = (in_e_i-i0_e_i+1)*(jn_e_i-j0_e_i+1)*(l_nk-k0+1)

            allocate (ii_e_i(4*nind_e_i))

         endif

         if (l_north) then

            i0_n_o = i0_e
            in_n_o = in_e
            j0_n_o = l_nj-pil_n+1
            jn_n_o = jn_e

            nind_n_o = (in_n_o-i0_n_o+1)*(jn_n_o-j0_n_o+1)*(l_nk-k0+1)

            allocate (ii_n_o(4*nind_n_o))

            i0_n_i = 1+pil_w
            in_n_i = l_ni-pil_e
            j0_n_i = l_nj-pil_n-jext
            jn_n_i = l_nj-pil_n

            nind_n_i = (in_n_i-i0_n_i+1)*(jn_n_i-j0_n_i+1)*(l_nk-k0+1)

            allocate (ii_n_i(4*nind_n_i))

         endif

      endif

      done_L = .TRUE.

      !Pre-compute indices ii used in: adv_tricub_lag3d_loop
      !-----------------------------------------------------
      if (.NOT.Adv_done_precompute_L) then

         if (l_west) then

            call adv_get_indices (ii_w_o, F_x, F_y, F_z, F_num , nind_w_o, &
                                  i0_w_o, in_w_o, j0_w_o, jn_w_o, k0 , l_nk, 't')


            call adv_get_indices (ii_w_i, F_x, F_y, F_z, F_num , nind_w_i, &
                                  i0_w_i, in_w_i, j0_w_i, jn_w_i, k0 , l_nk, 't')

         endif

         if (l_south) then

            call adv_get_indices (ii_s_o, F_x, F_y, F_z, F_num , nind_s_o, &
                                  i0_s_o, in_s_o, j0_s_o, jn_s_o, k0 , l_nk, 't')

            call adv_get_indices (ii_s_i, F_x, F_y, F_z, F_num , nind_s_i, &
                                  i0_s_i, in_s_i, j0_s_i, jn_s_i, k0 , l_nk, 't')

         endif

         if (l_east) then

            call adv_get_indices (ii_e_o, F_x, F_y, F_z, F_num , nind_e_o, &
                                  i0_e_o, in_e_o, j0_e_o, jn_e_o, k0 , l_nk, 't')

            call adv_get_indices (ii_e_i, F_x, F_y, F_z, F_num , nind_e_i, &
                                  i0_e_i, in_e_i, j0_e_i, jn_e_i, k0 , l_nk, 't')

         endif

         if (l_north) then

            call adv_get_indices (ii_n_o, F_x, F_y, F_z, F_num , nind_n_o, &
                                  i0_n_o, in_n_o, j0_n_o, jn_n_o, k0 , l_nk, 't')

            call adv_get_indices (ii_n_i, F_x, F_y, F_z, F_num , nind_n_i, &
                                  i0_n_i, in_n_i, j0_n_i, jn_n_i, k0 , l_nk, 't')

         endif

      endif

      Adv_done_precompute_L = .TRUE.

      !-------------------------
      !Estimate FLUX_out/FLUX_in
      !-------------------------
      F_cub_o = 0.0
      F_cub_i = 0.0

      nind_o = -1
      nind_i = -1

      if (l_west) then

         nind_o = nind_w_o
         nind_i = nind_w_i

         ii_o => ii_w_o
         ii_i => ii_w_i

      endif

#include "adv_tricub_lag3d_flux_loop_o.cdk"
#include "adv_tricub_lag3d_flux_loop_i.cdk"

      nind_o = -1
      nind_i = -1

      if (l_south) then

         nind_o = nind_s_o
         nind_i = nind_s_i

         ii_o => ii_s_o
         ii_i => ii_s_i

      endif

#include "adv_tricub_lag3d_flux_loop_o.cdk"
#include "adv_tricub_lag3d_flux_loop_i.cdk"

      nind_o = -1
      nind_i = -1

      if (l_east) then

         nind_o = nind_e_o
         nind_i = nind_e_i

         ii_o => ii_e_o
         ii_i => ii_e_i

      endif

#include "adv_tricub_lag3d_flux_loop_o.cdk"
#include "adv_tricub_lag3d_flux_loop_i.cdk"

      nind_o = -1
      nind_i = -1

      if (l_north) then

         nind_o = nind_n_o
         nind_i = nind_n_i

         ii_o => ii_n_o
         ii_i => ii_n_i

      endif

#include "adv_tricub_lag3d_flux_loop_o.cdk"
#include "adv_tricub_lag3d_flux_loop_i.cdk"

      call timing_stop (77)
!
!---------------------------------------------------------------------
!
      return

      end subroutine adv_tricub_lag3d_flux
