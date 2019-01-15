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
      use lun
      use ptopo 
      use gem_options
      use ver 
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


      integer nind_o,nind_i
      integer nind_w_o,nind_s_o,nind_e_o,nind_n_o,nind_w_i,nind_s_i,nind_e_i,nind_n_i
      integer,save,pointer,dimension(:) :: ii_w_o =>null(), ii_s_o =>null(), ii_e_o =>null(), ii_n_o =>null(), &
                                           ii_w_i =>null(), ii_s_i =>null(), ii_e_i =>null(), ii_n_i =>null()
      integer,     pointer,dimension(:) :: ii_o   =>null(), ii_i   =>null()
      common /precompute_I/ nind_w_o,nind_s_o,nind_e_o,nind_n_o,nind_w_i,nind_s_i,nind_e_i,nind_n_i

      logical :: zcubic_L

      integer,dimension(:),pointer :: p_lcz
      real*8, dimension(:),pointer :: p_bsz_8, p_zbc_8, p_zabcd_8
      real*8, dimension(:),pointer :: p_zbacd_8, p_zcabd_8, p_zdabc_8

      integer jext,err

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

      integer :: g_narrow_i_e,g_narrow_j_n,narrow_i_e,narrow_j_n
      integer :: g_narrow_i_w,g_narrow_j_s,narrow_i_w,narrow_j_s

      logical, save :: narrow_i_e_L,narrow_j_n_L,narrow_i_w_L,narrow_j_s_L,narrow_L

      integer :: i,j,k,ne,midxk,midxjk,m,F_minx_e,F_maxx_e,F_miny_e,F_maxy_e,F_ni,F_nj,F_nk, &
                 ii,jj,kk,ii1,jj1,kk1,idxk,idxjk,i0_i,in_i,j0_i,jn_i,sig, &
                 i0_e_out,in_e_out,j0_e_out,jn_e_out,i_near

      real,    save, pointer, dimension(:) :: px,py,pz

      integer, save :: num_PE_near_NE,num_PE_near_NW, &
                       num_PE_near_SE,num_PE_near_SW,num_e,i_near_E,i_near_W

      real :: i_h(adv_lminx:adv_lmaxx,adv_lminy:adv_lmaxy,l_nk)

      real*8  :: rri,rrj,rrk
!
!---------------------------------------------------------------------
!
      call timing_start2 (77, 'ADV_FLUX_', 39)

      !Vertical variable type:  Height--> sig <0 , Pressure --> sig >0
      sig=int((Ver_z_8%m(l_nk)-Ver_z_8%m(1))/(abs(  Ver_z_8%m(l_nk)-Ver_z_8%m(1) )))

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

      F_minx_e = adv_lminx
      F_maxx_e = adv_lmaxx
      F_miny_e = adv_lminy
      F_maxy_e = adv_lmaxy

      F_ni = l_ni
      F_nj = l_nj
      F_nk = l_nk

      !Establish scope of extended advection operations
      !------------------------------------------------
      call adv_get_ij0n_ext (i0_e,in_e,j0_e,jn_e)

      jext = Grd_maxcfl + 1

      !We do not modify how FLUX_out is calculated
      !-------------------------------------------
      i0_e_out = i0_e
      in_e_out = in_e
      j0_e_out = j0_e
      jn_e_out = jn_e

      !E grid: HALO_E and extended advection limits (in nesting zone)
      !--------------------------------------------------------------
      if (.not.l_west)  i0_e = F_minx_e
      if (.not.l_east)  in_e = F_maxx_e
      if (.not.l_south) j0_e = F_miny_e
      if (.not.l_north) jn_e = F_maxy_e

      if (.NOT.done_L) then

         !Check if PEs at WEST/SOUTH/EAST/NORTH boundaries are too narrow to calculate FLUX_in
         !------------------------------------------------------------------------------------
         narrow_i_w = 0
         narrow_j_s = 0
         narrow_i_e = 0
         narrow_j_n = 0

         if (l_west) then

            in_w_i = 1+pil_w+jext

            if (in_w_i > l_ni) narrow_i_w = 1

         endif

         if (l_south) then

            jn_s_i = 1+pil_s+jext

            if (jn_s_i > l_nj) narrow_j_s = 1

         endif

         if (l_east) then

            i0_e_i = l_ni-pil_e-jext

            if (i0_e_i < 1) narrow_i_e = 1

         endif

         if (l_north) then

            j0_n_i = l_nj-pil_n-jext

            if (j0_n_i < 1) narrow_j_n = 1

         endif

         call rpn_comm_ALLREDUCE(narrow_i_w,g_narrow_i_w,1,"MPI_INTEGER","MPI_MAX","GRID",err)
         call rpn_comm_ALLREDUCE(narrow_j_s,g_narrow_j_s,1,"MPI_INTEGER","MPI_MAX","GRID",err)
         call rpn_comm_ALLREDUCE(narrow_i_e,g_narrow_i_e,1,"MPI_INTEGER","MPI_MAX","GRID",err)
         call rpn_comm_ALLREDUCE(narrow_j_n,g_narrow_j_n,1,"MPI_INTEGER","MPI_MAX","GRID",err)

         narrow_i_w_L = g_narrow_i_w/=0
         narrow_j_s_L = g_narrow_j_s/=0
         narrow_i_e_L = g_narrow_i_e/=0
         narrow_j_n_L = g_narrow_j_n/=0

         narrow_L = narrow_i_w_L.or.narrow_j_s_L.or.narrow_i_e_L.or.narrow_j_n_L

         if (Lun_out>0) then
                              write(Lun_out,*) ''
            if (narrow_i_w_L) write(Lun_out,*) 'ADV_TRICUB_LAG3D_FLUX: PEs WEST  are narrow: We communicate with neighbors'
            if (narrow_j_s_L) write(Lun_out,*) 'ADV_TRICUB_LAG3D_FLUX: PEs SOUTH are narrow: We communicate with neighbors'
            if (narrow_i_e_L) write(Lun_out,*) 'ADV_TRICUB_LAG3D_FLUX: PEs EAST  are narrow: We communicate with neighbors'
            if (narrow_j_n_L) write(Lun_out,*) 'ADV_TRICUB_LAG3D_FLUX: PEs NORTH are narrow: We communicate with neighbors'
                              write(Lun_out,*) ''
         endif

         if (narrow_L) then

            num_e = (F_maxx_e-F_minx_e+1)*(F_maxy_e-F_miny_e+1)*F_nk

            allocate (px(num_e),py(num_e),pz(num_e))

         endif

         if (l_west) then

            i0_w_o = i0_e_out
            in_w_o = pil_w
            j0_w_o = j0_e_out
            jn_w_o = jn_e_out

            nind_w_o = (in_w_o-i0_w_o+1)*(jn_w_o-j0_w_o+1)*(l_nk-k0+1)

            allocate (ii_w_o(4*nind_w_o))

            i0_w_i = 1+pil_w
            in_w_i = 1+pil_w+jext
            j0_w_i = 1+pil_s
            jn_w_i = l_nj-pil_n

            if (narrow_i_w_L) in_w_i = max(in_w_i,F_maxx_e)

            nind_w_i = (in_w_i-i0_w_i+1)*(jn_w_i-j0_w_i+1)*(l_nk-k0+1)

            allocate (ii_w_i(4*nind_w_i))

         endif

         if (l_south) then

            i0_s_o = i0_e_out
            in_s_o = in_e_out
            j0_s_o = j0_e_out
            jn_s_o = pil_s

            nind_s_o = (in_s_o-i0_s_o+1)*(jn_s_o-j0_s_o+1)*(l_nk-k0+1)

            allocate (ii_s_o(4*nind_s_o))

            i0_s_i = 1+pil_w
            in_s_i = l_ni-pil_e
            j0_s_i = 1+pil_s
            jn_s_i = 1+pil_s+jext

            if (narrow_j_s_L) jn_s_i = max(jn_s_i,F_maxy_e)

            nind_s_i = (in_s_i-i0_s_i+1)*(jn_s_i-j0_s_i+1)*(l_nk-k0+1)

            allocate (ii_s_i(4*nind_s_i))

         endif

         if (l_east) then

            i0_e_o = l_ni-pil_e+1
            in_e_o = in_e_out
            j0_e_o = j0_e_out
            jn_e_o = jn_e_out

            nind_e_o = (in_e_o-i0_e_o+1)*(jn_e_o-j0_e_o+1)*(l_nk-k0+1)

            allocate (ii_e_o(4*nind_e_o))

            i0_e_i = l_ni-pil_e-jext
            in_e_i = l_ni-pil_e
            j0_e_i = 1+pil_s
            jn_e_i = l_nj-pil_n

            if (narrow_i_e_L) i0_e_i = max(i0_e_i,F_minx_e)

            nind_e_i = (in_e_i-i0_e_i+1)*(jn_e_i-j0_e_i+1)*(l_nk-k0+1)

            allocate (ii_e_i(4*nind_e_i))

         endif

         if (l_north) then

            i0_n_o = i0_e_out
            in_n_o = in_e_out
            j0_n_o = l_nj-pil_n+1
            jn_n_o = jn_e_out

            nind_n_o = (in_n_o-i0_n_o+1)*(jn_n_o-j0_n_o+1)*(l_nk-k0+1)

            allocate (ii_n_o(4*nind_n_o))

            i0_n_i = 1+pil_w
            in_n_i = l_ni-pil_e
            j0_n_i = l_nj-pil_n-jext
            jn_n_i = l_nj-pil_n

            if (narrow_j_n_L) j0_n_i = max(j0_n_i,F_miny_e)

            nind_n_i = (in_n_i-i0_n_i+1)*(jn_n_i-j0_n_i+1)*(l_nk-k0+1)

            allocate (ii_n_i(4*nind_n_i))

         endif

         num_PE_near_SW =  1 + 1
         num_PE_near_SE =  Ptopo_npex - 1 - 1 
         num_PE_near_NW = (Ptopo_npex * Ptopo_npey -1) - Ptopo_npex + 1
         num_PE_near_NE = (Ptopo_npex * Ptopo_npey -1) - 1

         !Initialize i_near_E = i0_e_i of PE at E 
         !---------------------------------------
         if (narrow_i_e_L.and.(narrow_j_s_L.or.narrow_j_n_L)) then
            i_near = i0_e_i
            call RPN_COMM_allreduce(i_near,i_near_E,1,"mpi_integer","mpi_min","GRID",err)
         endif

         !Initialize i_near_W = in_w_i of PE at W 
         !---------------------------------------
         if (narrow_i_w_L.and.(narrow_j_s_L.or.narrow_j_n_L)) then
            i_near = in_w_i
            call RPN_COMM_allreduce(i_near,i_near_W,1,"mpi_integer","mpi_min","GRID",err)
         endif

         done_L = .TRUE.

      endif !DONE ALLOCATION

      !Pre-compute indices ii used in: adv_tricub_lag3d_loop
      !-----------------------------------------------------
      if (.NOT.Adv_done_precompute_L) then

         !Exchange HALO_E for positions
         !-----------------------------
         if (narrow_L) then

!$omp parallel private(i,j,k,n,ne)     &
!$omp shared (px,py,pz,F_ni,F_nj,F_nk, &
!$omp F_minx_e,F_maxx_e,F_miny_e,F_maxy_e) 
!$omp do 
            do k=k0,F_nk
               do j=1,F_nj
                  do i=1,F_ni

                     n  = (k-1)*F_ni*F_nj + (j-1)*F_ni + i
                     ne = (k-1)*(F_maxy_e-F_miny_e+1)*(F_maxx_e-F_minx_e+1) + (j-F_miny_e)*(F_maxx_e-F_minx_e+1) + i-F_minx_e + 1

                     px(ne) = F_x(n)
                     py(ne) = F_y(n)
                     pz(ne) = F_z(n)

                  end do
               end do
            end do
!$omp end do
!$omp end parallel

            call rpn_comm_xch_halo(px,F_minx_e,F_maxx_e,F_miny_e,F_maxy_e,F_ni,F_nj,F_nk,1-F_minx_e,1-F_miny_e,.false.,.false.,F_ni,0)
            call rpn_comm_xch_halo(py,F_minx_e,F_maxx_e,F_miny_e,F_maxy_e,F_ni,F_nj,F_nk,1-F_minx_e,1-F_miny_e,.false.,.false.,F_ni,0)
            call rpn_comm_xch_halo(pz,F_minx_e,F_maxx_e,F_miny_e,F_maxy_e,F_ni,F_nj,F_nk,1-F_minx_e,1-F_miny_e,.false.,.false.,F_ni,0)

            !Clipping to limit the upstream positions (NOTE: This is done with respect to ADV_HALO)
            !--------------------------------------------------------------------------------------
            call adv_cliptraj_h (px,py,F_minx_e,F_maxx_e,F_miny_e,F_maxy_e,F_ni,F_nj,F_nk,i0_e,in_e,j0_e,jn_e,k0,'FLUX')

         endif

         if (l_west) then

            call adv_get_indices (ii_w_o, F_x, F_y, F_z, F_num , nind_w_o, &
                                  i0_w_o, in_w_o, j0_w_o, jn_w_o, k0 , l_nk, 't')


            if (.not.narrow_i_w_L) & 
            call adv_get_indices (ii_w_i, F_x, F_y, F_z, F_num , nind_w_i, &
                                  i0_w_i, in_w_i, j0_w_i, jn_w_i, k0 , l_nk, 't')

         endif

         if (l_south) then

            call adv_get_indices (ii_s_o, F_x, F_y, F_z, F_num , nind_s_o, &
                                  i0_s_o, in_s_o, j0_s_o, jn_s_o, k0 , l_nk, 't')

            if (.not.narrow_j_s_L) & 
            call adv_get_indices (ii_s_i, F_x, F_y, F_z, F_num , nind_s_i, &
                                  i0_s_i, in_s_i, j0_s_i, jn_s_i, k0 , l_nk, 't')

         endif

         if (l_east) then

            call adv_get_indices (ii_e_o, F_x, F_y, F_z, F_num , nind_e_o, &
                                  i0_e_o, in_e_o, j0_e_o, jn_e_o, k0 , l_nk, 't')

           
            if (.not.narrow_i_e_L) & 
            call adv_get_indices (ii_e_i, F_x, F_y, F_z, F_num , nind_e_i, &
                                  i0_e_i, in_e_i, j0_e_i, jn_e_i, k0 , l_nk, 't')

         endif

         if (l_north) then

            call adv_get_indices (ii_n_o, F_x, F_y, F_z, F_num , nind_n_o, &
                                  i0_n_o, in_n_o, j0_n_o, jn_n_o, k0 , l_nk, 't')

            if (.not.narrow_j_n_L) & 
            call adv_get_indices (ii_n_i, F_x, F_y, F_z, F_num , nind_n_i, &
                                  i0_n_i, in_n_i, j0_n_i, jn_n_i, k0 , l_nk, 't')

         endif

         Adv_done_precompute_L = .TRUE.

      endif !DONE PRECOMPUTE

      !-------------------------
      !Estimate FLUX_out/FLUX_in
      !-------------------------
      F_cub_o = 0.0
      F_cub_i = 0.0

      nind_o = -1
      nind_i = -1

      i_h = 0.0

      if (l_west) then

         nind_o = nind_w_o
         nind_i = nind_w_i

         ii_o => ii_w_o
         ii_i => ii_w_i

         i0_i = i0_w_i
         in_i = in_w_i
         j0_i = j0_w_i
         jn_i = jn_w_i

         if (narrow_i_w_L) then
#include "adv_tricub_lag3d_flux_loop_iH.cdk"
         endif

      endif

#include "adv_tricub_lag3d_flux_loop_o.cdk"
      if (.not.narrow_i_w_L) then
#include "adv_tricub_lag3d_flux_loop_i.cdk"
      endif

      nind_o = -1
      nind_i = -1

      if (l_south) then

         nind_o = nind_s_o
         nind_i = nind_s_i

         ii_o => ii_s_o
         ii_i => ii_s_i

         i0_i = i0_s_i
         in_i = in_s_i
         j0_i = j0_s_i
         jn_i = jn_s_i

         if (narrow_j_s_L) then
#include "adv_tricub_lag3d_flux_loop_iH.cdk"
        endif

      endif

#include "adv_tricub_lag3d_flux_loop_o.cdk"
      if (.NOT.narrow_j_s_L) then
#include "adv_tricub_lag3d_flux_loop_i.cdk"
      endif 

      nind_o = -1
      nind_i = -1

      if (l_east) then

         nind_o = nind_e_o
         nind_i = nind_e_i

         ii_o => ii_e_o
         ii_i => ii_e_i

         i0_i = i0_e_i
         in_i = in_e_i
         j0_i = j0_e_i
         jn_i = jn_e_i

         if (narrow_i_e_L) then
#include "adv_tricub_lag3d_flux_loop_iH.cdk"
         endif

      endif

#include "adv_tricub_lag3d_flux_loop_o.cdk"
      if (.NOT.narrow_i_e_L) then
#include "adv_tricub_lag3d_flux_loop_i.cdk"
      endif

      nind_o = -1
      nind_i = -1

      if (l_north) then

         nind_o = nind_n_o
         nind_i = nind_n_i

         ii_o => ii_n_o
         ii_i => ii_n_i

         i0_i = i0_n_i
         in_i = in_n_i
         j0_i = j0_n_i
         jn_i = jn_n_i

         if (narrow_j_n_L) then
#include "adv_tricub_lag3d_flux_loop_iH.cdk"
        endif

      endif

#include "adv_tricub_lag3d_flux_loop_o.cdk"
      if (.NOT.narrow_j_n_L) then
#include "adv_tricub_lag3d_flux_loop_i.cdk"
      endif

      if (l_west.and.     narrow_i_w_L.and.l_north.and..not.narrow_j_n_L) i_h(i0_w_i:in_w_i,j0_n_i:jn_n_i,k0:F_nk) = 0.
      if (l_east.and.     narrow_i_e_L.and.l_north.and..not.narrow_j_n_L) i_h(i0_e_i:in_e_i,j0_n_i:jn_n_i,k0:F_nk) = 0.
      if (l_west.and.     narrow_i_w_L.and.l_south.and..not.narrow_j_s_L) i_h(i0_w_i:in_w_i,j0_s_i:jn_s_i,k0:F_nk) = 0.
      if (l_east.and.     narrow_i_e_L.and.l_south.and..not.narrow_j_s_L) i_h(i0_e_i:in_e_i,j0_s_i:jn_s_i,k0:F_nk) = 0.

      if (l_west.and..not.narrow_i_w_L.and.l_north.and.     narrow_j_n_L) i_h(i0_w_i:in_w_i,j0_n_i:jn_n_i,k0:F_nk) = 0.
      if (l_east.and..not.narrow_i_e_L.and.l_north.and.     narrow_j_n_L) i_h(i0_e_i:in_e_i,j0_n_i:jn_n_i,k0:F_nk) = 0.
      if (l_west.and..not.narrow_i_w_L.and.l_south.and.     narrow_j_s_L) i_h(i0_w_i:in_w_i,j0_s_i:jn_s_i,k0:F_nk) = 0.
      if (l_east.and..not.narrow_i_e_L.and.l_south.and.     narrow_j_s_L) i_h(i0_e_i:in_e_i,j0_s_i:jn_s_i,k0:F_nk) = 0.

      if (l_west.and.     narrow_i_w_L.and.l_north.and.     narrow_j_n_L) i_h(l_ni+1:in_w_i,     1:jn_n_i,k0:F_nk) = 0.
      if (l_west.and.     narrow_i_w_L.and.l_north.and.     narrow_j_n_L) i_h(i0_w_i:  l_ni,j0_n_i:     0,k0:F_nk) = 0.

      if (l_east.and.     narrow_i_e_L.and.l_north.and.     narrow_j_n_L) i_h(i0_e_i:     0,     1:jn_n_i,k0:F_nk) = 0.
      if (l_east.and.     narrow_i_e_L.and.l_north.and.     narrow_j_n_L) i_h(     1:in_e_i,j0_n_i:     0,k0:F_nk) = 0.

      if (l_west.and.     narrow_i_w_L.and.l_south.and.     narrow_j_s_L) i_h(l_ni+1:in_w_i,j0_s_i:  l_nj,k0:F_nk) = 0.
      if (l_west.and.     narrow_i_w_L.and.l_south.and.     narrow_j_s_L) i_h(i0_w_i:  l_ni,l_nj+1:jn_s_i,k0:F_nk) = 0.

      if (l_east.and.     narrow_i_e_L.and.l_south.and.     narrow_j_s_L) i_h(i0_e_i:     0,j0_s_i:  l_nj,k0:F_nk) = 0.
      if (l_east.and.     narrow_i_e_L.and.l_south.and.     narrow_j_s_L) i_h(     1:in_e_i,l_nj+1:jn_s_i,k0:F_nk) = 0.

      if (num_PE_near_NW==Ptopo_myproc.and.narrow_i_w_L.and.narrow_j_n_L) i_h(            1:i_near_W-l_ni,j0_n_i:0,     k0:F_nk) = 0.

      if (num_PE_near_NE==Ptopo_myproc.and.narrow_i_e_L.and.narrow_j_n_L) i_h(i_near_E+l_ni:l_ni,         j0_n_i:0,     k0:F_nk) = 0.

      if (num_PE_near_SW==Ptopo_myproc.and.narrow_i_w_L.and.narrow_j_s_L) i_h(            1:i_near_W-l_ni,l_nj+1:jn_s_i,k0:F_nk) = 0.

      if (num_PE_near_SE==Ptopo_myproc.and.narrow_i_e_L.and.narrow_j_s_L) i_h(i_near_E+l_ni:l_ni,         l_nj+1:jn_s_i,k0:F_nk) = 0.

      if (narrow_L) then

         !Adjoint of Fill East-West/North-South Halo
         !------------------------------------------
         call rpn_comm_adj_halo (i_h,F_minx_e,F_maxx_e,F_miny_e,F_maxy_e,F_ni,F_nj,F_nk,1-F_minx_e,1-F_miny_e,.false.,.false.,F_ni,0)

         do k=1,F_nk
         do j=1,F_nj
         do i=1,F_ni
            n = (k-1)*F_ni*F_nj + (j-1)*F_ni + i
            F_cub_i(n) = i_h(i,j,k) + F_cub_i(n)
         enddo
         enddo
         enddo

      endif

      call timing_stop (77)
!
!---------------------------------------------------------------------
!
      return

      end subroutine adv_tricub_lag3d_flux
