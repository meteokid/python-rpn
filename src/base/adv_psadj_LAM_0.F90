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
 
!**s/p adv_psadj_LAM_0 - Estimate FLUX_out/FLUX_in when LAM using Flux calculations based on Aranami et al. (2015) and
!                        call psadj_LAM to adjust surface pressure (NO LEGACY)

      subroutine adv_psadj_LAM_0 ()

      implicit none

#include <arch_specific.hf>

!Author Monique Tanguay
!
!revision
! v4_80 - Tanguay M.        - initial MPI version
!
!**/
#include "glb_ld.cdk"
#include "adv_grid.cdk"
#include "adv_pos.cdk"
#include "lam.cdk"

      integer :: i, j, k, k0, nbpts, i0_e, in_e, j0_e, jn_e, i_bidon, j_bidon
      real, pointer, dimension(:,:,:) :: cub_o,cub_i,w_cub_o_c,w_cub_i_c,adw_o,adw_i
      real, dimension(l_minx:l_maxx,l_miny:l_maxy,l_nk) :: in_o,in_i,mixing
      real,                   target :: no_conserv, no_slice, no_flux, no_advection
      integer,                target :: no_indices_1 
      integer,  dimension(1), target :: no_indices_2 
!     
!--------------------------------------------------------------------------------
!
      k0 = Lam_gbpil_t+1

      nbpts = l_ni*l_nj*l_nk

      !Establish scope of extended advection operations
      !------------------------------------------------
      call adv_get_ij0n_ext (i0_e,in_e,j0_e,jn_e)

      allocate (cub_o(l_minx:l_maxx,l_miny:l_maxy,l_nk), &
                cub_i(l_minx:l_maxx,l_miny:l_maxy,l_nk))

      !Initialize F_in_i/F_in_o/F_cub_i/F_cub_o using MIXING=1 
      !-------------------------------------------------------
      cub_o = 0.0 ; cub_i = 0.0

      mixing = 1.0

      call adv_set_flux_in (in_o,in_i,mixing,l_minx,l_maxx,l_miny,l_maxy,l_nk,k0)

      allocate (adw_o(adv_lminx:adv_lmaxx,adv_lminy:adv_lmaxy,l_nk), &
                adw_i(adv_lminx:adv_lmaxx,adv_lminy:adv_lmaxy,l_nk))

      allocate (w_cub_o_c(l_ni,l_nj,l_nk), &
                w_cub_i_c(l_ni,l_nj,l_nk))

      adw_o = 0.
      adw_i = 0.

      call rpn_comm_xch_halox( in_o, l_minx, l_maxx, l_miny, l_maxy , &
        l_ni, l_nj, l_nk, adv_halox, adv_haloy, G_periodx, G_periody , &
        adw_o, adv_lminx,adv_lmaxx,adv_lminy,adv_lmaxy, l_ni, 0)

      call rpn_comm_xch_halox( in_i, l_minx, l_maxx, l_miny, l_maxy , &
        l_ni, l_nj, l_nk, adv_halox, adv_haloy, G_periodx, G_periody , &
        adw_i, adv_lminx,adv_lmaxx,adv_lminy,adv_lmaxy, l_ni, 0)

      !Estimate FLUX_out/FLUX_in when LAM using Flux calculations based on Aranami et al. (2015)
      !-----------------------------------------------------------------------------------------
      call adv_tricub_lag3d (no_advection, no_conserv, no_conserv, no_conserv, no_conserv, no_advection, no_slice, 0, &
                             w_cub_o_c, adw_o, w_cub_i_c, adw_i, 2, pxt, pyt, pzt,                                    &
                             no_slice, no_slice, no_slice, no_slice, no_slice, no_slice, nbpts,                       &
                             no_indices_1, no_indices_2,  k0, l_nk, .false., .false., 't')

!$omp parallel
!$omp do
      do k = k0,l_nk
         cub_o(i0_e:in_e,j0_e:jn_e,k)= w_cub_o_c(i0_e:in_e,j0_e:jn_e,k)
         cub_i(i0_e:in_e,j0_e:jn_e,k)= w_cub_i_c(i0_e:in_e,j0_e:jn_e,k)
      enddo
!$omp enddo
!$omp end parallel

      call psadj_LAM (cub_o,cub_i,l_minx,l_maxx,l_miny,l_maxy,l_nk,k0)

      deallocate (cub_o,cub_i,adw_o,adw_i,w_cub_o_c,w_cub_i_c)

!---------------------------------------------------------------------
!
      return
      end subroutine adv_psadj_LAM_0
