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
 
!**s/p adx_psadj_LAM_0 - Estimate FLUX_out/FLUX_in when LAM using Flux calculations based on Aranami et al. (2015) and
!                        call psadj_LAM to adjust surface pressure (LEGACY)

      subroutine adx_psadj_LAM_0 ()

      implicit none

#include <arch_specific.hf>

!Author Monique Tanguay
!
!revision
! v4_80 - Tanguay M.        - initial MPI version
!
!**/
#include "adx_dims.cdk"
#include "adx_pos.cdk"

      logical, parameter :: EXTEND_L = .true.
      integer :: i, j, k, k0, F_nk, nbpts, i0_e, in_e, j0_e, jn_e
      real, pointer, dimension(:,:,:) :: cub_o,cub_i,w_cub_o_c,w_cub_i_c,adw_o,adw_i
      real, dimension(adx_mlminx:adx_mlmaxx,adx_mlminy:adx_mlmaxy,adx_lnk) :: in_o,in_i,mixing
      real,    target :: no_conserv,no_advection
      integer, target :: no_ind
!     
!--------------------------------------------------------------------------------
!
      k0 = adx_gbpil_t+1 

      F_nk = adx_lnk

      nbpts = adx_mlni*adx_mlnj*F_nk

      !Establish scope of extended advection operations
      !------------------------------------------------
      call adx_get_ij0n_ext (i0_e,in_e,j0_e,jn_e)

      allocate (cub_o(adx_mlminx:adx_mlmaxx,adx_mlminy:adx_mlmaxy,F_nk), &
                cub_i(adx_mlminx:adx_mlmaxx,adx_mlminy:adx_mlmaxy,F_nk))

      !Initialize F_in_i/F_in_o/F_cub_i/F_cub_o using MIXING=1 
      !-------------------------------------------------------
      cub_o = 0.0 ; cub_i = 0.0

      mixing = 1.0

      call adx_set_flux_in (in_o,in_i,mixing,adx_mlminx,adx_mlmaxx,adx_mlminy,adx_mlmaxy,F_nk,k0)

      allocate (adw_o(adx_lminx:adx_lmaxx,adx_lminy:adx_lmaxy,F_nk), &
                adw_i(adx_lminx:adx_lmaxx,adx_lminy:adx_lmaxy,F_nk))

      allocate (w_cub_o_c(adx_mlni,adx_mlnj,F_nk), &
                w_cub_i_c(adx_mlni,adx_mlnj,F_nk))

      adw_o = 0.
      adw_i = 0.

      call adx_grid_scalar (adw_o,in_o,adx_lminx,adx_lmaxx,adx_lminy,adx_lmaxy,&
                            adx_mlminx,adx_mlmaxx,adx_mlminy,adx_mlmaxy, F_nk, .false., EXTEND_L)

      call adx_grid_scalar (adw_i,in_i,adx_lminx,adx_lmaxx,adx_lminy,adx_lmaxy,&
                            adx_mlminx,adx_mlmaxx,adx_mlminy,adx_mlmaxy, F_nk, .false., EXTEND_L)

      !Estimate FLUX_out/FLUX_in when LAM using Flux calculations based on Aranami et al. (2015)
      !-----------------------------------------------------------------------------------------
      call adx_tricub_lag3d7 (no_advection, no_conserv, no_conserv, no_conserv, no_conserv, no_advection, &
                              w_cub_o_c, adw_o, w_cub_i_c, adw_i, 2,                                      &
                              pxt, pyt, pzt, nbpts,                                                       &
                              .false., .false., no_ind, no_ind, no_ind, no_ind, k0, F_nk, 't')

!$omp parallel
!$omp do
      do k = k0,F_nk
         cub_o(i0_e:in_e,j0_e:jn_e,k)= w_cub_o_c(i0_e:in_e,j0_e:jn_e,k)
         cub_i(i0_e:in_e,j0_e:jn_e,k)= w_cub_i_c(i0_e:in_e,j0_e:jn_e,k)
      enddo
!$omp enddo
!$omp end parallel

      call psadj_LAM (cub_o,cub_i,adx_mlminx,adx_mlmaxx,adx_mlminy,adx_mlmaxy,F_nk,k0)

      deallocate (cub_o,cub_i,adw_o,adw_i,w_cub_o_c,w_cub_i_c)

!---------------------------------------------------------------------
!
      return
      end subroutine adx_psadj_LAM_0
