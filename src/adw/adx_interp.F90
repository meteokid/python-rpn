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

!*
subroutine adx_interp7 ( F_out, F_cub, F_mono, F_lin, F_min, F_max, F_in, F_cub_o, F_in_o, F_cub_i, F_in_i, F_flux_n, F_c1, &
                         F_capx1, F_capy1, F_capz1, &
                         F_capx2, F_capy2, F_capz2, &
                         Minx,Maxx,Miny,Maxy,F_nk,  &
                         F_wind_L,F_mono_L,F_clip_positive_L,F_conserv_L,i0,in,j0,jn,k0, F_lev_S)
   implicit none
#include <arch_specific.hf>
!
   !@objective
!
   !@arguments
   character (len=*), intent(in) :: F_lev_S !I, m/t : Momemtum/thermo level
   integer, intent(in) :: Minx,Maxx,Miny,Maxy
   integer, intent(in) :: i0,in,j0,jn,k0    !I, scope of operator
   integer, intent(in) :: F_nk              !I, number of vertical levels
   integer, intent(in) :: F_c1(*)
   logical, intent(in) :: F_mono_L          !I, .true. monotonic interpolation
   logical, intent(in) :: F_clip_positive_L !I, .true. positive advection
   logical, intent(in) :: F_conserv_L       !I, .true. conservation
   logical, intent(in) :: F_wind_L          !I, .true. if field is wind like
   integer, intent(in) :: F_flux_n          ! 0=NO FLUX; 1=TRACER+FLUX; 2=FLUX only
   real, intent(in)    :: F_capx1(*), F_capy1(*), F_capz1(*)
   real, intent(in)    :: F_capx2(*), F_capy2(*), F_capz2(*)
   real, dimension(Minx:Maxx,Miny:Maxy,F_nk), intent(in) :: F_in, F_in_o, F_in_i
   real, dimension(Minx:Maxx,Miny:Maxy,F_nk), intent(out) :: F_out, F_cub, F_mono, F_lin, F_min, F_max, F_cub_o, F_cub_i
!
   !@author alain patoine
   !@revisions
   ! v2_31 - Tanguay M.        - correction parameters adx_vder
   ! v3_00 - Desgagne & Lee    - Lam configuration
   ! v3_10 - Corbeil & Desgagne & Lee - AIXport+Opti+OpenMP
   ! v3_20 - Gravel & Valin & Tanguay - Lagrange 3D
   ! v3_21 - Desgagne M.       - Revision Openmp
   ! v3_30 - McTaggart-Cowan   - Add truncated lag3d interpolator
   ! v3_30 - McTaggart-Cowan   - Vectorization subroutines *_vec
   ! v3_30 - Tanguay M.        - adjust OPENMP for LAM
   ! v4_06 - Gaudreault S.     - Code optimization, Positivity-preserving advection
   ! v4_80 - Tanguay M.        - GEM4 Mass-Conservation and FLUX calculations 
!**/

#include "adx_dims.cdk"
#include "adx_poles.cdk"
#include "adx_nml.cdk"

   logical, parameter :: EXTEND_L = .true.
   integer :: i, j, k, istat, nbpts, i0_e, in_e, j0_e, jn_e
   real    :: dummy
   real    :: fld_adw(adx_lminx:adx_lmaxx,adx_lminy:adx_lmaxy,F_nk)
   real, dimension(:), pointer             :: wrka
   real, dimension(adx_mlni,adx_mlnj,F_nk) :: wrkb, wrkc
   real, dimension(:), pointer             :: w_mono_a, w_lin_a, w_min_a, w_max_a
   real, dimension(adx_mlni,adx_mlnj,F_nk) :: w_mono_b, w_lin_b, w_min_b, w_max_b, &
                                              w_mono_c, w_lin_c, w_min_c, w_max_c
   real, dimension(:,:,:), pointer         :: adw_o,adw_i,w_cub_o_c, w_cub_i_c
   real, dimension(1,1,1), target          :: no_flux
   real, dimension(1),     target          :: no_conserv

   !---------------------------------------------------------------------
   nullify(wrka)
   call msg(MSG_DEBUG,'adx_interp')
   if (.not.adx_lam_L) allocate (wrka(max(1,adx_fro_a)))

   if (.not.adx_lam_L) then
      if (F_conserv_L) then
         allocate (w_mono_a(max(1,adx_fro_a)),w_lin_a(max(1,adx_fro_a)), &
                   w_min_a (max(1,adx_fro_a)),w_max_a(max(1,adx_fro_a)))
      else
         w_mono_a => no_conserv
         w_lin_a  => no_conserv
         w_min_a  => no_conserv
         w_max_a  => no_conserv
      endif
   endif

   call adx_grid_scalar (fld_adw, F_in, adx_lminx,adx_lmaxx,adx_lminy,adx_lmaxy,&
                                   Minx,Maxx,Miny,Maxy, F_nk, F_wind_L, EXTEND_L)

   if (F_flux_n>0) then

      !Establish scope of extended advection operations
      !------------------------------------------------
      call adx_get_ij0n_ext (i0_e,in_e,j0_e,jn_e)

      !Initialize F_in_i/F_in_o/F_cub_i/F_cub_o
      !----------------------------------------
      F_cub_o = 0.0 ; F_cub_i = 0.0

      call adx_set_flux_in (F_in_o,F_in_i,F_in,Minx,Maxx,Miny,Maxy,F_nk,k0)

      allocate (adw_o(adx_lminx:adx_lmaxx,adx_lminy:adx_lmaxy,F_nk), &
                adw_i(adx_lminx:adx_lmaxx,adx_lminy:adx_lmaxy,F_nk))

      allocate (w_cub_o_c(adx_mlni,adx_mlnj,F_nk), &
                w_cub_i_c(adx_mlni,adx_mlnj,F_nk))

      adw_o = 0.
      adw_i = 0.

      call adx_grid_scalar (adw_o,F_in_o,adx_lminx,adx_lmaxx,adx_lminy,adx_lmaxy,&
                                   Minx,Maxx,Miny,Maxy, F_nk, F_wind_L, EXTEND_L)

      call adx_grid_scalar (adw_i,F_in_i,adx_lminx,adx_lmaxx,adx_lminy,adx_lmaxy,&
                                   Minx,Maxx,Miny,Maxy, F_nk, F_wind_L, EXTEND_L)

   else
 
        adw_o   => no_flux
        adw_i   => no_flux
      w_cub_o_c => no_flux
      w_cub_i_c => no_flux

   endif

   nbpts = adx_mlni*adx_mlnj*F_nk

   if (adw_catmullrom_L) then
      call adx_tricub_catmullrom (wrkc, fld_adw, F_capx1, F_capy1, F_capz1, nbpts, &
                                  F_mono_L, i0,in,j0,jn,k0, F_nk, F_lev_S)
   else
      call adx_tricub_lag3d7 (wrkc, w_mono_c, w_lin_c, w_min_c, w_max_c, fld_adw, &
                              w_cub_o_c, adw_o, w_cub_i_c, adw_i, F_flux_n,       &
                              F_capx1, F_capy1, F_capz1, nbpts, &
                              F_mono_L, F_conserv_L, i0,in,j0,jn,k0, F_nk, F_lev_S)
   end if

   if (.not. adx_lam_L) then
      if (adx_fro_a > 0 ) then
         call adx_tricub_lag3d7 (wrka, w_mono_a, w_lin_a, w_min_a, w_max_a, fld_adw,   &
                                                no_flux, no_flux, no_flux, no_flux, 0, &
                                                F_capx2, F_capy2, F_capz2, adx_fro_a,  &
                                                F_mono_L, F_conserv_L, 1,adx_fro_a,1,1,1,1, F_lev_S)
      endif

      if (.NOT.F_conserv_L) then
      call adx_exch_2 ( wrkb, dummy, dummy, dummy, dummy, &
                        wrka, dummy, dummy, dummy, dummy, &
           adx_for_n, adx_for_s, adx_for_a, &
           adx_fro_n, adx_fro_s, adx_fro_a, 1)
      else
      call adx_exch_2 ( wrkb, w_mono_b, w_lin_b, w_min_b, w_max_b, &
                        wrka, w_mono_a, w_lin_a, w_min_a, w_max_a, &
           adx_for_n, adx_for_s, adx_for_a, &
           adx_fro_n, adx_fro_s, adx_fro_a, 5)
      endif

      if (adx_for_a > 0.and..NOT.F_conserv_L) &
           call adx_exch_3b (wrkc, dummy, dummy, dummy, dummy, &
                             wrkb, dummy, dummy, dummy, dummy, F_c1, 1,adx_mlni,adx_mlnj,F_nk)

      if (adx_for_a > 0.and.F_conserv_L) &
           call adx_exch_3b (wrkc, w_mono_c, w_lin_c , w_min_c , w_max_c , &
                             wrkb, w_mono_b, w_lin_b , w_min_b , w_max_b, F_c1, 5,adx_mlni,adx_mlnj,F_nk)

   endif

   if (F_clip_positive_L) then
!$omp parallel
!$omp do
   do k = k0, F_nk
      where (wrkc(i0:in,j0:jn,k) < 0.)
         F_out(i0:in,j0:jn,k) = 0.
      elsewhere
         F_out(i0:in,j0:jn,k) = wrkc(i0:in,j0:jn,k)
      end where
   enddo
!$omp enddo
!$omp end parallel
   else
!$omp parallel
   if (.NOT.F_conserv_L) then 
!$omp do
   do k = k0, F_nk
      F_out(i0:in,j0:jn,k) = wrkc(i0:in,j0:jn,k)
   enddo
!$omp enddo
   else
!$omp do
   do k = k0, F_nk
      F_cub (i0:in,j0:jn,k) = wrkc    (i0:in,j0:jn,k)
      F_mono(i0:in,j0:jn,k) = w_mono_c(i0:in,j0:jn,k)
      F_lin (i0:in,j0:jn,k) = w_lin_c (i0:in,j0:jn,k)
      F_min (i0:in,j0:jn,k) = w_min_c (i0:in,j0:jn,k)
      F_max (i0:in,j0:jn,k) = w_max_c (i0:in,j0:jn,k)

      if (F_flux_n>0) then
      F_cub_o(i0_e:in_e,j0_e:jn_e,k)= w_cub_o_c(i0_e:in_e,j0_e:jn_e,k)
      F_cub_i(i0_e:in_e,j0_e:jn_e,k)= w_cub_i_c(i0_e:in_e,j0_e:jn_e,k)
      endif
   enddo
!$omp enddo
   endif 
!$omp end parallel
   endif

   if (.not.adx_lam_L) deallocate(wrka)
   if (.not.adx_lam_L.and.F_conserv_L) deallocate(w_mono_a,w_lin_a,w_min_a,w_max_a)
   if (F_flux_n>0) deallocate(adw_o,adw_i,w_cub_o_c,w_cub_i_c)
   call msg(MSG_DEBUG,'adx_interp [end]')

   !---------------------------------------------------------------------

   return
end subroutine adx_interp7
