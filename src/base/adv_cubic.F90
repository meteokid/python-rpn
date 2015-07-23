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
!

subroutine adv_cubic (F_name,fld_out ,fld_in, F_capx, F_capy, F_capz, &
                      F_ni, F_nj, F_nk, F_minx, F_maxx, F_miny, F_maxy, &
                      i0, in, j0, jn, k0, lev_S, mono_kind,mass_kind)

implicit none

#include <arch_specific.hf>

!
!@objective Interpolation of rhs
!
!@arguments
   character(len=*), intent(in) :: F_name
   integer, intent(in) :: F_ni,F_nj,F_nk                       ! dims of position fields
   integer, intent(in) :: F_minx,F_maxx,F_miny, F_maxy         ! wind fields array bounds
	logical :: F_is_mom_L                                       !I, momentum level if .true. (thermo if not)
	logical :: F_doAdwStat_L                                    !I, compute stats if .true.
   integer :: k0                                               !I, vertical scope k0 to F_nk
	real, intent(in)::  F_capx(*), F_capy(*), F_capz(*)         !I, upstream positions at t1

!
!@author alain patoine
!@revisions
! v2_31 - Desgagne & Tanguay  - removed stkmemw, introduce tracers
! v2_31                       - tracers not monotone in anal mode
! v3_00 - Desgagne & Lee      - Lam configuration
! v3_02 - Tanguay             - Restore tracers monotone in anal mode
! v3_02 - Lee V.              - revert adv_exch_1 for GLB only,
! v3_02                         added adv_ckbd_lam,adv_cfl_lam for LAM only
! v3_03 - Tanguay M.          - stop if adv_exch_1 is activated when anal mode
! v3_10 - Corbeil & Desgagne & Lee - AIXport+Opti+OpenMP
! v3_11 - Gravel S.           - introduce key adv_mono_L
! v3_20 - Gravel & Valin & Tanguay - Lagrange 3D
! v3_20 - Tanguay M.          - Improve alarm when points outside advection grid
! v3_20 - Dugas B.            - correct calculation for LAM when Glb_pil gt 7
! v3_21 - Desgagne M.         - if  Lagrange 3D, call adv_main_3_intlag
! v4_04 - Tanguay M.          - Staggered version TL/AD
! v4_05 - Lepine M.           - VMM replacement with GMM
! v1_10 - Plante A.           - Thermo upstream positions
! v4_40 - Tanguay M.          - Revision TL/AD
! v4_XX - Tanguay M.          - GEM4 Mass-Conservation
!**/

#include "msg.h"
#include "gmm.hf"
#include "adv_nml.cdk"
#include "schm.cdk"
#include "vt_tracers.cdk"
#include "ver.cdk"

	logical :: mono_L, conserv_L
   character(len=1) :: lev_S
   integer :: n,i0,j0,in,jn
   integer, intent(in) :: mono_kind      !I, Kind of Shape preservation
   integer, intent(in) :: mass_kind      !I, Kind of  Mass conservation
	logical, parameter :: EXTEND_L = .true.
   integer ::  i, j, k, nbpts
	real, dimension(F_ni,F_nj,F_nk) :: wrkc, w_mono_c, w_lin_c, w_min_c, w_max_c
	real, dimension(F_minx:F_maxx, F_miny:F_maxy ,F_nk), intent(in)  :: Fld_in
	real, dimension(F_minx:F_maxx, F_miny:F_maxy ,F_nk), intent(out) :: Fld_out
   type(gmm_metadata) :: mymeta
	real, dimension(1,1,1), target :: no_conserv
   integer :: err

     
        call msg(MSG_DEBUG,'adv_interp')

         mono_L = .false.

         if (F_name(1:3) == 'TR/') then
              mono_L = adw_mono_L
         endif
                
         conserv_L = mono_kind/=0.or.mass_kind/=0

         nbpts = F_ni*F_nj*F_nk

       if (conserv_L) then
       nullify(fld_cub,fld_mono,fld_lin,fld_min,fld_max)
        err = gmm_get(gmmk_cub_s ,fld_cub ,mymeta)
        err = gmm_get(gmmk_mono_s,fld_mono,mymeta)
        err = gmm_get(gmmk_lin_s ,fld_lin ,mymeta)
        err = gmm_get(gmmk_min_s ,fld_min ,mymeta)
        err = gmm_get(gmmk_max_s ,fld_max ,mymeta)
       else
        fld_cub  => no_conserv
        fld_mono => no_conserv
        fld_lin  => no_conserv
        fld_min  => no_conserv
        fld_max  => no_conserv
       endif

       if (adw_catmullrom_L) then
               call adv_tricub_catmullrom (wrkc, fld_in, F_capx, F_capy, F_capz, nbpts, &
                                           mono_L, i0,in,j0,jn,k0, F_nk, lev_S)
       else
               call adv_tricub_lag3d (wrkc, w_mono_c, w_lin_c, w_min_c, w_max_c, fld_in, F_capx, F_capy, F_capz, nbpts, &
                                      mono_L, conserv_L, i0,in,j0,jn,k0, F_nk, lev_S)
       end if

!$omp parallel
      if (.NOT. conserv_L) then
!$omp do
        do k = k0, F_nk
             Fld_out(i0:in,j0:jn,k) = wrkc(i0:in,j0:jn,k)
        enddo
!$omp enddo
      else
!$omp do
        do k = k0, F_nk
             Fld_cub (i0:in,j0:jn,k) = wrkc    (i0:in,j0:jn,k)
             Fld_mono(i0:in,j0:jn,k) = w_mono_c(i0:in,j0:jn,k)
             Fld_lin (i0:in,j0:jn,k) = w_lin_c (i0:in,j0:jn,k)
             Fld_min (i0:in,j0:jn,k) = w_min_c (i0:in,j0:jn,k)
             Fld_max (i0:in,j0:jn,k) = w_max_c (i0:in,j0:jn,k)
        enddo
!$omp enddo
 endif
!$omp end parallel

if (conserv_L) call adv_tracers_mono_mass (F_name, fld_out, fld_cub,fld_mono, fld_lin, fld_min, fld_max , fld_in , &
                                           F_minx, F_maxx , F_miny, F_maxy ,F_nk ,  & 
                                           i0, in ,j0 ,jn ,k0 , mono_kind, mass_kind )

 call msg(MSG_DEBUG,'advInterp_cubic [end]')

end subroutine adv_cubic