!COMP_ARCH=intel13sp1u2 ; -suppress=-C
!COMP_ARCH=intel-2016.1.156; -suppress=-C

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
!/@*

subroutine adx_int_rhs ( F_px, F_py, F_pz, F_is_mom_L, F_doAdwStat_L, &
                         F_pxu,F_pyu,F_pzu,F_pxv,F_pyv,F_pzv, &
                                F_ni,F_nj,F_k0,F_nk)
   implicit none
#include <arch_specific.hf>
!
   !@objective Interpolation of rhs
!
   !@arguments
   logical :: F_is_mom_L     !I, momentum level if .true. (thermo if not)
   logical :: F_doAdwStat_L  !I, compute stats if .true.
   integer :: F_ni,F_nj,F_nk !I, pos array dims
   integer :: F_k0           !I, vertical scope F_k0 to F_nk
   real, dimension(F_ni,F_nj,F_nk) :: F_px, F_py, F_pz !I, upstream positions at t1
   real, dimension(F_ni,F_nj,F_nk) :: F_pxu, F_pyu, F_pzu
   real, dimension(F_ni,F_nj,F_nk) :: F_pxv, F_pyv, F_pzv
!
   !@author alain patoine
   !@revisions
   ! v2_31 - Desgagne & Tanguay  - removed stkmemw, introduce tracers
   ! v2_31                       - tracers not monotone in anal mode
   ! v3_00 - Desgagne & Lee      - Lam configuration
   ! v3_02 - Tanguay             - Restore tracers monotone in anal mode
   ! v3_02 - Lee V.              - revert adx_exch_1 for GLB only, 
   ! v3_02                         added adx_ckbd_lam,adx_cfl_lam for LAM only
   ! v3_03 - Tanguay M.          - stop if adx_exch_1 is activated when anal mode
   ! v3_10 - Corbeil & Desgagne & Lee - AIXport+Opti+OpenMP
   ! v3_11 - Gravel S.           - introduce key adx_mono_L 
   ! v3_20 - Gravel & Valin & Tanguay - Lagrange 3D
   ! v3_20 - Tanguay M.          - Improve alarm when points outside advection grid
   ! v3_20 - Dugas B.            - correct calculation for LAM when Glb_pil gt 7
   ! v3_21 - Desgagne M.         - if  Lagrange 3D, call adx_main_3_intlag
   ! v4_04 - Tanguay M.          - Staggered version TL/AD
   ! v4_05 - Lepine M.           - VMM replacement with GMM
   ! v1_10 - Plante A.           - Thermo upstream positions
   ! v4_40 - Tanguay M.          - Revision TL/AD
   ! v4_XX - Tanguay M.          - GEM4 Mass-Conservation
!*@/

#include "adx_nml.cdk"
#include "adx_dims.cdk"
#include "adx_poles.cdk"
#include "orh.cdk"
#include "schm.cdk"
#include "rhsc.cdk"

   integer, external :: adx_ckbd3

   character(len=1) :: level_type_S
   integer :: n,i0,j0,in,jn,i0u,inu,j0v,jnv,jext,istat
   integer, dimension(F_ni,F_nj,F_nk) :: exch_c1
   real   , dimension(F_ni,F_nj,F_nk) :: exch_n1, exch_xgg1, exch_xdd1, py_store
   real   , dimension(:), allocatable :: capx2,capy2,capz2
   real  :: dummy

   !---------------------------------------------------------------------

   call adx_get_ij0n (i0,in,j0,jn)

   if (adx_lam_L) then

      i0u= i0
      inu= in
      j0v= j0
      jnv= jn

      jext = 2
      if (adx_yinyang_L) jext = 0

      if (adx_is_west)  i0u= i0 + jext
      if (adx_is_east)  inu= in - jext
      if (adx_is_south) j0v= j0 + jext
      if (adx_is_north) jnv= jn - jext
  !   if (adx_is_west) print*,'in int_rhs, i0u,i0=',i0u,i0
  !   if (adx_is_east) print*,'in int_rhs, in,inu=',in,inu

   endif

   level_type_S = 't'
   if (F_is_mom_L) level_type_S = 'm'

   if (adx_lam_L) then

      if (F_doAdwStat_L) call adx_cfl_lam3 (F_px, F_py, F_pz, i0,in,j0,jn, &
                                            F_ni,F_nj,F_k0,F_nk, level_type_S)

      call adx_cliptraj3 ( F_px , F_py , i0, in, j0, jn, F_ni,F_nj,F_k0,F_nk,  &
                                        'INTERP '//trim(level_type_S))
      call adx_cliptraj3 ( F_pxu, F_pyu, i0u, inu, j0, jn, F_ni,F_nj,F_k0,F_nk,  &
                                        'INTERP '//trim(level_type_S))
      call adx_cliptraj3 ( F_pxv, F_pyv, i0, in, j0v, jnv, F_ni,F_nj,F_k0,F_nk,  &
                                        'INTERP '//trim(level_type_S))

   else

      py_store=F_py

      call adx_exch_1c ( exch_n1, exch_xgg1, exch_xdd1, exch_c1, &
                              F_px, F_py, F_pz, F_ni,F_nj,F_k0,F_nk )

      allocate ( capx2(max(1,adx_fro_a)), &
                 capy2(max(1,adx_fro_a)), &
                 capz2(max(1,adx_fro_a)) )

      call adx_exch_2 ( capx2, capy2, capz2, dummy, dummy,           &
                        exch_n1, exch_xgg1, exch_xdd1, dummy, dummy, &
                        adx_fro_n, adx_fro_s, adx_fro_a, &
                        adx_for_n, adx_for_s, adx_for_a, 3)

      istat = 1
      if (adx_fro_a>0 .and. adw_ckbd_L) &
      istat = adx_ckbd3 (capy2,adx_fro_n,adx_fro_s)

   endif

   if (F_is_mom_L) then
      if(adx_lam_L) then
         call adx_interp_gmm7 ( gmmk_rhsu_s, gmmk_orhsu_s , .true.     , &
                                F_pxu, F_pyu, F_pzu, capx2, capy2, capz2 , &
                                exch_c1, F_nk, i0u, inu, j0, jn, F_k0, 'm', 0, 0)
         call adx_interp_gmm7 ( gmmk_rhsv_s, gmmk_orhsv_s , .true.     , &
                                F_pxv, F_pyv, F_pzv, capx2, capy2, capz2 , &
                                exch_c1, F_nk, i0, in, j0v, jnv, F_k0, 'm', 0, 0)
      else
         call adx_interp_gmm7 ( gmmk_ruw2_s, gmmk_ruw1_s , .true.     , &
                                F_px, F_py, F_pz, capx2, capy2, capz2 , &
                                exch_c1, F_nk, i0, in, j0, jn, F_k0, 'm', 0, 0)
         call adx_interp_gmm7 ( gmmk_rvw2_s, gmmk_rvw1_s , .true.     , &
                                F_px, F_py, F_pz, capx2, capy2, capz2 , &
                                exch_c1, F_nk, i0, in, j0, jn, F_k0, 'm', 0, 0)
      endif
      call adx_interp_gmm7 ( gmmk_rhsc_s, gmmk_orhsc_s, .false.    , &
                             F_px, F_py, F_pz, capx2, capy2, capz2 , &
                             exch_c1, F_nk, i0, in, j0, jn, F_k0, 'm', 0, 0)
   else
      call adx_interp_gmm7 ( gmmk_rhst_s, gmmk_orhst_s, .false.   , &
                             F_px, F_py, F_pz, capx2, capy2, capz2, &
                             exch_c1, F_nk, i0, in, j0, jn, F_k0, 't', 0, 0)
      if(Schm_nologT_L) then
         call adx_interp_gmm7 ( gmmk_rhsx_s, gmmk_orhsx_s, .false.   , &
                                F_px, F_py, F_pz, capx2, capy2, capz2, &
                                exch_c1, F_nk, i0, in, j0, jn, F_k0, 't', 0, 0)
      endif
      if(.not.Schm_hydro_L.or.(Schm_hydro_L.and.(.not.Schm_nologT_L))) then
         call adx_interp_gmm7 ( gmmk_rhsf_s, gmmk_orhsf_s, .false.   , &
                                F_px, F_py, F_pz, capx2, capy2, capz2, &
                                exch_c1, F_nk, i0, in, j0, jn, F_k0, 't', 0, 0)
      endif
      if (.not. Schm_hydro_L) then
         call adx_interp_gmm7 ( gmmk_rhsw_s, gmmk_orhsw_s, .false.   , &
                                F_px, F_py, F_pz, capx2, capy2, capz2, &
                                exch_c1, F_nk, i0, in, j0, jn, F_k0, 't', 0, 0)
         if(Schm_nologT_L) then
            call adx_interp_gmm7 ( gmmk_rhsq_s, gmmk_orhsq_s, .false.   , &
                                   F_px, F_py, F_pz, capx2, capy2, capz2, &
                                   exch_c1, F_nk, i0, in, j0, jn, F_k0, 't', 0, 0)
         endif
      endif
   endif

   if (.not.adx_lam_L)then
      deallocate(capx2, capy2, capz2)
      if(adx_for_a > 0 ) F_py=py_store
   endif

   !---------------------------------------------------------------------
   return
end subroutine adx_int_rhs
