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
!**s/r prep - Add metric corrections to r.h.s. of momentum equations.
!               Compute advective contributions on geopotential grid.
!               Interpolate advection contribution from geopotential
!               grid to wind grids. Update r.h.s with advective
!               contributions.
!               Combine some rhs obtaining Rt", Rf" and Rc", the linear
!               contributions to the rhs of Helmholtz equation
!
      subroutine pre ( F_ru  ,F_rv  ,F_ruw1 ,F_ruw2 ,F_rvw1 ,F_rvw2, &
                       F_fis  ,F_rc   ,F_rt  , &
                       F_rw  ,F_rf  ,F_oru  ,F_orv , &
                       F_rb,F_nest_t, Minx,Maxx,Miny,Maxy, &
                       i0, j0, in, jn, k0, ni, nj, Nk )
      implicit none
#include <arch_specific.hf>

      integer Minx,Maxx,Miny,Maxy, i0, j0, in, jn, k0, ni, nj, Nk
      real F_ru    (Minx:Maxx,Miny:Maxy,Nk)  ,F_rv    (Minx:Maxx,Miny:Maxy,Nk)  , &
           F_ruw1  (Minx:Maxx,Miny:Maxy,Nk)  ,F_ruw2  (Minx:Maxx,Miny:Maxy,Nk)  , &
           F_rvw1  (Minx:Maxx,Miny:Maxy,Nk)  ,F_rvw2  (Minx:Maxx,Miny:Maxy,Nk)  , &
           F_rc    (Minx:Maxx,Miny:Maxy,Nk)  ,F_rt    (Minx:Maxx,Miny:Maxy,Nk)  , &
           F_rw    (Minx:Maxx,Miny:Maxy,Nk)  ,F_rf    (Minx:Maxx,Miny:Maxy,Nk)  , &
           F_oru   (Minx:Maxx,Miny:Maxy,Nk)  ,F_orv   (Minx:Maxx,Miny:Maxy,Nk)  , &
           F_rb    (Minx:Maxx,Miny:Maxy)     ,F_nest_t(Minx:Maxx,Miny:Maxy,Nk)  , &
           F_fis   (Minx:Maxx,Miny:Maxy)                                        

!author
!     Alain Patoine
!revision
! v2_00 - Desgagne M.       - initial MPI version (from rhs v1_03)
! v2_21 - Lee V.            - modification for LAM version
! v2_31 - Desgagne M.       - remove stkmemw and switch to adw_*
! v3_00 - Desgagne & Lee    - Lam configuration
! v3_10 - Corbeil & Desgagne & Lee - AIXport+Opti+OpenMP
! v3_11 - Gravel S.         - modify for theoretical cases
! v4_00 - Plante & Girard   - Log-hydro-pressure coord on Charney-Phillips grid
! v4_05 - Girard C.         - Open top
! v4_40 - Qaddouri/Lee      - expand range of calculation for Yin-Yang only
! v4.70 - Gaudreault S.     - Reformulation in terms of real winds (removing wind images)
! v4.80 - Lee V.            - correction in range for xch halo on Ru, Rv

#include "gmm.hf"
#include "glb_ld.cdk"
#include "lun.cdk"
#include "lam.cdk"
#include "cstv.cdk"
#include "dcst.cdk"
#include "grd.cdk"
#include "geomg.cdk"
#include "schm.cdk"
#include "nest.cdk"
#include "ver.cdk"

      integer :: i0u, inu, j0v, jnv
      integer :: i, j, k, km, k0t
      real*8  x, y, z, cx, cy, cz, rx, ry, rz, mumu, &
              rdiv, w1, w2, w3, w4, w5
      real    w_rt
      real*8, parameter :: zero=0.d0, one=1.d0 , &
                           alpha1=-1.d0/16.d0 , alpha2=9.d0/16.d0
!
!     ---------------------------------------------------------------
!  
      if (Lun_debug_L) write (Lun_out,1000)

      k0t= k0
      if(Schm_opentop_L) k0t= k0-1

      call rpn_comm_xch_halo( F_ru, l_minx,l_maxx,l_miny,l_maxy, l_niu, l_nj,G_nk, &
                              G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      call rpn_comm_xch_halo( F_rv, l_minx,l_maxx,l_miny,l_maxy, l_ni, l_njv,G_nk, &
                              G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )

!*************************************
! Combination of governing equations *
!*************************************

!$omp parallel private(km,rdiv,w1,w2,w3,w4,w5,w_rt)

!$omp do
      do k=k0, l_nk
         km=max(k-1,1)
         w1= Ver_igt_8*Ver_wpA_8(k)
         w2= Ver_igt_8*Ver_wmA_8(k)*Ver_onezero(k)
         do j= j0, jn
         do i= i0, in

!           Combine continuity & w equations
!           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            F_rc(i,j,k) = F_rc(i,j,k) - w1*F_rw(i,j,k ) - w2*F_rw(i,j,km)

!           Compute the divergence of the RHS of momentum equations
!           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            rdiv  = (F_ru(i,j,k)-F_ru(i-1,j,k))*geomg_invDXM_8(j) &
                  + (F_rv(i,j,k)*geomg_cyM_8(j)-F_rv(i,j-1,k)*geomg_cyM_8(j-1))*geomg_invDYM_8(j)

!           Combine divergence & continuity equations : Rc"
!           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            F_rc(i,j,k) = rdiv - F_rc(i,j,k) / Cstv_tau_m_8

         end do
         end do

      end do
!$omp enddo

      if (Schm_opentop_L) then
         w3=one
         w4=Dcst_rgasd_8*Ver_Tstar_8%t(k0t)
!$omp do
         do j= j0, jn
         do i= i0, in
            F_rt(i,j,k0t)=F_rt(i,j,k0t)*w3
            F_rb(i,j) = F_rt(i,j,k0t)
         end do
         end do
!$omp enddo
      endif

!$omp do
      do k=k0t,l_nk

!        Compute Rt" & Rf"
!        ~~~~~~~~~~~~~~~~~

         w1 = Dcst_cappa_8 /( Dcst_Rgasd_8 * Ver_Tstar_8%t(k) )
         w2 = Cstv_invT_m_8 / ( Dcst_cappa_8 + Ver_epsi_8(k) )
         do j= j0, jn
         do i= i0, in
!           Combine Rt and Rw
!           ~~~~~~~~~~~~~~~~~
            w_rt = F_rt(i,j,k) + Ver_igt_8 * F_rw(i,j,k)

!           Compute Rt"
!           ~~~~~~~~~~~
            F_rt(i,j,k) = w2 * ( w_rt + Ver_igt2_8 * F_rf(i,j,k) )

!           Compute Rf"
!           ~~~~~~~~~~~
            F_rf(i,j,k) = w2 * ( w_rt - w1 * F_rf(i,j,k) )
         end do
         end do

      enddo
!$omp enddo

!$omp do
      do j= j0, jn
      do i= i0, in
!        Adjust Rt" at last level : Rt"'
!        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         F_rt(i,j,l_nk) = (F_rt(i,j,l_nk)-Ver_wmstar_8(l_nk)*F_rt(i,j,l_nk-1)) &
                          /Ver_wpstar_8(l_nk)
      end do
      end do
!$omp enddo

!************************************************************
! The linear contributions to the RHS of Helmholtz equation *
!************************************************************

!     Finish computations of RP(in Rc), combining Rc", Rt", Rf"
!     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

!$omp do
      do k=k0,l_nk
         km=max(k-1,1)
         w1= Ver_idz_8%m(k) + Ver_wp_8%m(k)
         w2=(Ver_idz_8%m(k) - Ver_wm_8%m(k))*Ver_onezero(k)
         w3=Ver_wpA_8(k)*Ver_epsi_8(k)
         w4=Ver_wmA_8(k)*Ver_epsi_8(km)*Ver_onezero(k)
         do j= j0, jn
         do i= i0, in
            F_rc(i,j,k) = F_rc(i,j,k) - Cstv_bar0_8 * F_fis(i,j) &
                           - w1 * F_rt(i,j,k) + w2 * F_rt(i,j,km) &
                           - w3 * F_rf(i,j,k) - w4 * F_rf(i,j,km)
         end do
         end do
      end do
!$omp enddo

      if (Schm_opentop_L) then

!        Apply opentop boundary conditions

         w1=Cstv_invT_8/Ver_Tstar_8%t(k0t)
!$omp do
         do j= j0, jn
         do i= i0, in
            F_rb(i,j) = F_rt(i,j,k0t) - Ver_ikt_8*(F_rb(i,j) - w1*(F_nest_t(i,j,k0t)-Ver_Tstar_8%t(k0t)))
            F_rc(i,j,k0  ) = F_rc(i,j,k0  ) - Ver_cstp_8 * F_rb(i,j)
         end do
         end do
!$omp enddo

      endif

!     Apply lower boundary conditions
!
      w1 = Cstv_invT_8*Cstv_invT_m_8 / ( Dcst_Rgasd_8 * Ver_Tstar_8%m(l_nk+1) )
!$omp do
      do j= j0, jn
      do i= i0, in
         F_rt(i,j,l_nk) = F_rt(i,j,l_nk) - w1 * F_fis(i,j)
         F_rt(i,j,l_nk) = Ver_wpstar_8(l_nk) * F_rt(i,j,l_nk)
         F_rc(i,j,l_nk) = F_rc(i,j,l_nk) + Ver_cssp_8 * F_rt(i,j,l_nk)
      end do
      end do
!$omp enddo
!$omp end parallel

1000  format(3X,'UPDATE  THE RIGHT-HAND-SIDES: (S/R PRE)')
!
!     ---------------------------------------------------------------
!  
      return
      end
