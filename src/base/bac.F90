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
!---------------------------------- LICENCE END --------------------------------

!**s/r  bacp - backsubstitution: obtain new values of the variables:
!                                                   u,v,w,t,q,s,zd
!                from new P , the right-hand sides (Ru,Rv,Rt,Rw,Rf)
!                             and non-linear terms (Nu,Nv,Nt,Nw,Nf)
!revision
! v4_50 - Qaddouri/Lee      - Yin-Yang, to exchange ZD, winds,T,S,Q
! v4.7  - Gaudreault S.     - Reformulation in terms of real winds (removing wind images)
!


      subroutine bac ( F_lhs_sol, F_fis , &
                       F_u   , F_v     , F_w  , F_t       , &
                       F_s   , F_zd    , F_q  , F_nest_q  , &
                       F_ru  , F_rv    , F_rt , F_rw  , F_rf , F_rb, &
                       F_nu  , F_nv    , F_nt , F_nw  , F_nf , F_nb, &
                       F_xd  , F_qd    , F_rx , F_rq  , &
                       Minx,Maxx,Miny,Maxy, ni,nj,Nk, i0, j0, k0, in, jn )
      implicit none
#include <arch_specific.hf>
!
      integer  Minx,Maxx,Miny,Maxy, ni,nj,Nk , i0, j0, k0, in, jn
      real*8   F_lhs_sol (ni,nj,Nk)
      real     F_fis (Minx:Maxx,Miny:Maxy)                                    , &
               F_u   (Minx:Maxx,Miny:Maxy,  Nk)  , F_v     (Minx:Maxx,Miny:Maxy,  Nk)  , &
               F_w   (Minx:Maxx,Miny:Maxy,  Nk)  , F_t     (Minx:Maxx,Miny:Maxy,  Nk)  , &
               F_s   (Minx:Maxx,Miny:Maxy)       , F_zd    (Minx:Maxx,Miny:Maxy,  Nk)  , &
               F_q   (Minx:Maxx,Miny:Maxy,2:Nk+1), F_nest_q(Minx:Maxx,Miny:Maxy,2:Nk+1), &
               F_ru  (Minx:Maxx,Miny:Maxy,  Nk)  , F_rv    (Minx:Maxx,Miny:Maxy,  Nk)  , &
               F_rt  (Minx:Maxx,Miny:Maxy,  Nk)  , F_rw    (Minx:Maxx,Miny:Maxy,  Nk)  , &
               F_rf  (Minx:Maxx,Miny:Maxy,  Nk)  , F_rb    (Minx:Maxx,Miny:Maxy)       , &
               F_xd  (Minx:Maxx,Miny:Maxy,  Nk)  , F_qd    (Minx:Maxx,Miny:Maxy,  Nk)  , &
               F_rx  (Minx:Maxx,Miny:Maxy,  Nk)  , F_rq    (Minx:Maxx,Miny:Maxy,  Nk)  , &
               F_nu  (Minx:Maxx,Miny:Maxy,  Nk)  , F_nv    (Minx:Maxx,Miny:Maxy,  Nk)  , &
               F_nt  (Minx:Maxx,Miny:Maxy,  Nk)  , F_nw    (Minx:Maxx,Miny:Maxy,  Nk)  , &
               F_nf  (Minx:Maxx,Miny:Maxy,  Nk)  , F_nb    (Minx:Maxx,Miny:Maxy)
!
#include "glb_pil.cdk"
#include "glb_ld.cdk"
#include "lun.cdk"
#include "grd.cdk"
#include "cstv.cdk"
#include "dcst.cdk"
#include "geomg.cdk"
#include "type.cdk"
#include "ver.cdk"
#include "schm.cdk"
#include "ptopo.cdk"
#include "lam.cdk"
#include "wil_williamson.cdk"
!
      integer i, j, k, km, kq, nij, k0t, istat
      real*8  w1, w2, w3, Pbar, qbar
      real*8, dimension(i0:in,j0:jn):: xtmp_8, ytmp_8
      real  , dimension(:,:,:), allocatable :: GP
      real*8, parameter :: zero=0.d0, one=1.d0
!     __________________________________________________________________
!
      if (Schm_autobar_L.and.Williamson_case.eq.1) return

      if (Lun_debug_L) write(Lun_out,1000)

      allocate (GP(Minx:Maxx,Miny:Maxy,Nk+1))

      nij = (in - i0 + 1)*(jn - j0 + 1)

      k0t=k0
      if(Schm_opentop_L) k0t=k0-1

      do k=k0,l_nk
         do j= j0, jn
         do i= i0, in
            GP(i,j,k) = sngl(F_lhs_sol(i,j,k))
         enddo
         enddo
      end do
!
!$omp parallel private(w1,w2,w3,qbar,Pbar,km,kq,xtmp_8,ytmp_8)
!
!     Compute P at top and bottom
!     ~~~~~~~~~~~~~~~~~~~~~~~~~~~
!
      if ( Schm_opentop_L ) then
!$omp do
         do j= j0, jn
         do i= i0, in
            GP(i,j,k0-1) = Ver_alfat_8 * F_lhs_sol(i,j,k0) &
                         + Ver_cst_8*(F_rb(i,j)-F_nb(i,j))
         end do
         end do
!$omp enddo
      endif
!$omp do
      do j= j0, jn
      do i= i0, in
         GP(i,j,l_nk+1)  = Ver_alfas_8 * F_lhs_sol(i,j,l_nk)  &
                         - Ver_css_8*(F_rt(i,j,l_nk)-F_nt(i,j,l_nk))
      end do
      end do
!$omp enddo

!$omp single
      call rpn_comm_xch_halo(GP,l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj,G_nk+1, &
                  G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
!$omp end single

!     Compute U & V
!     ~~~~~~~~~~~~~
!$omp do
      do k=k0,l_nk
         do j= j0, jn
         do i= i0, l_niu-pil_e
            F_u(i,j,k) = Cstv_tau_8*(F_ru(i,j,k)-F_nu(i,j,k) - (GP(i+1,j,k)-GP(i,j,k))*geomg_invDXMu_8(j))
         end do
         end do

         do j= j0, l_njv-pil_n
         do i= i0, in
            F_v(i,j,k) = Cstv_tau_8*(F_rv(i,j,k)-F_nv(i,j,k) - (GP(i,j+1,k)-GP(i,j,k))*geomg_invDYMv_8(j))
         end do
         end do
      enddo
!$omp enddo

      if(.not.Schm_hydro_L.or.(Schm_hydro_L.and.(.not.Schm_nolog_L))) then
!
!        Compute w
!        ~~~~~~~~~
!$omp do
         do k=k0t,l_nk
            w1 = Cstv_tau_8*Dcst_Rgasd_8*Ver_Tstr_8(k)/Dcst_grav_8
            do j= j0, jn
            do i= i0, in
               Pbar= Ver_wp_8%t(k)*GP(i,j,k+1)+Ver_wm_8%t(k)*GP(i,j,k)
               F_w(i,j,k) = w1 * ( F_rf(i,j,k) - F_nf(i,j,k) &
               + Ver_gama_8(k) * ( (GP(i,j,k+1)-GP(i,j,k))*Ver_idz_8%t(k) &
                                        + Dcst_cappa_8 * Pbar ) )
            end do
            end do
         end do
!$omp enddo

      endif

      if(.not.Schm_hydro_L) then

!        Compute q
!        ~~~~~~~~~
!
!        N.B.  Top Boundary condition:
!                 Closed Top(k0.eq.1):  F_q(i,j,k0) = 0
!                   Open Top(k0.ne.1):  F_q(i,j,k0t) is externally specified

         if (Schm_opentop_L) then
!$omp do
            do j= j0, jn
            do i= i0, in
               F_q(i,j,k0t)=F_nest_q(i,j,k0t)
            end do
            end do
!$omp enddo
         endif
!
!        Note : we cannot use omp on loop k
!               due to vertical dependency F_q(i,j,k)
         do k=k0t,l_nk
            kq=max(k,2)
            w3 = one/(one+Ver_wp_8%t(k)*Ver_dz_8%t(k))
            w2 =  (one-Ver_wm_8%t(k)*Ver_dz_8%t(k))*w3
            w1 = Cstv_tau_8*Ver_igt_8*Ver_dz_8%t(k)*w3
!$omp do
            do j= j0, jn
            do i= i0, in
               F_q(i,j,k+1) = w2 * F_q(i,j,kq)*Ver_onezero(k)   &
                            - w1 * ( F_rw(i,j,k) - F_nw(i,j,k)  &
                                    - Cstv_invT_8 * F_w(i,j,k)  )
            end do
            end do
!$omp enddo
         end do

      endif

!     Compute s
!     ~~~~~~~~~
    
      w1 = one/(Dcst_Rgasd_8*Ver_Tstr_8(l_nk))
!$omp do
      do j= j0, jn
      do i= i0, in
         F_s(i,j) = w1*(GP(i,j,l_nk+1)-F_fis(i,j))-F_q(i,j,l_nk+1)
      end do
      end do
!$omp enddo

!     Compute zd
!     ~~~~~~~~~~

!        N.B.  Top Boundary condition:
!                 Closed Top(k0t.eq.1):  F_zd(i,j,k0t) = 0
!                   Open Top(k0t.ne.1):  F_zd(i,j,k0t) is computed

!$omp do
      do k=k0t,l_nk-1
         kq=max(k,2)
         w1=Ver_gama_8(k)*Ver_idz_8%t(k)
         w2=Ver_gama_8(k)*Ver_epsi_8(k)
         w3=Cstv_invT_8*Cstv_bar1_8
         do j= j0, jn
         do i= i0, in
            Pbar= Ver_wp_8%t(k)*GP(i,j,k+1)+Ver_wm_8%t(k)*GP(i,j,k)
            qbar=(Ver_wp_8%t(k)*F_q(i,j,k+1)+Ver_wm_8%t(k)*F_q(i,j,kq)*Ver_onezero(k))
            F_zd(i,j,k)=-Cstv_tau_8*( F_rt(i,j,k)- F_nt(i,j,k) &
                       + w1 * ( GP(i,j,k+1)-GP(i,j,k) ) - w2 * Pbar ) &
                       - w3 * ( Ver_b_8%t(k)*F_s(i,j)+qbar )
         enddo
         enddo
      enddo
!$omp enddo

      if(schm_nolog_L) then
!$omp do
         do k=k0t,l_nk
         do j= j0, jn
         do i= i0, in
            F_xd(i,j,k)=Cstv_invT_8*Ver_b_8%t(k)*F_s(i,j)+F_zd(i,j,k)-F_rx(i,j,k)
         end do
         end do
         end do
!$omp enddo
         if(.not.Schm_hydro_L) then
!$omp do
            do k=k0t,l_nk
            kq=max(k,2)
            do j= j0, jn
            do i= i0, in
               qbar=(Ver_wp_8%t(k)*F_q(i,j,k+1)+Ver_wm_8%t(k)*F_q(i,j,kq)*Ver_onezero(k))
               F_qd(i,j,k)=Cstv_invT_8*qbar-F_rq(i,j,k)
            end do
            end do
            end do
!$omp enddo
         endif
      endif

!     Compute FI' (into GP)
!     ~~~~~~~~~~~

!$omp do
      do k=k0t,l_nk
         km=max(k-1,1)
         kq=max(k,2)
         w1=Dcst_Rgasd_8*(Ver_wp_8%m(k)*Ver_Tstr_8(k)+Ver_wm_8%m(k)*Ver_Tstr_8(km))
         do j= j0, jn
         do i= i0, in
            GP(i,j,k)=GP(i,j,k)-w1*(Ver_b_8%m(k)*F_s(i,j)+F_q(i,j,kq)*Ver_onezero(k))
         enddo
         enddo
      enddo
!$omp enddo

      do j= j0, jn
      do i= i0, in
         GP(i,j,l_nk+1)=F_fis(i,j)
      enddo
      enddo

!     Compute T
!     ~~~~~~~~~

!$omp do
      do k=k0t,l_nk
         kq=max(k,2)
         do j= j0, jn
         do i= i0, in
            qbar=(Ver_wp_8%t(k)*F_q(i,j,k+1)+Ver_wm_8%t(k)*F_q(i,j,kq)*Ver_onezero(k))
            ytmp_8(i,j)=-qbar
         enddo
         enddo
         call vexp( xtmp_8, ytmp_8, nij )
         do j= j0, jn
         do i= i0, in
            xtmp_8(i,j)=xtmp_8(i,j)*(one+Ver_dbdz_8%t(k)*F_s(i,j))
         enddo
         enddo
         call vrec ( ytmp_8, xtmp_8, nij )
         w1=Ver_idz_8%t(k)/Dcst_rgasd_8
         do j= j0, jn
            do i= i0, in
               F_t(i,j,k)=ytmp_8(i,j)*(Ver_Tstr_8(k)-w1*(GP(i,j,k+1)-GP(i,j,k)))
            enddo
         enddo
      enddo
!$omp enddo

      if(Schm_nolog_L.and.Schm_hydro_L) then
!$omp do
         do k=k0t,l_nk
            do j= j0, jn
            do i= i0, in
               F_w(i,j,k)=-F_xd(i,j,k)*Dcst_rgasd_8*F_t(i,j,k)/Dcst_grav_8
            end do
            end do
         end do
!$omp enddo
      endif

      if(Schm_autobar_L) then
         F_t=Cstv_Tstr_8 ; F_zd=0. ! not necessary but safer
      endif

!$omp end parallel
!
      deallocate (GP)

      if (Grd_yinyang_L) then
         call yyg_nestuv(F_u,F_v, l_minx,l_maxx,l_miny,l_maxy, G_nk)
         call yyg_xchng (F_t , l_minx,l_maxx,l_miny,l_maxy, G_nk,&
                         .false., 'CUBIC')
         call yyg_xchng (F_zd, l_minx,l_maxx,l_miny,l_maxy, G_nk,&
                         .false., 'CUBIC')
         call yyg_xchng (F_s , l_minx,l_maxx,l_miny,l_maxy, 1   ,&
                         .false., 'CUBIC')
         if (.not.Schm_hydro_L) &
         call yyg_xchng (F_q , l_minx,l_maxx,l_miny,l_maxy, G_nk,&
                         .false., 'CUBIC')
      endif

1000  format (5X,'BACK SUBSTITUTION: (S/R BAC)')
!     __________________________________________________________________
!
      return
      end

