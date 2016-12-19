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
!**s/r nlip   - compute non-linear terms:  Nu, Nv, Nt, Nc, Nw, Nf,
!             - compute full right-hand side of Helmholtz eqn: Rp=Rc-Nc
!
!**********************************************************************
!
      subroutine nli ( F_nu , F_nv , F_nt   , F_nc , F_nw , F_nf  , F_nx , &
                       F_u  , F_v  , F_t    , F_s  , F_zd , F_q   , &
                       F_rhs, F_rc , F_fis  , F_nb , F_xd , F_qd  , F_hu , &
                       Minx,Maxx,Miny,Maxy, Nk , ni,nj, i0,j0,in,jn,k0, icln )
      implicit none
#include <arch_specific.hf>

      integer Minx,Maxx,Miny,Maxy, Nk,ni,nj,i0,j0,in,jn,k0, icln
      real    F_nu   (Minx:Maxx,Miny:Maxy,Nk)    ,F_nv   (Minx:Maxx,Miny:Maxy,Nk)    , &
              F_nt   (Minx:Maxx,Miny:Maxy,Nk)    ,F_nc   (Minx:Maxx,Miny:Maxy,Nk)    , &
              F_nw   (Minx:Maxx,Miny:Maxy,Nk)    ,F_nf   (Minx:Maxx,Miny:Maxy,Nk)    , &
              F_nx   (Minx:Maxx,Miny:Maxy,Nk)    , &
              F_u    (Minx:Maxx,Miny:Maxy,Nk)    ,F_v    (Minx:Maxx,Miny:Maxy,Nk)    , &
              F_t    (Minx:Maxx,Miny:Maxy,Nk)    ,F_s    (Minx:Maxx,Miny:Maxy)       , &
              F_zd   (Minx:Maxx,Miny:Maxy,Nk)    ,F_hu   (Minx:Maxx,Miny:Maxy,Nk)    , &
              F_q    (Minx:Maxx,Miny:Maxy,2:Nk+1),F_rc   (Minx:Maxx,Miny:Maxy,Nk)    , &
              F_xd   (Minx:Maxx,Miny:Maxy,Nk)    ,F_qd   (Minx:Maxx,Miny:Maxy,Nk)    , &
              F_fis  (Minx:Maxx,Miny:Maxy)       ,F_nb   (Minx:Maxx,Miny:Maxy)
      real*8  F_rhs  (ni,nj,Nk)

!author
!     Alain Patoine - split from nli.ftn
!
!revision
! v2_00 - Desgagne M.       - initial MPI version (from rhs v1_03)
! v2_21 - Lee V.            - modifications for LAM version
! v2_30 - Edouard S.        - adapt for vertical hybrid coordinate
!                             remove F_pptt and introduce Ncn
! v3_00 - Qaddouri & Lee    - For LAM, set Nu, Nv values on the boundaries
! v3_00                       of the LAM grid to zeros.
! v3_10 - Corbeil & Desgagne & Lee - AIXport+Opti+OpenMP
! v3_21 - Desgagne M.       - Revision OpenMP
! v4_00 - Plante & Girard   - Log-hydro-pressure coord on Charney-Phillips grid
! v4_05 - Girard C.         - Open top
! v4_40 - Lee/Qaddouri      - Adjust range of calculation for Yin-Yang
! v4.70 - Gaudreault S.     - Reformulation in terms of real winds (removing wind images)
!                           - Explicit integration of metric terms (optional)

#include "glb_ld.cdk"
#include "grd.cdk"
#include "cstv.cdk"
#include "dcst.cdk"
#include "geomg.cdk"
#include "cori.cdk"
#include "schm.cdk"
#include "inuvl.cdk"
#include "ptopo.cdk"
#include "ver.cdk"
#include "lun.cdk"

      logical, save :: done=.false.
      integer i, j, k, km,kmq,kq,kp, i0u, inu, j0v, jnv, nij, k0t, onept
      real    w_nt
      real*8  c1,qbar,ndiv,w1,w2,w3,w4,w5,barz,barzp,MUlin,dlnTstr_8, &
              t_interp, mu_interp, u_interp, v_interp, xdot, delta_8, &
              dBdzpBk,dBdzpBkm
      real*8  wk1(Minx:Maxx,Miny:Maxy), wk2(Minx:Maxx,Miny:Maxy)
      real*8 , dimension(i0:in,j0:jn) :: xtmp_8, ytmp_8
      real*8, parameter :: one=1.d0, half=0.5d0
      real, dimension(:,:,:), pointer :: BsPq, BsPrq, frq, FI, MU
      save MU
!     __________________________________________________________________
!
      if (Lun_debug_L)  write(Lun_out,1000)

      nullify  ( BsPq, BsPrq, frq, FI )
      allocate (  BsPq(Minx:Maxx,Miny:Maxy,l_nk+1), &
                 BsPrq(Minx:Maxx,Miny:Maxy,l_nk+1), &
                   frq(Minx:Maxx,Miny:Maxy,l_nk+1), &
                    FI(Minx:Maxx,Miny:Maxy,l_nk+1))

      if (.not.done) then
         nullify  ( MU )
         allocate ( MU(Minx:Maxx,Miny:Maxy,l_nk) )
      endif

      if(icln.gt.1) then
      call rpn_comm_xch_halo( F_u   ,l_minx,l_maxx,l_miny,l_maxy,l_niu,l_nj ,G_nk, &
                      G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      call rpn_comm_xch_halo( F_v   ,l_minx,l_maxx,l_miny,l_maxy,l_ni ,l_njv,G_nk, &
                      G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      call rpn_comm_xch_halo( F_t   ,l_minx,l_maxx,l_miny,l_maxy,l_ni ,l_nj ,G_nk, &
                      G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      call rpn_comm_xch_halo( F_s   ,l_minx,l_maxx,l_miny,l_maxy,l_ni ,l_nj ,1   , &
                      G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      if (.not.Schm_hydro_L) &
           call rpn_comm_xch_halo( F_q,l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj,G_nk, &
                      G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      endif

      c1 = Dcst_rayt_8**2

      delta_8 = 0.d0
      if(Schm_capa_var_L) delta_8 = 0.233d0

      k0t=k0
      if(Schm_opentop_L) k0t=k0-1
      nij = (in - i0 +1)*(jn - j0 +1)

      onept= 0
      if(Grd_yinyang_L) onept=1
!
!***********************************************************
! The nonlinear deviation of horizontal momentum equations *
!***********************************************************
!
!     Indices

      i0u = i0-1
      j0v = j0-1
      inu = l_niu-pil_e
      jnv = l_njv-pil_n

      if (l_west.and.G_lam) i0u=i0 -onept
      if (l_south)          j0v=j0 -onept
      if (l_east )          inu=inu+onept
      if (l_north)          jnv=jnv+onept

      call diag_fip( FI, F_s, F_t, F_q, F_fis, l_minx,l_maxx,l_miny,l_maxy,&
                                               l_nk, i0u, inu+1, j0v, jnv+1 )

      if (Schm_hydro_L) then
         if (.not.done) then
            MU= 0.
            done= .true.
         endif
      else
         call diag_mu ( MU, F_q, F_s, l_minx,l_maxx,l_miny,l_maxy, l_nk,&
                                                 i0u, inu+1, j0v, jnv+1 )
      endif

!$omp parallel private(km,kmq,kq,kp,w_nt,barz,barzp,ndiv,xdot,wk1,wk2, &
!$omp dlnTstr_8,w1,w2,w3,w4,w5,qbar,t_interp,u_interp,v_interp,xtmp_8,ytmp_8)

!$omp do
       do k=k0t,l_nk+1
          kq=max(2,k)
          do j=j0v,jnv+1
          do i=i0u,inu+1
             BsPq(i,j,k)  = Ver_b_8%m(k) * F_s(i,j) + F_q(i,j,kq)*Ver_onezero(k)
             BsPrq(i,j,k) = Ver_b_8%m(k) * F_s(i,j) + Cstv_rE_8*F_q(i,j,kq)*Ver_onezero(k)
             frq(i,j,k)   = F_q(i,j,kq)*Ver_onezero(k)

          enddo
          enddo
      enddo
!$omp enddo

!$omp do
      do k=k0,l_nk
      km=max(k-1,1)

!     Compute Nu
!     ~~~~~~~~~~

!     V barY stored in wk2
!     ~~~~~~~~~~~~~~~~~~~~
      if(Schm_cub_Coriolis_L) then
         do j = j0, jn
         do i = i0u-1, inu+2
            wk2(i,j) = inuvl_wyvy3_8(j,1) * F_v(i,j-2,k) &
                     + inuvl_wyvy3_8(j,2) * F_v(i,j-1,k) &
                     + inuvl_wyvy3_8(j,3) * F_v(i,j  ,k) &
                     + inuvl_wyvy3_8(j,4) * F_v(i,j+1,k)
         end do
         end do
         do j= j0, jn
         do i= i0u, inu
            wk1(i,j) = inuvl_wxxu3_8(i,1) * wk2(i-1,j) &
                     + inuvl_wxxu3_8(i,2) * wk2(i  ,j) &
                     + inuvl_wxxu3_8(i,3) * wk2(i+1,j) &
                     + inuvl_wxxu3_8(i,4) * wk2(i+2,j)
         end do
         end do
      else
         do j= j0, jn
         do i= i0u, inu
            wk1(i,j) = 0.25d0*(F_v(i,j,k)+F_v(i,j-1,k)+F_v(i+1,j,k)+F_v(i+1,j-1,k))
         end do
         end do
      endif

      
      do j= j0, jn
      do i= i0u, inu

!        mu barXZ
!        ~~~~~~~~
         barz  = Ver_wpM_8(k)*MU(i  ,j,k)+Ver_wmM_8(k)*MU(i  ,j,km)
         barzp = Ver_wpM_8(k)*MU(i+1,j,k)+Ver_wmM_8(k)*MU(i+1,j,km)
         mu_interp = (barz+barzp)*half

!        Pressure gradient and mu terms: RT barXZ * dBsPq/dX + mu barXZ * dfi'/dX
!                                        - RTstr barZ * dBsPrq/dX         
!        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         barz  = Ver_wpM_8(k)*(F_t(i  ,j,k)-Ver_Tstar_8%t(k))+Ver_wmM_8(k)*(F_t(i  ,j,km)-Ver_Tstar_8%t(km))
         barzp = Ver_wpM_8(k)*(F_t(i+1,j,k)-Ver_Tstar_8%t(k))+Ver_wmM_8(k)*(F_t(i+1,j,km)-Ver_Tstar_8%t(km))
         t_interp = (barz+barzp)*half 

         barz  = Ver_wpM_8(k)*Ver_Tstar_8%t(k)+Ver_wmM_8(k)*Ver_Tstar_8%t(km)

         w1 = ( BsPq(i+1,j,k)  - BsPq(i,j,k)  ) * Geomg_invDXMu_8(j)
         w2 = (   FI(i+1,j,k)  -   FI(i,j,k)  ) * Geomg_invDXMu_8(j)
         w3 = (  frq(i+1,j,k)  -  frq(i,j,k)  ) * Geomg_invDXMu_8(j)

         F_nu(i,j,k) = Dcst_rgasd_8 * t_interp * w1 + mu_interp * w2 &
                     + (one-Cstv_rE_8)*Dcst_rgasd_8 * barz * w3
         

!        Coriolis term & metric terms: - (f + tan(phi)/a * U ) * V barXY
!        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         v_interp = wk1(i,j)

         F_nu(i,j,k) = F_nu(i,j,k) - ( Cori_fcoru_8(i,j) + geomg_tyoa_8(j) * F_u(i,j,k) ) * v_interp

      end do
      end do

!     Compute Nv
!     ~~~~~~~~~~

!     U barX stored in wk2
!     ~~~~~~~~~~~~~~~~~~~~
      if(Schm_cub_Coriolis_L) then
         do j = j0v-1, jnv+2
         do i = i0, in
            wk2(i,j) = inuvl_wxux3_8(i,1)*F_u(i-2,j,k) &
                     + inuvl_wxux3_8(i,2)*F_u(i-1,j,k) &
                     + inuvl_wxux3_8(i,3)*F_u(i  ,j,k) &
                     + inuvl_wxux3_8(i,4)*F_u(i+1,j,k)
         end do
         end do
         do j = j0v, jnv
         do i = i0, in
            wk1(i,j) = inuvl_wyyv3_8(j,1) * wk2(i,j-1) &
                     + inuvl_wyyv3_8(j,2) * wk2(i,j  ) &
                     + inuvl_wyyv3_8(j,3) * wk2(i,j+1) &
                     + inuvl_wyyv3_8(j,4) * wk2(i,j+2)
         end do
         end do
      else
         do j = j0v, jnv
         do i = i0, in
            wk1(i,j) = 0.25d0*(F_u(i,j,k)+F_u(i-1,j,k)+F_u(i,j+1,k)+F_u(i-1,j+1,k))
         end do
         end do
      endif


      do j = j0v, jnv
      do i = i0, in

!        mu barYZ
!        ~~~~~~~~
         barz  = Ver_wpM_8(k)*MU(i,j  ,k)+Ver_wmM_8(k)*MU(i,j  ,km)
         barzp = Ver_wpM_8(k)*MU(i,j+1,k)+Ver_wmM_8(k)*MU(i,j+1,km)
         mu_interp = (barz+barzp)*half

!        Pressure gradient and Mu term: RT' barYZ * dBsPq/dY + mu barYZ * dfi'/dY
!                                     - RTstr barZ * dBsPrq/dY        
!        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         barz  = Ver_wpM_8(k)*(F_t(i,j  ,k)-Ver_Tstar_8%t(k))+Ver_wmM_8(k)*(F_t(i,j  ,km)-Ver_Tstar_8%t(km))
         barzp = Ver_wpM_8(k)*(F_t(i,j+1,k)-Ver_Tstar_8%t(k))+Ver_wmM_8(k)*(F_t(i,j+1,km)-Ver_Tstar_8%t(km))
         t_interp = (barz+barzp)*half

         barz  = Ver_wpM_8(k)*Ver_Tstar_8%t(k)+Ver_wmM_8(k)*Ver_Tstar_8%t(km)

         w1 = (  BsPq(i,j+1,k) -  BsPq(i,j,k) ) * Geomg_invDYMv_8(j)
         w2 = (    FI(i,j+1,k) -    FI(i,j,k) ) * Geomg_invDYMv_8(j)
         w3 = (   frq(i,j+1,k) -   frq(i,j,k) ) * Geomg_invDYMv_8(j)

         F_nv(i,j,k) = Dcst_rgasd_8 * t_interp * w1 + mu_interp * w2 &
                     + (one-Cstv_rE_8)*Dcst_rgasd_8 * barz * w3

!        Coriolis term & metric terms: + f * U barXY + tan(phi)/a * (U barXY)^2
!        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         u_interp = wk1(i,j)

         F_nv(i,j,k) = F_nv(i,j,k) + ( Cori_fcorv_8(i,j) + geomg_tyoav_8(j) * u_interp ) * u_interp

      end do
      end do

      end do
!$omp enddo

!     Set  Nu=0  on the east and west boundaries of the LAM grid
!     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      if (.not.Grd_yinyang_L) then

         if (G_lam) then
            if (l_west) then
!$omp do
               do k=1,l_nk
               do j=j0,jn
                  F_nu(pil_w,j,k) = 0.
               end do
               enddo
!$omp enddo
            endif
            if (l_east) then
!$omp do
               do k=1,l_nk
               do j=j0,jn
                  F_nu(l_ni-pil_e,j,k) = 0.
               end do
               enddo
!$omp enddo
            endif
         endif

!     Set  Nv=0  on the north and south boundaries  of the LAM grid
!     and        at the north and south poles       of the GLOBAL grid
!     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

         if (l_south) then
!$omp do
            do k=1,l_nk
            do i=i0,in
               F_nv(i,pil_s,k) = 0.
            end do
            enddo
!$omp enddo
         endif
         if (l_north) then
!$omp do
            do k=1,l_nk
            do i=i0,in
               F_nv(i,l_nj-pil_n,k) = 0.
            end do
            enddo
!$omp enddo
         endif

      endif

!$omp do
      do k=k0t,l_nk
         km=max(k-1,1)
         kp=min(k+1,l_nk)
         kq=max(2,k)
         kmq=max(2,k-1)
         kmq=max(kmq,k0t)

!**********************************
!   The nonlinear deviation of    *
! the thermodynamic equation: Nt' *
!**********************************

!        Compute Nw and Nt' (Nf=0)
!        ~~~~~~~~~~~~~~~~~~~~~~~~~
         if(Schm_nologT_L) then
            if(Schm_opentop_L.and.k.eq.k0t) then
               do j= j0, jn
               do i= i0, in
                  w3=Dcst_cappa_8 * ( one - delta_8 * F_hu(i,j,k) )
                  F_nb(i,j)= - (w3*F_t(i,j,k)/Ver_Tstar_8%t(k)-Dcst_cappa_8) * (F_xd(i,j,k)+F_qd(i,j,k))
               end do
               end do
            endif
            w1 = Cstv_invT_8 / Ver_Tstar_8%t(k)
            w2 = Cstv_invT_8 / Ver_Tstar_8%t(k) * Ver_idz_8%t(k) / Dcst_Rgasd_8
            do j= j0, jn
            do i= i0, in
               w3=Dcst_cappa_8 * ( one - delta_8 * F_hu(i,j,k) )
               w4=Ver_wpstar_8(k)*(F_xd(i,j,k)+F_qd(i,j,k))+Ver_wmstar_8(k)*(F_xd(i,j,km)+F_qd(i,j,km))
               w5=Ver_wpstar_8(k)*F_qd(i,j,k) + Ver_wmstar_8(k)*F_qd(i,j,km)

               qbar=Ver_wpstar_8(k)*F_q(i,j,k+1)+Ver_wmstar_8(k)*half*(F_q(i,j,kq)+F_q(i,j,kmq))
               qbar=Ver_wp_8%t(k)*qbar+Ver_wm_8%t(k)*F_q(i,j,kq)*Ver_onezero(k)
               MUlin=Ver_idz_8%t(k)*(F_q(i,j,k+1)-F_q(i,j,kq)*Ver_onezero(k)) + qbar

               F_nt(i,j,k) = - Cstv_rE_8*Cstv_invT_8*MUlin &
                             + w1*(F_t(i,j,k)-Ver_Tstar_8%t(k) ) &
                             + w2*( FI(i,j,k+1)+Dcst_Rgasd_8*Ver_Tstar_8%m(k+1)*BsPrq(i,j,k+1) &
                                  - FI(i,j,k  )-Dcst_Rgasd_8*Ver_Tstar_8%m(k  )*BsPrq(i,j,k  ) ) &
                             - (w3*F_t(i,j,k)/Ver_Tstar_8%t(k)-Dcst_cappa_8)*w4 &
                             + (Cstv_rE_8 - one)*Dcst_cappa_8*w5
                          
               F_nw(i,j,k) = - Dcst_grav_8 * ( MU(i,j,k) - MUlin )
               F_nf(i,j,k) = 0.0
               F_nx(i,j,k) = 0.
            end do
            end do
         else
            w1 = one / Ver_Tstar_8%t(k)
            do j= j0, jn
            do i= i0, in
               xtmp_8(i,j) = F_t(i,j,k) * w1
            end do
            end do
            call vlog ( ytmp_8, xtmp_8, nij )
            if(Schm_opentop_L.and.k.eq.k0t) then
               do j= j0, jn
               do i= i0, in
                  F_nb(i,j) = Cstv_invT_8*(ytmp_8(i,j)-xtmp_8(i,j)+one)
               end do
               end do
            endif
            w1 = Ver_idz_8%t(k) / Dcst_Rgasd_8 / Ver_Tstar_8%t(k)
            w2 = one / Ver_Tstar_8%t(k) * Ver_idz_8%t(k) / Dcst_Rgasd_8
            do j= j0, jn
            do i= i0, in
               w3=Dcst_cappa_8 * ( one - delta_8 * F_hu(i,j,k) )
               w4=Ver_wpstar_8(k)*F_zd(i,j,k)+Ver_wmstar_8(k)*F_zd(i,j,km)
               qbar=Ver_wpstar_8(k)*F_q(i,j,k+1)+Ver_wmstar_8(k)*half*(F_q(i,j,kq)+F_q(i,j,kmq))
               qbar=Ver_wp_8%t(k)*qbar+Ver_wm_8%t(k)*F_q(i,j,kq)*Ver_onezero(k)
               w5=Ver_wpstar_8(k)*BsPrq(i,j,k+1)+Ver_wmstar_8(k)*half*(BsPrq(i,j,kq)+BsPrq(i,j,kmq))
               w5=Ver_wp_8%t(k)*w5+Ver_wm_8%t(k)*BsPrq(i,j,kq)*Ver_onezero(k)               
               MUlin=Ver_idz_8%t(k)*(F_q(i,j,k+1)-F_q(i,j,kq)*Ver_onezero(k)) + qbar
               F_nw(i,j,k) = - Dcst_grav_8 * ( MU(i,j,k) - MUlin )
               F_nt(i,j,k) = Cstv_invT_8*(ytmp_8(i,j) - w3*(one - Cstv_rE_8)*qbar &
                                    + w2*( FI(i,j,k+1)+Dcst_Rgasd_8*Ver_Tstar_8%m(k+1)*BsPrq(i,j,k+1) &
                                     - FI(i,j,k  )-Dcst_Rgasd_8*Ver_Tstar_8%m(k  )*BsPrq(i,j,k  ) ) &
                                         -Cstv_rE_8*MUlin + (Dcst_cappa_8-w3)*(w5+w4*Cstv_tau_8))
               F_nf(i,j,k) = 0.0
            end do
            end do
         endif

         if(Cstv_Tstr_8.lt.0.) then        
            dlnTstr_8=(Ver_Tstar_8%m(k+1)-Ver_Tstar_8%m(k))*Ver_idz_8%t(k)/Ver_Tstar_8%t(k) 
            do j= j0, jn
            do i= i0, in
               w2=Ver_wpstar_8(k)*BsPrq(i,j,k+1)+Ver_wmstar_8(k)*half*(BsPrq(i,j,kq)+BsPrq(i,j,kmq))
               w2=Ver_wp_8%t(k)*w2+Ver_wm_8%t(k)*BsPrq(i,j,kq)*Ver_onezero(k)
               w3=Ver_wpstar_8(k)*Ver_Tstar_8%m(k+1)*BsPrq(i,j,k+1)+Ver_wmstar_8(k)*half* &
                  (Ver_Tstar_8%m(k)*BsPrq(i,j,kq)+Ver_Tstar_8%m(km)*BsPrq(i,j,kmq))
               w3=Ver_wp_8%t(k)*w3+Ver_wm_8%t(k)*Ver_Tstar_8%m(k)*BsPrq(i,j,kq)*Ver_onezero(k)
               w1=Ver_wpstar_8(k)*F_zd(i,j,k)+Ver_wmstar_8(k)*F_zd(i,j,km)
               F_nt(i,j,k) = F_nt(i,j,k) + w1*dlnTstr_8
               F_nf(i,j,k) = Dcst_Rgasd_8 * Cstv_invT_8 * ( Ver_Tstar_8%t(k)*w2 - w3 )
            end do
            end do
         endif

!        Compute Nt" and Nf"
!        ~~~~~~~~~~~~~~~~~~~
         w1 = Dcst_cappa_8/ ( Dcst_Rgasd_8 * Ver_Tstar_8%t(k) )
         w2 = Cstv_invT_m_8 / ( Dcst_cappa_8 + Ver_epsi_8(k) )
         do j= j0, jn
         do i= i0, in
            w_nt = F_nt(i,j,k) + Ver_igt_8 * F_nw(i,j,k)
            F_nt(i,j,k) = w2 * ( w_nt + Ver_igt2_8 * F_nf(i,j,k) )
            F_nf(i,j,k) = w2 * ( w_nt - w1 * F_nf(i,j,k) )
         end do
         end do

      end do
!$omp enddo

!***************************************
!     The nonlinear deviation of       *
!   the continuity equation: Nc and    *
! the horizontal Divergence of (Nu,Nv) *
!   combined with Nc (stored in Nc)    *
!***************************************

!$omp do
      do k=k0,l_nk

!        Compute Nc
!        ~~~~~~~~~~
         km=max(k-1,1)
         kp=min(k+1,l_nk)
         if(Schm_nologinC_L) then
            do j = j0, jn
            do i = i0, in
               ndiv = (F_u(i,j,k)-F_u(i-1,j,k)) * geomg_invDXM_8(j) &
                    + (F_v(i,j,k)*geomg_cyM_8(j)-F_v(i,j-1,k)*geomg_cyM_8(j-1))*Geomg_invDYM_8(j) &
                    + (F_zd(i,j,k)-Ver_onezero(k)*F_zd(i,j,km))*Ver_idz_8%m(k)
               xdot = Ver_wp_8%m(k) * F_xd(i,j,k) + Ver_wm_8%m(k) * Ver_onezero(k) * F_xd(i,j,km)
               F_nc(i,j,k) = Cstv_invT_8  * Cstv_bar1_8 * (Ver_b_8%m(k)-Ver_bzz_8(k))*F_s(i,j) &
                           + Ver_dbdz_8%m(k)*F_s(i,j)*(ndiv+xdot)
            end do
            end do
         else
            do j = j0, jn
            do i = i0, in
               xtmp_8(i,j) = one + Ver_dbdz_8%m(k) * F_s(i,j)
            end do
            end do
            call vlog(ytmp_8, xtmp_8, nij)
            do j = j0, jn
            do i = i0, in
               F_nc(i,j,k) = Cstv_invT_8 * ( ytmp_8(i,j) +  &
                             ( Cstv_bar1_8 * (Ver_b_8%m(k)-Ver_bzz_8(k)) &
                             - Ver_dbdz_8%m(k) )*F_s(i,j) ) &
                                      + (Ver_wpM_8(k)-Ver_wp_8%m(k)) * F_zd(i,j,k) &
                                      + (Ver_wmM_8(k)-Ver_wm_8%m(k)) * Ver_onezero(k) * F_zd(i,j,km)
            end do
            end do
         endif

      end do
!$omp enddo

!$omp do
      do k=k0,l_nk

!        Compute Nc"
!        ~~~~~~~~~~~
         km=max(k-1,1)
         w1=Ver_igt_8*Ver_wpA_8(k)
         w2=Ver_igt_8*Ver_wmA_8(k)*Ver_onezero(k)
         w3= Ver_idz_8%m(k) + Ver_wp_8%m(k)
         w4=(Ver_idz_8%m(k) - Ver_wm_8%m(k))*Ver_onezero(k)
         do j = j0, jn
         do i = i0, in
            ndiv = (F_nu(i,j,k)-F_nu(i-1,j,k)) * geomg_invDXM_8(j) &
               + (F_nv(i,j,k)*geomg_cyM_8(j)-F_nv(i,j-1,k)*geomg_cyM_8(j-1))*Geomg_invDYM_8(j)
            F_nc(i,j,k) = ndiv  - Cstv_invT_m_8 * ( F_nc(i,j,k) - w1*F_nw(i,j,k) - w2*F_nw(i,j,km) )
         end do
         end do

      end do
!$omp enddo

!**********************************************************
! The full contributions to the RHS of Helmholtz equation *
!**********************************************************

!     Finish computations of NP (combining Nc", Nt", Nf")
!     Substract NP from RP(Rc") and store result(RP-NP) in RP

!$omp do
      do j= j0, jn
      do i= i0, in
         F_nt(i,j,l_nk) = (F_nt(i,j,l_nk) - Ver_wmstar_8(l_nk)*F_nt(i,j,l_nk-1)) &
                          /Ver_wpstar_8(l_nk)
      end do
      end do
!$omp enddo

!$omp do
      do k=k0,l_nk
         km=max(k-1,1)
         w1=(Ver_idz_8%m(k) + Ver_wp_8%m(k))
         w2=(Ver_idz_8%m(k) - Ver_wm_8%m(k))*Ver_onezero(k)
         w3=Ver_wpA_8(k)*Ver_epsi_8(k)
         w4=Ver_wmA_8(k)*Ver_epsi_8(km)*Ver_onezero(k)
         do j= j0, jn
         do i= i0, in
            F_rhs(i,j,k) =  c1 * ( F_rc(i,j,k) - F_nc(i,j,k) &
                                  + w1 * F_nt(i,j,k) - w2 * F_nt(i,j,km)  &
                                  + w3 * F_nf(i,j,k) + w4 * F_nf(i,j,km)  )
         enddo
         enddo
      enddo
!$omp enddo

!     Apply boundary conditions
!     ~~~~~~~~~~~~~~~~~~~~~~~~~

      if(Schm_opentop_L) then
         F_rhs(:,:,1:k0t) = 0.0
!$omp do
         do j= j0, jn
         do i= i0, in
            F_nb(i,j)    = F_nt(i,j,k0t)-Ver_ikt_8*F_nb(i,j)
            F_rhs(i,j,k0)= F_rhs(i,j,k0) + c1 * Ver_cstp_8 * F_nb(i,j)
         end do
         end do
!$omp enddo
      endif

      if(Schm_nologT_L) then
!$omp do
         do j= j0, jn
         do i= i0, in
             F_nt(i,j,l_nk) =  F_nt(i,j,l_nk) - Cstv_invT_8 * F_nx(i,j,l_nk)
         end do
         end do
!$omp enddo
      endif
!$omp do
      do j= j0, jn
      do i= i0, in
          F_nt(i,j,l_nk) =  Ver_wpstar_8(l_nk) * F_nt(i,j,l_nk)
          F_rhs(i,j,l_nk) = F_rhs(i,j,l_nk) - c1 * Ver_cssp_8 * F_nt(i,j,l_nk)
      end do
      end do
!$omp enddo

!$omp end parallel

      deallocate ( BsPq, BsPrq, frq, FI)
      if (.not.Schm_hydro_L) deallocate ( MU )

1000 format(/,5X,'COMPUTE NON-LINEAR RHS: (S/R NLI)')
!     __________________________________________________________________
!
      return
      end

