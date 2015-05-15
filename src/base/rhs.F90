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
!*s/r rhs - compute the right-hand sides: Ru, Rv, Rc, Rt, Rw, Rf,
!            save the results for next iteration in the o's
!
      subroutine rhs ( F_oru, F_orv, F_orc, F_ort, F_orw, F_orf,F_orx,F_orq, &
                       F_ruw1,F_rvw1,F_u,F_v,F_w,F_t,F_s,F_zd,F_q,F_xd,F_qd, &
                              F_fis, Minx,Maxx,Miny,Maxy, Nk )
      implicit none
#include <arch_specific.hf>

      integer Minx,Maxx,Miny,Maxy, Nk
      real F_oru   (Minx:Maxx,Miny:Maxy,  Nk)  ,F_orv   (Minx:Maxx,Miny:Maxy,  Nk), &
           F_orc   (Minx:Maxx,Miny:Maxy,  Nk)  ,F_ort   (Minx:Maxx,Miny:Maxy,  Nk), &
           F_orw   (Minx:Maxx,Miny:Maxy,  Nk)  ,F_orf   (Minx:Maxx,Miny:Maxy,  Nk), &
           F_orx   (Minx:Maxx,Miny:Maxy,  Nk)  ,F_orq   (Minx:Maxx,Miny:Maxy,  Nk), &
           F_ruw1  (Minx:Maxx,Miny:Maxy,  Nk)  ,F_rvw1  (Minx:Maxx,Miny:Maxy,  Nk), &
           F_u     (Minx:Maxx,Miny:Maxy,  Nk)  ,F_v     (Minx:Maxx,Miny:Maxy,  Nk), &
           F_w     (Minx:Maxx,Miny:Maxy,  Nk)  ,F_t     (Minx:Maxx,Miny:Maxy,  Nk), &
           F_xd    (Minx:Maxx,Miny:Maxy,  Nk)  ,F_qd    (Minx:Maxx,Miny:Maxy,  Nk), &
           F_s     (Minx:Maxx,Miny:Maxy)       ,F_zd    (Minx:Maxx,Miny:Maxy,  Nk), &
           F_q     (Minx:Maxx,Miny:Maxy,2:Nk+1),F_fis   (Minx:Maxx,Miny:Maxy)

!author
!     Alain Patoine
!
!revision
! v2_00 - Desgagne M.       - initial MPI version (from rhs v1_03)
! v2_21 - Lee V.            - modifications for LAM version
! v2_30 - Edouard  S.       - adapt for vertical hybrid coordinate
!                             (Change to Rcn)
! v2_31 - Desgagne M.       - remove treatment of hut1 and qct1
! v3_00 - Qaddouri & Lee    - For LAM, Change Ru, Rv values on the boundaries
! v3_00                       of the LAM grid with values from Nesting data
! v3_02 - Edouard S.        - correct bug in Ru and Rv in the non hydrostatic version
! v3_10 - Corbeil & Desgagne & Lee - AIXport+Opti+OpenMP
! v4_00 - Plante & Girard   - Log-hydro-pressure coord on Charney-Phillips grid
! v4_40 - Qaddouri/Lee      - Exchange and interp winds between Yin, Yang
! v4.70 - Gaudreault S.     - Reformulation in terms of real winds (removing wind images)
!                           - Explicit integration of metric terms (optional)

#include "glb_ld.cdk"
#include "cori.cdk"
#include "cstv.cdk"
#include "dcst.cdk"
#include "geomg.cdk"
#include "schm.cdk"
#include "intuv.cdk"
#include "inuvl.cdk"
#include "type.cdk"
#include "ver.cdk"
#include "lun.cdk"
#include "grd.cdk"

      logical, save :: done=.false.
      integer :: i0,  in,  j0,  jn
      integer :: i0u, inu, j0v, jnv
      integer :: i, j, k, km, kq, kp, nij, jext
      real*8  tdiv, BsPqbarz, fipbarz, dlnTstr_8, barz, barzp, &
              u_interp, v_interp, t_interp, mu_interp, zdot, xdot, &
              xtmp_8(l_ni,l_nj), ytmp_8(l_ni,l_nj)
      real, dimension(:,:,:), pointer :: BsPq, FI, MU
      save MU
      real*8, parameter :: one=1.d0, zero=0.d0, half=0.5d0 , &
                           alpha1= -1.d0/16.d0 , alpha2 =9.d0/16.d0
!
!     ---------------------------------------------------------------
!      
      if (Lun_debug_L) write (Lun_out,1000)

      nullify  ( BsPq, FI )
      allocate ( BsPq(Minx:Maxx,Miny:Maxy,l_nk+1), FI(Minx:Maxx,Miny:Maxy,l_nk+1) )

      if (.not.done) then
         nullify  ( MU )
         allocate ( MU(Minx:Maxx,Miny:Maxy,l_nk) )
      endif

!     Exchanging halos for derivatives & interpolation
!     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      call rpn_comm_xch_halo( F_u , l_minx,l_maxx,l_miny,l_maxy,l_niu,l_nj ,G_nk, &
                              G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      call rpn_comm_xch_halo( F_v , l_minx,l_maxx,l_miny,l_maxy,l_ni ,l_njv,G_nk, &
                              G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      call rpn_comm_xch_halo( F_t , l_minx,l_maxx,l_miny,l_maxy,l_ni ,l_nj ,G_nk, &
                              G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      call rpn_comm_xch_halo( F_s , l_minx,l_maxx,l_miny,l_maxy,l_ni ,l_nj ,1   , &
                              G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      if (.not.Schm_hydro_L) then
         call rpn_comm_xch_halo(  F_q, l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj,G_nk,&
                                  G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      endif

      nij = l_ni*l_nj

!     Indices to compute Rc, Rt, Rf, Rw
      i0 = 1
      in = l_ni
      j0 = 1
      jn = l_nj
      if (G_lam) then
         jext = Grd_extension - Grd_maxcfl - 1
         if (l_west)  i0 = 1   +jext
         if (l_east)  in = l_ni-jext
         if (l_south) j0 = 1   +jext
         if (l_north) jn = l_nj-jext
      endif

!     Additional indices to compute Ru, Rv
      i0u = 1
      inu = l_niu
      j0v = 1
      jnv = l_njv
      if (G_lam) then
         jext = 0
         if(Grd_yinyang_L) jext = 1
         if (l_west ) i0u = i0 - jext
         if (l_east ) inu = in + jext-1
         if (l_south) j0v = j0 - jext
         if (l_north) jnv = jn + jext-1
     !!  if (l_west) print*,'in rhs, i0u,i0=',i0u,i0
     !!  if (l_east) print*,'in rhs, in,inu=',in,inu
      endif

      call diag_fi (FI, F_s, F_t, F_q, F_fis, l_minx,l_maxx,l_miny,l_maxy, l_nk, &
                    i0u,inu+1,j0v,jnv+1)

      if (Schm_hydro_L) then
         if (.not.done) then
            MU= 0.
            done= .true.
         endif
      else
         call diag_mu(MU, F_q, F_s, l_minx,l_maxx,l_miny,l_maxy, l_nk, &
                    i0u,inu+1,j0v,jnv+1)
      endif

!$omp parallel private(km,kq,kp,barz,barzp, &
!$omp     u_interp,v_interp,t_interp,mu_interp,zdot,xdot, &
!$omp     dlnTstr_8,BsPqbarz,fipbarz,tdiv,xtmp_8,ytmp_8)

!$omp do
      do k=1,l_nk+1
         kq=max(2,k)
         do j=j0v,jnv+1
            do i=i0u,inu+1
               BsPq(i,j,k) = Ver_b_8%m(k) * F_s(i,j) + F_q(i,j,kq) * Ver_onezero(k)
            enddo
         enddo
      enddo
!$omp enddo

!$omp do
      do k = 1,l_nk
      km=max(k-1,1)
      kp=min(k+1,l_nk)
      kq=max(k,2)

!********************************
! Compute Ru: RHS of U equation *
!********************************

      do j= j0,  jn
      do i= i0u, inu

         barz  = Ver_wp_8%m(k)*MU(i  ,j,k)+Ver_wm_8%m(k)*MU(i  ,j,km)
         barzp = Ver_wp_8%m(k)*MU(i+1,j,k)+Ver_wm_8%m(k)*MU(i+1,j,km)
         mu_interp = ( barz + barzp)*half

         barz  = Ver_wp_8%m(k)*F_t(i  ,j,k)+Ver_wm_8%m(k)*F_t(i  ,j,km)
         barzp = Ver_wp_8%m(k)*F_t(i+1,j,k)+Ver_wm_8%m(k)*F_t(i+1,j,km)
         t_interp = (barz + barzp)*half

         v_interp = 0.25d0*(F_v(i,j,k)+F_v(i,j-1,k)+F_v(i+1,j,k)+F_v(i+1,j-1,k))

         F_oru(i,j,k) = Cstv_invT_8  * F_u(i,j,k) - Cstv_Beta_8 * ( &
                        Dcst_rgasd_8 * t_interp * ( BsPq(i+1,j,k) - BsPq(i,j,k) ) * geomg_invDXMu_8(j)  &
                            + ( one + mu_interp)* (   FI(i+1,j,k) -   FI(i,j,k) ) * geomg_invDXMu_8(j)  &
                                     - ( Cori_fcoru_8(i,j) + geomg_tyoa_8(j) * F_u(i,j,k) ) * v_interp )
      end do
      end do

!********************************
! Compute Rv: RHS of V equation *
!********************************

      do j = j0v, jnv
      do i = i0,  in

         barz  = Ver_wp_8%m(k)*MU(i,j  ,k)+Ver_wm_8%m(k)*MU(i,j  ,km)
         barzp = Ver_wp_8%m(k)*MU(i,j+1,k)+Ver_wm_8%m(k)*MU(i,j+1,km)
         mu_interp = ( barz + barzp )*half

         barz  = Ver_wp_8%m(k)*F_t(i,j  ,k)+Ver_wm_8%m(k)*F_t(i,j  ,km)
         barzp = Ver_wp_8%m(k)*F_t(i,j+1,k)+Ver_wm_8%m(k)*F_t(i,j+1,km)
         t_interp = ( barz + barzp)*half

         u_interp = 0.25d0*(F_u(i,j,k)+F_u(i-1,j,k)+F_u(i,j+1,k)+F_u(i-1,j+1,k))

         F_orv(i,j,k) = Cstv_invT_8  * F_v(i,j,k) - Cstv_Beta_8 * ( &
                        Dcst_rgasd_8 * t_interp * ( BsPq(i,j+1,k) - BsPq(i,j,k) ) * geomg_invDYMv_8(j) &
                            + ( one + mu_interp)* (   FI(i,j+1,k) -   FI(i,j,k) ) * geomg_invDYMv_8(j) &
                                     + ( Cori_fcorv_8(i,j) + geomg_tyoav_8(j) * u_interp ) * u_interp )
      end do
      end do


!********************************************
! Compute Rt: RHS of thermodynamic equation *
! Compute Rf: RHS of FI equation            *
! Compute Rw: RHS of  w equation            *
!********************************************


      if (Schm_nolog_L) then
         do j= j0, jn
         do i= i0, in
            BsPqbarz = Ver_wp_8%t(k)*BsPq(i,j,k+1)+Ver_wm_8%t(k)*BsPq(i,j,k)
            F_ort(i,j,k) = Cstv_invT_8 * ( F_t(i,j,k)/Ver_Tstr_8(k)-one - Cstv_bar1_8 * Dcst_cappa_8 * BsPqbarz ) &
                         + Cstv_Beta_8 * Dcst_cappa_8 * (F_zd(i,j,k)+(F_t(i,j,k)/Ver_Tstr_8(k)-one)*(F_xd(i,j,k)+F_qd(i,j,k)))
            F_orx(i,j,k) = Cstv_invT_8 * Ver_b_8%t(k)*F_s(i,j) &
                         + Cstv_Beta_8 * (F_xd(i,j,k)-F_zd(i,j,k))
            F_orq(i,j,k) = Cstv_invT_8 * (Ver_wp_8%t(k)*F_q(i,j,k+1)+Ver_wm_8%t(k)*F_q(i,j,kq)*Ver_onezero(k)) &
                         + Cstv_Beta_8 * F_qd(i,j,k)
         end do
         end do
      else
         if (G_lam) then
            xtmp_8(:,:) = one
         endif
         do j = j0, jn
         do i = i0, in
            xtmp_8(i,j) = F_t(i,j,k) / Ver_Tstr_8(k)
         end do
         end do
         call vlog( ytmp_8, xtmp_8, nij )
         do j= j0, jn
         do i= i0, in
            BsPqbarz = Ver_wp_8%t(k)*BsPq(i,j,k+1)+Ver_wm_8%t(k)*BsPq(i,j,k)
            F_ort(i,j,k) = Cstv_invT_8 * ( ytmp_8(i,j) - Cstv_bar1_8 * Dcst_cappa_8 * BsPqbarz ) &
                         + Cstv_Beta_8 * Dcst_cappa_8 * F_zd(i,j,k)
         end do
         end do
      endif

      do j= j0, jn
      do i= i0, in
         fipbarz  = Ver_wp_8%t(k)* (FI(i,j,k+1)-Ver_FIstr_8(k+1))+Ver_wm_8%t(k)* (FI(i,j,k)-Ver_FIstr_8(k))
         F_orw(i,j,k) = Cstv_invT_8 * F_w(i,j,k) &
                      + Cstv_Beta_8 * Dcst_grav_8 * MU(i,j,k)
         F_orf(i,j,k) = Cstv_invT_8 * fipbarz  &
                      + Cstv_Beta_8 * Dcst_Rgasd_8 * Ver_Tstr_8(k) * F_zd(i,j,k) &
                      + Cstv_Beta_8 * Dcst_grav_8 * F_w(i,j,k)
      end do
      end do

      if(Cstv_Tstr_8.lt.0.) then
         barz  = Ver_wp_8%m(k )*Ver_Tstr_8(k )+Ver_wm_8%m(k )*Ver_Tstr_8(km)
         barzp = Ver_wp_8%m(kp)*Ver_Tstr_8(kp)+Ver_wm_8%m(kp)*Ver_Tstr_8(k )
         dlnTstr_8=log(barzp/barz)*Ver_idz_8%t(k)
         do j = j0, jn
         do i = i0, in
            F_ort(i,j,k) = F_ort(i,j,k) - Cstv_Beta_8 * F_zd(i,j,k) * dlnTstr_8
         end do
         end do
      endif

!*****************************************
! Compute Rc: RHS of Continuity equation *
!*****************************************

      if(Schm_nolog_L) then
         do j = j0, jn
         do i = i0, in
            tdiv = (F_u (i,j,k)-F_u (i-1,j,k))*geomg_invDXM_8(j) &
                 + (F_v (i,j,k)*geomg_cyM_8(j)-F_v (i,j-1,k)*geomg_cyM_8(j-1))*geomg_invDYM_8(j) &
                 + (F_zd(i,j,k)-Ver_onezero(k)*F_zd(i,j,km))*Ver_idz_8%m(k)
            zdot = Ver_wp_8%m(k) * F_zd(i,j,k) + Ver_wm_8%m(k) * Ver_onezero(k) * F_zd(i,j,km)
            xdot = Ver_wp_8%m(k) * F_xd(i,j,k) + Ver_wm_8%m(k) * Ver_onezero(k) * F_xd(i,j,km)
            F_orc(i,j,k) = Cstv_invT_8 * ( Cstv_bar1_8*Ver_b_8%m(k) + Ver_dbdz_8%m(k) ) * F_s(i,j) &
                         - Cstv_Beta_8 * ( tdiv + zdot + Ver_dbdz_8%m(k)*F_s(i,j)*(tdiv+xdot) )
         end do
         end do
      else
         do j = j0, jn
         do i = i0, in
            xtmp_8(i,j) = one + Ver_dbdz_8%m(k) * F_s(i,j)
         end do
         end do
         call vlog( ytmp_8, xtmp_8, nij)
         do j = j0, jn
         do i = i0, in
            tdiv = (F_u (i,j,k)-F_u (i-1,j,k))*geomg_invDXM_8(j) &
                 + (F_v (i,j,k)*geomg_cyM_8(j)-F_v (i,j-1,k)*geomg_cyM_8(j-1))*geomg_invDYM_8(j) &
              + (F_zd(i,j,k)-Ver_onezero(k)*F_zd(i,j,km))*Ver_idz_8%m(k) &
              + Ver_wp_8%m(k) * F_zd(i,j,k) &
              + Ver_wm_8%m(k) * Ver_onezero(k) * F_zd(i,j,km)
         F_orc (i,j,k) = Cstv_invT_8 * ( Cstv_bar1_8*Ver_b_8%m(k)*F_s(i,j) + ytmp_8(i,j) ) &
                       - Cstv_Beta_8 * tdiv
         end do
         end do
      endif

   end do
!$omp enddo

!$omp  end parallel

      if( .not.G_lam ) then
!******************************************************
! Interpolate Ru, Rv from U-, V-grid to G-grid, resp. *
!******************************************************

         call rpn_comm_xch_halo ( F_oru, l_minx,l_maxx,l_miny,l_maxy,l_niu,l_nj,G_nk, &
              G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
         call rpn_comm_xch_halo ( F_orv, l_minx,l_maxx,l_miny,l_maxy,l_ni,l_njv,G_nk, &
              G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )

!$omp parallel
!$omp do
         do k=1,l_nk
            do j = j0, jn
               do i = i0, in
                  F_ruw1(i,j,k) =  alpha1 * ( F_oru(i-2,j,k) +  F_oru(i+1,j,k) )&
                                 + alpha2 * ( F_oru(i-1,j,k) +  F_oru(i  ,j,k) )
                  F_rvw1(i,j,k) =  inuvl_wyvy3_8(j,1) * F_orv(i,j-2,k) &
                                 + inuvl_wyvy3_8(j,2) * F_orv(i,j-1,k) &
                                 + inuvl_wyvy3_8(j,3) * F_orv(i,j  ,k) &
                                 + inuvl_wyvy3_8(j,4) * F_orv(i,j+1,k)
               end do
            end do
         end do
!$omp enddo
!$omp  end parallel

      endif

      deallocate ( BsPq, FI )
      if (.not.Schm_hydro_L) deallocate ( MU )

1000  format(3X,'COMPUTE THE RIGHT-HAND-SIDES: (S/R RHS)')
!
!     ---------------------------------------------------------------
!      
      return
      end
