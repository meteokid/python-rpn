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

!**   s/r set_dync - initialize the dynamics model configuration

      subroutine set_dync
      use matvec_mod, only: matvec_init
      implicit none
#include <arch_specific.hf>

#include "gmm.hf"
#include "cstv.cdk"
#include "dcst.cdk"
#include "glb_ld.cdk"
#include "geomg.cdk"
#include "lam.cdk"
#include "lun.cdk"
#include "schm.cdk"
#include "sol.cdk"
#include "ver.cdk"
#include "vt1.cdk"

      integer k,err,istat,k0,i,j
      real tmean(G_nk)
      real*8  w1, w2, w3, w4
      real*8, parameter :: zero=0.d0, one=1.d0, half=.5d0
!
!     ---------------------------------------------------------------

      if (lun_out.gt.0) then
          write(Lun_out,*)'SETTING up OPR,ADW,...(S/R SET_DYNC)'
          write(Lun_out,*)'===================================='
      endif

      k0=1+Lam_gbpil_T

      Cstv_tau_8   = Cstv_dt_8 * Cstv_bA_8
      Cstv_invT_8  = one/Cstv_tau_8
      Cstv_Beta_8  = (one-Cstv_bA_8)/Cstv_bA_8

!     Parameters for the nonhydrostatic case
      Cstv_tau_nh_8   = Cstv_dt_8 * Cstv_bA_nh_8
      Cstv_invT_nh_8  = one/Cstv_tau_nh_8
      Cstv_Beta_nh_8  = (one-Cstv_bA_nh_8)/Cstv_bA_nh_8
      Cstv_rEp_8=Cstv_rE_8*Cstv_tau_8/Cstv_tau_nh_8

      if (Schm_hydro_L .or. & 
         (abs(one-Cstv_rE_8).lt.1e-5 .and. abs(Cstv_bA_8-Cstv_bA_nh_8).lt.1e-5)) then
         Cstv_rEp_8=one
         Cstv_tau_nh_8=Cstv_tau_8
         Cstv_invT_nh_8=Cstv_invT_8
         Cstv_Beta_nh_8=Cstv_Beta_8
      endif


      if (Schm_advec.eq.1) then ! traditional advection
         Cstv_dtA_8 = Cstv_dt_8 * 0.5d0
      endif
      if (Schm_advec.eq.2) then ! consistant advection
         Cstv_dtA_8 = Cstv_tau_8
      endif
      Cstv_dtD_8 = Cstv_dt_8 - Cstv_dtA_8

      if (Schm_advec.eq.0) then ! no advection
         Cstv_dtA_8 = 0.d0
         Cstv_dtD_8 = 0.d0
      endif

      Ver_igt_8    = Cstv_invT_nh_8/Dcst_grav_8
      Ver_ikt_8    = Cstv_invT_8/Dcst_cappa_8
      if(Schm_hydro_L) Ver_igt_8=zero
      Ver_igt2_8   = Cstv_rE_8*Ver_igt_8**2 ! Modified epsilon
      Ver_igt_8    = Cstv_rE_8*Ver_igt_8 ! Modified epsilon

      if( Cstv_Tstr_8 .lt. 0. ) then
         ! TSTAR variable in the vertical
         err = gmm_get(gmmk_tt1_s,tt1)
         call estimate_tmean (tmean,tt1,l_minx,l_maxx,l_miny,l_maxy,G_nk)
         Ver_Tstar_8%t(1:G_nk) = tmean(1:G_nk)
      else
         do k=1,G_nk
            Ver_Tstar_8%t(k) = Cstv_Tstr_8
         enddo
      endif

      Ver_Tstar_8%m(1) = Ver_Tstar_8%t(1)
      do k=2,G_nk
         Ver_Tstar_8%m(k) = Ver_wp_8%m(k)*Ver_Tstar_8%t(k) + Ver_wm_8%m(k)*Ver_Tstar_8%t(k-1)
      enddo
      Ver_Tstar_8%m(G_nk+1) = Ver_Tstar_8%t(G_nk)

      Ver_fistr_8(G_nk+1)= 0.d0
      do k = G_nk, 1, -1
         Ver_fistr_8(k) = Ver_fistr_8(k+1)-Dcst_Rgasd_8*Ver_Tstar_8%t(k)*(Ver_z_8%m(k)-Ver_z_8%m(k+1))
      enddo

      do k=1,G_nk
         Ver_epsi_8(k)=Dcst_Rgasd_8*Ver_Tstar_8%t(k)*Ver_igt2_8
         Ver_gama_8(k)=Cstv_invT_8**2/ &
              (Dcst_Rgasd_8*Ver_Tstar_8%t(k)*(Dcst_cappa_8+Ver_epsi_8(k)))
      enddo

      Cstv_hco0_8 = Dcst_rayt_8**2
      Cstv_hco1_8 = zero
      Cstv_hco2_8 = one

      Ver_alfat_8 = one
      Ver_cst_8   = zero
      Ver_cstp_8  = zero

      if(Schm_opentop_L) then
         w1 = Ver_wm_8%m(k0)*(Ver_idz_8%t(k0-1) &
                 +(one-Dcst_cappa_8)*Ver_epsi_8(k0-1)*Ver_wm_8%t(k0-1))
         w2 = one/(Ver_idz_8%t(k0-1)+Ver_epsi_8(k0-1)*Ver_wm_8%t(k0-1))
         Ver_alfat_8 = (Ver_idz_8%t(k0-1) &
                         - Ver_epsi_8(k0-1)*Ver_wp_8%t(k0-1)) * w2
         Ver_cst_8   =                 one / Ver_gama_8(k0-1) * w2
         Ver_cstp_8  = (Ver_idz_8%t(k0-1)*Ver_idz_8%m(k0)-w1) * w2
      endif

      Ver_css_8   = one/Ver_gama_8(G_nk) &
                   /(Ver_idz_8%t(G_nk)+Dcst_cappa_8*Ver_wpstar_8(G_nk))
      w1 = Ver_wmstar_8(G_nk)*half*(Ver_gama_8(G_nk  )*Ver_epsi_8(G_nk) &
                                   -Ver_gama_8(G_nk-1)*Ver_epsi_8(G_nk-1))
      w2 = Ver_wmstar_8(G_nk)*Ver_gama_8(G_nk-1)*Ver_idz_8%t(G_nk-1)
      Ver_alfas_8 = Ver_css_8*Ver_gama_8(G_nk)*Ver_idz_8%t(G_nk) &
                  + Ver_css_8 * ( w1 + w2 )
      Ver_betas_8 = Ver_css_8 * ( w1 - w2 )
      w1=Ver_gama_8(G_nk)*Ver_idz_8%t(G_nk)*(Ver_idz_8%m(G_nk)+Ver_wp_8%m(G_nk))/Ver_wpstar_8(G_nk)
      w2=Ver_wp_8%m(G_nk)*Ver_gama_8(G_nk)*Ver_epsi_8(G_nk)*(one-Dcst_cappa_8)
      Ver_cssp_8  = Ver_css_8 * ( w1 - w2 )

      Cstv_bar0_8 = zero
      Cstv_bar1_8 = one
      if(Schm_autobar_L) then
         Cstv_bar0_8 = Cstv_invT_8**2/Ver_FIstr_8(1)
         Cstv_bar1_8 = zero
         Ver_alfas_8 = one
         Ver_css_8   = zero
         Ver_cssp_8  = zero
         Cstv_hco1_8 = Cstv_bar0_8
         Cstv_hco2_8 = zero
      endif

      call set_opr

      if ( G_lam ) then
         if ( Schm_adxlegacy_L ) then 
            call itf_adx_set
         else
            call adv_setgrid
            call adv_param 
         endif
      else
         call itf_adx_set
      endif

      call grid_area_mask (Geomg_area_8, Geomg_mask_8, l_ni,l_nj)

      if (Sol_type_S == 'ITERATIVE_3D') call matvec_init()
!
!     ---------------------------------------------------------------
!
      return
      end
