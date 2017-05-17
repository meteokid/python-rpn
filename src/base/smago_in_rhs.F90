!i---------------------------------- LICENCE BEGIN -------------------------------
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

!**s/r smago_in_rhs - Applies horizontal Smagorinsky-type nonlinear diffusion
!
      subroutine smago_in_rhs (F_du, F_dv, F_dw, F_dlth, F_u, F_v, F_w, F_t, F_s, &
                               lminx, lmaxx, lminy, lmaxy, nk)
      use gmm_smag
      use hzd_mod
      use grid_options
      use gem_options
      implicit none
#include <arch_specific.hf>

      integer, intent(in) :: lminx,lmaxx,lminy,lmaxy, nk
      real, dimension(lminx:lmaxx,lminy:lmaxy,nk), intent(inout) :: F_du, F_dv, F_dw, F_dlth
      real, dimension(lminx:lmaxx,lminy:lmaxy,nk), intent(in) :: F_u, F_v, F_w, F_t
      real, dimension(lminx:lmaxx,lminy:lmaxy),    intent(in) :: F_s
!
!author
!     Claude Girard
!
!revision
! v5_0 - Girard C.    - initial version based on hzd_smago by S. Gaudreault
! v5_0 - Husain S.    - added vertically variable background diffusion      

#include "gmm.hf"
#include "dcst.cdk"
#include "geomg.cdk"
#include "glb_ld.cdk"
#include "ver.cdk"
#include "tr3d.cdk"
#include "cstv.cdk"

      integer :: i, j, k, istat, i0, in, j0, jn
      real, dimension(lminx:lmaxx,lminy:lmaxy) :: tension, shear_z, kt, kz
      real, dimension(lminx:lmaxx,lminy:lmaxy) :: smagcoef_z, smagcoef_u, smagcoef_v
      real, dimension(lminx:lmaxx,lminy:lmaxy) :: pi, th
      real, pointer, dimension (:,:,:) :: hu
      real, dimension(nk) :: base_coefM, base_coefT      
      real :: cdelta2, tension_z, shear, tension_u, shear_u, tension_v, shear_v
      real :: fact, smagparam, ismagprandtl, ismagprandtl_hu
      real :: max_coef, crit_coef
      logical :: switch_on_THETA, switch_on_hu, switch_on_fric_heat

      if( (hzd_smago_param <= 0.) .and. (hzd_smago_lnr <=0.) ) return

      smagparam= hzd_smago_param
      ismagprandtl    = 1./Hzd_smago_prandtl
      ismagprandtl_hu = 1./Hzd_smago_prandtl_hu
      switch_on_THETA = Hzd_smago_prandtl > 0.
      switch_on_hu    = (Hzd_smago_prandtl_hu > 0.)
      switch_on_fric_heat = (Hzd_smago_fric_heat > 0.)

      cdelta2 = (smagparam * Dcst_rayt_8 * Geomg_hy_8)**2
      crit_coef = 0.25*(Dcst_rayt_8*Geomg_hy_8)**2/Cstv_dt_8

      i0  = 1    + pil_w
      in  = l_ni - pil_e

      j0  = 1    + pil_s
      jn  = l_nj - pil_n

      istat = gmm_get (gmmk_smag_s, smag)
      istat = gmm_get('TR/'//trim(Tr3d_name_S(1))//':P' ,hu)

      call rpn_comm_xch_halo( hu , l_minx,l_maxx,l_miny,l_maxy,l_ni ,l_nj ,G_nk, &
                              G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )

!$omp parallel private(tension_z,shear,tension_u,shear_u,&
!$omp tension, shear_z, smagcoef_u, smagcoef_v, kt, kz, &
!$omp tension_v, shear_v, smagcoef_z, pi, th)                           

!$omp do   
      do k=1,nk
         base_coefM(k)=Hzd_smago_lnrM_8(k)*crit_coef
         base_coefT(k)=Hzd_smago_lnrT_8(k)*crit_coef
      end do
!$omp enddo

!$omp do  
      do k=1,nk

         do j=j0-2, jn+2
         do i=i0-2, in+2

            tension(i,j) = ((F_u(i,j,k) - F_u(i-1,j,k)) * Geomg_invDX_8(j)) &
                         - ((F_v(i,j,k) * geomg_invcyv_8(j) - F_v(i,j-1,k) * geomg_invcyv_8(j-1)) &
                              * geomg_invDY_8 * Geomg_cy_8(j))

            shear_z(i,j) = ((F_v(i+1,j,k) - F_v(i,j,k)) * Geomg_invDXv_8(j)) &
                         + ((F_u(i,j+1,k) * geomg_invcy_8(j+1) - F_u(i,j,k) * geomg_invcy_8(j)) &
                              * geomg_invDY_8 * Geomg_cyv_8(j))
         end do
         end do

         if(hzd_smago_param <= 0. ) then
            do j=j0-1, jn+1
            do i=i0-1, in+1

               smagcoef_u(i,j) = base_coefM(k) * geomg_invDX_8(j)
               smagcoef_v(i,j) = base_coefM(k) * geomg_cyv_8(j) * geomg_invDY_8

               kt(i,j) = geomg_cy2_8(j)  * base_coefM(k) * tension(i,j)
               kz(i,j) = geomg_cyv2_8(j) * base_coefM(k) * shear_z(i,j)

            end do
            end do
         else
            do j=j0-1, jn+1
            do i=i0-1, in+1

               tension_z = 0.25d0 * (tension(i,j) + tension(i+1,j) + tension(i,j+1) + tension(i+1,j+1))
               shear     = 0.25d0 * (shear_z(i,j) + shear_z(i-1,j) + shear_z(i,j-1) + shear_z(i-1,j-1))
               tension_u = 0.5d0 * (tension(i,j) + tension(i+1,j))
               shear_u   = 0.5d0 * (shear_z(i,j) + shear_z(i,j-1))
               tension_v = 0.5d0 * (tension(i,j+1) + tension(i,j))
               shear_v   = 0.5d0 * (shear_z(i,j) + shear_z(i-1,j))

               smag(i,j,k) = min(cdelta2 * sqrt(tension(i,j)**2 + shear**2) + base_coefM(k), crit_coef)
               smagcoef_z(i,j) = min(cdelta2 * sqrt(tension_z**2 + shear_z(i,j)**2) + base_coefM(k), crit_coef)
               smagcoef_u(i,j) = min(cdelta2 * sqrt(tension_u**2 + shear_u**2) + base_coefT(k), crit_coef)
               smagcoef_v(i,j) = min(cdelta2 * sqrt(tension_v**2 + shear_v**2) + base_coefT(k), crit_coef)

               smagcoef_u(i,j) = smagcoef_u(i,j) * geomg_invDX_8(j)
               smagcoef_v(i,j) = smagcoef_v(i,j) * geomg_cyv_8(j) * geomg_invDY_8

               kt(i,j) = geomg_cy2_8(j)  * smag(i,j,k) * tension(i,j)
               kz(i,j) = geomg_cyv2_8(j) * smagcoef_z(i,j) * shear_z(i,j)

            end do
            end do
         endif

         fact=Cstv_dt_8*Cstv_invT_m_8
         do j=j0, jn
         do i=i0, in

            F_du(i,j,k) = F_du(i,j,k) + fact * geomg_invcy2_8(j) * ( &
                     ( kt(i+1,j) - kt(i,j) ) * geomg_invDX_8(j) &
                   + ( kz(i,j) - kz(i,j-1) ) * geomg_invDY_8 )

            F_dv(i,j,k) = F_dv(i,j,k) + fact * geomg_invcyv2_8(j) * ( &
                     ( kz(i,j) - kz(i-1,j) ) * Geomg_invDXv_8(j) &
                   - ( kt(i,j+1) - kt(i,j) ) * geomg_invDY_8 )

         end do
         end do

         if (.not.Schm_hydro_L) then

            fact=Cstv_dt_8*Cstv_invT_nh_8
            do j=j0, jn
            do i=i0, in

               F_dw(i,j,k) = F_dw(i,j,k) + fact * ( &
                    geomg_invDXMu_8(j)*((smagcoef_u(i,j)   * (F_w(i+1,j,k) - F_w(i,j,k))) - &
                                        (smagcoef_u(i-1,j) * (F_w(i,j,k) - F_w(i-1,j,k))) ) &
                  + geomg_invcy_8(j)*Geomg_invDYMv_8(j) &
                                      *((smagcoef_v(i,j)   * (F_w(i,j+1,k) - F_w(i,j,k))) - &
                                        (smagcoef_v(i,j-1) * (F_w(i,j,k) - F_w(i,j-1,k))) ) )
            end do
            end do

         endif

         if (switch_on_THETA) then

            do j=j0-1, jn+1
            do i=i0-1, in+1
               pi(i,j)=exp(Dcst_cappa_8*(Ver_a_8%t(k)+Ver_b_8%t(k)*F_s(i,j)))
               th(i,j)=F_t(i,j,k)/pi(i,j)
            end do
            end do

            fact=Cstv_dt_8*Cstv_invT_8*ismagprandtl
            do j=j0, jn
            do i=i0, in

               F_dlth(i,j,k) = F_dlth(i,j,k) + fact / th(i,j) * ( &
                    geomg_invDXMu_8(j)*((smagcoef_u(i,j)   * (th(i+1,j) - th(i,j))) - &
                                        (smagcoef_u(i-1,j) * (th(i,j) - th(i-1,j))) ) &
                  + geomg_invcy_8(j)*Geomg_invDYMv_8(j) &
                                      *((smagcoef_v(i,j)   * (th(i,j+1) - th(i,j))) - &
                                        (smagcoef_v(i,j-1) * (th(i,j) - th(i,j-1))) ) )
            end do
            end do

         end if

         if (switch_on_hu) then
            fact=Cstv_dt_8*ismagprandtl_hu        
            do j=j0, jn
            do i=i0, in
            th(i,j)=fact * ( &
                    geomg_invDXMu_8(j)*((smagcoef_u(i,j)   * (hu(i+1,j,k) - hu(i,j,k))) - &
                                        (smagcoef_u(i-1,j) * (hu(i,j,k) - hu(i-1,j,k))) ) &
                  + geomg_invcy_8(j)*Geomg_invDYMv_8(j) &
                                      *((smagcoef_v(i,j)   * (hu(i,j+1,k) - hu(i,j,k))) - &
                                        (smagcoef_v(i,j-1) * (hu(i,j,k) - hu(i,j-1,k))) ) )
            end do
            end do
            do j=j0, jn
            do i=i0, in
            hu(i,j,k)=hu(i,j,k)+th(i,j)
            end do
            end do
         endif

      end do
!$omp enddo

      if (switch_on_fric_heat) then
         fact=0.5d0*Cstv_dt_8*Cstv_invT_8/cdelta2**2/Dcst_cpd_8
!$omp do
         do k=1, nk
            if(k.ne.nk) then
               do j=j0, jn
               do i=i0, in
                 F_dlth(i,j,k)=F_dlth(i,j,k) &
                      +fact/F_t(i,j,k)*(smag(i,j,k+1)**3+smag(i,j,k)**3)
               end do
               end do
            else
               do j=j0, jn
               do i=i0, in
                 F_dlth(i,j,k)=F_dlth(i,j,k) &
                      +fact/F_t(i,j,k)*smag(i,j,k)**3
               end do
               end do
            endif
         end do
!$omp enddo
      endif

!$omp end parallel
      return
      end subroutine smago_in_rhs
