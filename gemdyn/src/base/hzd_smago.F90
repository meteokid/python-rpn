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

!**s/r hzd_smago - Applies horizontal Smagorinsky-type nonlinear diffusion
!
      subroutine hzd_smago (ut1, vt1, zdt1, tt1, lminx, lmaxx, lminy, lmaxy, nk)
      implicit none
#include <arch_specific.hf>

      integer, intent(in) :: lminx,lmaxx,lminy,lmaxy, nk
      real, dimension(lminx:lmaxx,lminy:lmaxy,nk), intent(inout) :: ut1, vt1, zdt1, tt1
!
!author
!     Stéphane Gaudreault -- June 2014
!
!revision
! v4_80 - Gaudreault S.    - initial version
#include "hzd.cdk"
#include "cstv.cdk"
#include "dcst.cdk"
#include "geomg.cdk"
#include "glb_ld.cdk"
#include "grd.cdk"

      integer :: i, j, k, istat, i0, in, j0, jn
      real, dimension(lminx:lmaxx,lminy:lmaxy,nk) :: smagcoef_theta_u, smagcoef_theta_v, smagcoef, smagcoef_u, smagcoef_v, du, dv
      real, dimension(lminx:lmaxx,lminy:lmaxy) :: tension, shear_z, kt, kz
      real :: cdelta2, smagcoef_z, tension_z, shear, tension_u, shear_u, tension_v, shear_v, smagparam, smagprandtl
!      real :: kmax
      logical :: switch_on_THETA

      smagparam= hzd_smago_param ;  smagprandtl= Hzd_smago_prandtl
      switch_on_THETA = smagprandtl > 0.

      cdelta2 = (smagparam * Dcst_rayt_8 * Geomg_hy_8)**2
!      kmax= (Dcst_rayt_8 * Geomg_hy_8)**2 / (8.0d0 * cstv_dt_8)
!      print*,'kmax=',kmax,'0.04*kmax=',0.04d0*kmax,'cdelta2=',cdelta2

      if (Grd_yinyang_L) then
         call yyg_nestuv(ut1, vt1, l_minx, l_maxx, l_miny, l_maxy, G_nk)
         call yyg_xchng(zdt1, l_minx, l_maxx, l_miny, l_maxy, G_nk, .false., 'CUBIC')

         if (switch_on_THETA) then
            call yyg_xchng(tt1, l_minx, l_maxx, l_miny, l_maxy, G_nk, .false., 'CUBIC')
         end if
      endif

      call rpn_comm_xch_halo( ut1, l_minx,l_maxx,l_miny,l_maxy,l_niu,l_nj, G_nk, &
                              G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      call rpn_comm_xch_halo( vt1, l_minx,l_maxx,l_miny,l_maxy,l_ni ,l_njv, G_nk, &
                              G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      call rpn_comm_xch_halo( zdt1, l_minx,l_maxx,l_miny,l_maxy,l_ni ,l_njv, G_nk, &
                              G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )

      if (switch_on_THETA) then
         call rpn_comm_xch_halo( tt1, l_minx,l_maxx,l_miny,l_maxy,l_ni ,l_njv, G_nk, &
                                 G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      end if

      i0  = 1    + pil_w
      in  = l_ni - pil_e

      j0  = 1    + pil_s
      jn  = l_nj - pil_n

      do k=1,nk

         do j=j0-2, jn+2
            do i=i0-2, in+2

               tension(i,j) = ((ut1(i,j,k) - ut1(i-1,j,k)) * Geomg_invDX_8(j)) &
                            - ((vt1(i,j,k) * geomg_invcyv_8(j) - vt1(i,j-1,k) * geomg_invcyv_8(j-1)) &
                                 * geomg_invDY_8 * Geomg_cy_8(j))

               shear_z(i,j) = ((vt1(i+1,j,k) - vt1(i,j,k)) * Geomg_invDXv_8(j)) &
                            + ((ut1(i,j+1,k) * geomg_invcy_8(j+1) - ut1(i,j,k) * geomg_invcy_8(j)) &
                                 * geomg_invDY_8 * Geomg_cyv_8(j))
            end do
         end do

         do j=j0-1, jn+1
            do i=i0-1, in+1

               tension_z = 0.25d0 * (tension(i,j) + tension(i+1,j) + tension(i,j+1) + tension(i+1,j+1))
               shear     = 0.25d0 * (shear_z(i,j) + shear_z(i-1,j) + shear_z(i,j-1) + shear_z(i-1,j-1))
               tension_u = 0.5d0 * (tension(i,j) + tension(i+1,j))
               shear_u   = 0.5d0 * (shear_z(i,j) + shear_z(i,j-1))
               tension_v = 0.5d0 * (tension(i,j+1) + tension(i,j))
               shear_v   = 0.5d0 * (shear_z(i,j) + shear_z(i-1,j))

               smagcoef(i,j,k)   = cdelta2 * sqrt(tension(i,j)**2 + shear**2)
               smagcoef_z        = cdelta2 * sqrt(tension_z**2 + shear_z(i,j)**2)
               smagcoef_u(i,j,k) = cdelta2 * sqrt(tension_u**2 + shear_u**2)
               smagcoef_v(i,j,k) = cdelta2 * sqrt(tension_v**2 + shear_v**2)

               kt(i,j) = geomg_cy2_8(j)  * smagcoef(i,j,k) * tension(i,j)
               kz(i,j) = geomg_cyv2_8(j) * smagcoef_z      * shear_z(i,j)
            end do
         end do

         do j=j0, jn
            do i=i0, in

            ut1(i,j,k) = ut1(i,j,k) + cstv_dt_8 * geomg_invcy2_8(j) * ( &
                         ( kt(i+1,j) - kt(i,j) ) * geomg_invDX_8(j) &
                       + ( kz(i,j) - kz(i,j-1) ) * geomg_invDY_8 )


            vt1(i,j,k) = vt1(i,j,k) + cstv_dt_8 * geomg_invcyv2_8(j) * ( &
                         ( kz(i,j) - kz(i-1,j) ) * Geomg_invDXv_8(j) &
                       - ( kt(i,j+1) - kt(i,j) ) * geomg_invDY_8 )


            zdt1(i,j,k) = zdt1(i,j,k) + cstv_dt_8 * ( &
                          geomg_invDXMu_8(j) * ( (smagcoef_u(i,j,k)   * geomg_invDX_8(j) * (zdt1(i+1,j,k) - zdt1(i,j,k))) - &
                                                 (smagcoef_u(i-1,j,k) * geomg_invDX_8(j) * (zdt1(i,j,k) - zdt1(i-1,j,k))) ) &
                        + geomg_invcy_8(j) * Geomg_invDYMv_8(j) &
                                             * ( (smagcoef_v(i,j,k)   * geomg_cyv_8(j)   * geomg_invDY_8 * (zdt1(i,j+1,k) - zdt1(i,j,k))) - &
                                                 (smagcoef_v(i,j-1,k) * geomg_cyv_8(j-1) * geomg_invDY_8 * (zdt1(i,j,k) - zdt1(i,j-1,k))) ) )

            end do
         end do

      end do

      if (switch_on_THETA) then

         do k=1,nk

            do j=j0-1, jn+1
               do i=i0-1, in+1
                  smagcoef_theta_u(i,j,k) = smagcoef_u(i,j,k) / smagprandtl
                  smagcoef_theta_v(i,j,k) = smagcoef_v(i,j,k) / smagprandtl
               end do
            end do

            do j=j0, jn
               do i=i0, in

                  tt1(i,j,k) = tt1(i,j,k) + cstv_dt_8 * ( &
                               geomg_invDXMu_8(j) * ( (smagcoef_theta_u(i,j,k)   * geomg_invDX_8(j) * (tt1(i+1,j,k) - tt1(i,j,k))) - &
                                                      (smagcoef_theta_u(i-1,j,k) * geomg_invDX_8(j) * (tt1(i,j,k) - tt1(i-1,j,k))) ) &
                             + geomg_invcy_8(j) * Geomg_invDYMv_8(j) &
                                                * ( (smagcoef_theta_v(i,j,k)   * geomg_cyv_8(j)   * geomg_invDY_8 * (tt1(i,j+1,k) - tt1(i,j,k))) - &
                                                    (smagcoef_theta_v(i,j-1,k) * geomg_cyv_8(j-1) * geomg_invDY_8 * (tt1(i,j,k) - tt1(i,j-1,k))) ) )
               end do
            end do
         end do

      end if

      return
      end subroutine hzd_smago
