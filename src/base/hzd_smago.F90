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
      subroutine hzd_smago (ut1, vt1, smagparam, lminx, lmaxx, lminy, lmaxy, nk)
      implicit none
#include <arch_specific.hf>

      integer, intent(in) :: lminx,lmaxx,lminy,lmaxy, nk
      real, dimension(lminx:lmaxx,lminy:lmaxy,nk), intent(inout) :: ut1, vt1
      real, intent(in) :: smagparam
!
!author
!     Stéphane Gaudreault -- June 2014
!
!revision
! v4_80 - Gaudreault S.    - initial version
#include "gmm.hf"

#include "cstv.cdk"
#include "dcst.cdk"
#include "geomg.cdk"
#include "glb_ld.cdk"
#include "grd.cdk"
#include "smago.cdk"

      integer :: i, j, k, istat, i0, in, j0, jn
      real, dimension(lminx:lmaxx,lminy:lmaxy,nk) :: smagcoef, du, dv
      real, dimension(lminx:lmaxx,lminy:lmaxy) :: tension, shear_z, kt, kz
      real cdelta2,kmax,smagcoef_z,tension_z,shear

      cdelta2 = (smagparam * Dcst_rayt_8 * Geomg_hy_8)**2
      kmax= (Dcst_rayt_8 * Geomg_hy_8)**2 / (8.0d0 * cstv_dt_8)
      print*,'kmax=',kmax,'0.04*kmax=',0.04d0*kmax,'cdelta2=',cdelta2

      istat = gmm_get(gmmk_smagU_s, smagU_p)
      istat = gmm_get(gmmk_smagV_s, smagV_p)

      if (Grd_yinyang_L) then
         call yyg_nestuv(ut1, vt1, l_minx, l_maxx, l_miny, l_maxy, G_nk)
      endif

      call rpn_comm_xch_halo( ut1, l_minx,l_maxx,l_miny,l_maxy,l_niu,l_nj, G_nk, &
                              G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      call rpn_comm_xch_halo( vt1, l_minx,l_maxx,l_miny,l_maxy,l_ni ,l_njv, G_nk, &
                              G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )

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

               smagcoef(i,j,k) = cdelta2 * sqrt(tension(i,j)**2 + shear**2)
               smagcoef_z = cdelta2 * sqrt(tension_z**2 + shear_z(i,j)**2)

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
            end do
         end do

      end do

     ! Save coefficients for output
      smagU_p = smagcoef
      smagV_p = smagcoef

      return
      end subroutine hzd_smago
