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
      subroutine hzd_smago (ut1, vt1, smagparam, delta, lminx, lmaxx, lminy, lmaxy, nk)
      implicit none
#include <arch_specific.hf>

      integer, intent(in) :: lminx,lmaxx,lminy,lmaxy, nk
      real, dimension(lminx:lmaxx,lminy:lmaxy,nk), intent(inout) :: ut1, vt1
      real, intent(in) :: smagparam, delta
!
!author
!     Stéphane Gaudreault -- June 2014
!
!revision
! v4_80 - Gaudreault S.    - initial version

#include "cstv.cdk"
#include "dcst.cdk"
#include "geomg.cdk"
#include "glb_ld.cdk"

      integer :: i, j, k
      real, dimension(lminx:lmaxx,lminy:lmaxy,nk) :: tension, shear_z, shear, tension_z, smagcoef, smagcoef_z

      do k=1,nk
         do j=-1,l_nj+2
            do i=-1,l_ni+2
               tension(i,j,k) = ((ut1(i,j,k) - ut1(i-1,j,k)) * Geomg_invDX_8(j)) &
                              - ((vt1(i,j,k)*geomg_invcyv_8(j) - vt1(i,j-1,k) * geomg_invcyv_8(j-1)) &
                                 * geomg_invDY_8 * Geomg_cy_8(j))

               shear_z(i,j,k) = ((vt1(i,j,k) - vt1(i-1,j,k)) * Geomg_invDXz_8(j)) &
                              + ((ut1(i,j+1,k) * geomg_invcy_8(j+1) - ut1(i,j,k) * geomg_invcy_8(j)) &
                                 * geomg_invDYMv_8(j) * Geomg_cyv_8(j))
            end do
         end do
      end do

      do k=1,nk
         do j=0,l_nj+1
            do i=0,l_ni+1
               tension_z(i,j,k) = 0.25 * (tension(i,j,k) + tension(i+1,j,k) + tension(i,j+1,k) + tension(i+1,j+1,k))

               shear(i,j,k) =  0.25 * (shear_z(i,j,k) + shear_z(i-1,j,k) + shear_z(i-1,j-1,k) + shear_z(i,j-1,k))

               smagcoef(i,j,k) = sqrt(tension(i,j,k)**2 + shear(i,j,k)**2)
               smagcoef_z(i,j,k) = sqrt(tension_z(i,j,k)**2 + shear_z(i,j,k)**2)
            end do
         end do
      end do

      do k=1,nk
         do j=1,l_nj
            do i=1,l_ni


            ut1(i,j,k) = ut1(i,j,k) +  cstv_dt_8 * ( (smagparam * delta)**2 * (smagcoef(i,j,k) * tension(i,j,k) &
                       -  smagcoef(i-1,j,k) * tension(i-1,j,k)) * geomg_invDXMu_8(j) &
                       + ((smagparam * geomg_invcyv_8(j) * delta)**2 * smagcoef_z(i,j,k) * shear_z(i,j,k) &
                       - (smagparam * geomg_invcyv_8(j-1) * delta)**2 * smagcoef_z(i,j-1,k) &
                       * shear_z(i,j-1,k)) * Geomg_invDY_8 * geomg_invcy2_8(j) )


            vt1(i,j,k) = vt1(i,j,k) +  cstv_dt_8 * ( (smagparam * delta)**2 * (smagcoef_z(i,j,k) * shear_z(i,j,k) &
                       -  smagcoef_z(i-1,j,k) * shear_z(i-1,j,k)) * Geomg_invDX_8(j) &
                       - ((smagparam * geomg_invcy_8(j+1) * delta)**2 * smagcoef(i,j+1,k) * tension(i,j+1,k) &
                       - (smagparam * geomg_invcy_8(j)   * delta)**2 * smagcoef(i,j,k) &
                       * tension(i,j,k)) * geomg_invDYMv_8(j) * geomg_invcyv2_8(j) )
            end do
         end do
      end do

      return
      end subroutine hzd_smago
