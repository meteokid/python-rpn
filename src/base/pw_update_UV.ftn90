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

!**s/r pw_update_UV - Update physical quantities UU and VV
!
      subroutine pw_update_UV
      implicit none
#include <arch_specific.hf>

!author
!     Michel Desgagne - May 2010
!
!revision
! v4_14 - Desgagne, M.     - Initial revision
! v4.7  - Gaudreault S.    - Removing wind images

#include "gmm.hf"
#include "glb_ld.cdk"
#include "vt1.cdk"
#include "pw.cdk"

      integer k, istat
!     ________________________________________________________________
!
      istat = gmm_get (gmmk_pw_uu_plus_s, pw_uu_plus)
      istat = gmm_get (gmmk_pw_vv_plus_s, pw_vv_plus)
      istat = gmm_get (gmmk_ut1_s       , ut1       )
      istat = gmm_get (gmmk_vt1_s       , vt1       )

!$omp parallel
!$omp do
      do k= 1, l_nk
         pw_uu_plus (1:l_ni,1:l_nj,k) = ut1(1:l_ni,1:l_nj,k)
         pw_vv_plus (1:l_ni,1:l_nj,k) = vt1(1:l_ni,1:l_nj,k)
      enddo
!$omp enddo
!$omp end parallel
!
      call itf_phy_uvgridscal ( pw_uu_plus, pw_vv_plus, &
               l_minx,l_maxx,l_miny,l_maxy,l_nk, .true. )
!     ________________________________________________________________
!
      return
      end
