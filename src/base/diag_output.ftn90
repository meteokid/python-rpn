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

!** s/r diag_output - Computes diagnostic variables for output
!
      subroutine diag_output
      implicit none
#include <arch_specific.hf>

!authors
!      Michel Desgagne   -  Spring 2014
!
!revision
!
! v4.70  - Desgagne M.   - initial version

#include <gmm.hf>
#include "glb_ld.cdk"
#include "cstv.cdk"
#include "pw.cdk"

      integer istat,k
      real*8 idt
!     ________________________________________________________________
!
      istat = gmm_get (gmmk_pw_uu_plus_s ,pw_uu_plus )
      istat = gmm_get (gmmk_pw_vv_plus_s ,pw_vv_plus )
      istat = gmm_get (gmmk_pw_tt_plus_s ,pw_tt_plus )
      istat = gmm_get (gmmk_pw_uu_moins_s,pw_uu_moins)
      istat = gmm_get (gmmk_pw_vv_moins_s,pw_vv_moins)
      istat = gmm_get (gmmk_pw_tt_moins_s,pw_tt_moins)
      istat = gmm_get (gmmk_pw_uu_copy_s ,pw_uu_copy )
      istat = gmm_get (gmmk_pw_vv_copy_s ,pw_vv_copy )
      istat = gmm_get (gmmk_pw_tt_copy_s ,pw_tt_copy )
      istat = gmm_get (gmmk_pw_uu_dyn_s ,pw_uu_dyn )
      istat = gmm_get (gmmk_pw_vv_dyn_s ,pw_vv_dyn )
      istat = gmm_get (gmmk_pw_tt_dyn_s ,pw_tt_dyn )

      idt = 1.d0/Cstv_dt_8

!$omp parallel
!$omp do
      do k= 1, G_nk
         pw_uu_dyn(1:l_ni,1:l_nj,k)= idt * (pw_uu_plus (1:l_ni,1:l_nj,k) &
              -pw_uu_moins(1:l_ni,1:l_nj,k)-pw_uu_copy (1:l_ni,1:l_nj,k))
         pw_vv_dyn(1:l_ni,1:l_nj,k)= idt * (pw_vv_plus (1:l_ni,1:l_nj,k) &
              -pw_vv_moins(1:l_ni,1:l_nj,k)-pw_vv_copy (1:l_ni,1:l_nj,k))
         pw_tt_dyn(1:l_ni,1:l_nj,k)= idt * (pw_tt_plus (1:l_ni,1:l_nj,k) &
              -pw_tt_moins(1:l_ni,1:l_nj,k)-pw_tt_copy (1:l_ni,1:l_nj,k))
      end do
!$omp enddo
!$omp end parallel


!     ________________________________________________________________
!
      return
      end
