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

!**s/r frstgss - Copy data from time level t1 that will be used as a
!                first guess at time level t0

      subroutine frstgss ()
      implicit none
#include <arch_specific.hf>

!author 
!     Michel Roch - rpn - nov 1993
!
!revision
! v2_00 - Desgagne M.       - initial MPI version
! v3_21 - Tanguay M.        - Revision Openmp
! v4_05 - Lepine M.         - VMM replacement with GMM

#include "gmm.hf"
#include "vt0.cdk"
#include "vt1.cdk"
#include "tr3d.cdk"
      integer :: istat, k
      character(len=GMM_MAXNAMELENGTH) :: tr_name
      real, pointer, dimension(:,:,:) :: plus,minus
!
!     ---------------------------------------------------------------
!
      do k=1,Tr3d_ntr
         nullify (plus, minus)
         istat = gmm_get('TR/'//trim(Tr3d_name_S(k))//':M',minus)
         istat = gmm_get('TR/'//trim(Tr3d_name_S(k))//':P',plus )
         minus = plus
      enddo

      istat = gmm_get(gmmk_ut0_s , ut0)
      istat = gmm_get(gmmk_ut1_s , ut1)
      istat = gmm_get(gmmk_vt0_s , vt0)
      istat = gmm_get(gmmk_vt1_s , vt1)
      istat = gmm_get(gmmk_tt0_s , tt0)
      istat = gmm_get(gmmk_tt1_s , tt1)     
      istat = gmm_get(gmmk_st0_s , st0)
      istat = gmm_get(gmmk_st1_s , st1)
      istat = gmm_get(gmmk_wt0_s , wt0)
      istat = gmm_get(gmmk_wt1_s , wt1)
      istat = gmm_get(gmmk_qt0_s , qt0)
      istat = gmm_get(gmmk_qt1_s , qt1)
      istat = gmm_get(gmmk_zdt0_s,zdt0)
      istat = gmm_get(gmmk_zdt1_s,zdt1)
      istat = gmm_get(gmmk_xdt0_s,xdt0)
      istat = gmm_get(gmmk_xdt1_s,xdt1)
      istat = gmm_get(gmmk_qdt0_s,qdt0)
      istat = gmm_get(gmmk_qdt1_s,qdt1)

      tt0 = tt1 ; zdt0 = zdt1 ; wt0 = wt1 ; xdt0 = xdt1
      ut0 = ut1 ; vt0  = vt1  ; qt0 = qt1 ; qdt0 = qdt1
      st0 = st1
!
!     ---------------------------------------------------------------
!
      return
      end


