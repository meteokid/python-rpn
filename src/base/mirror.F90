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

!**s/r mirror - mirror boundary condition for BUBBLE theoretical
!               case
!
      subroutine mirror ()
      use gmm_vt1
      use gem_options
      use glb_ld
      use lun
      use tr3d
      use gmm_itf_mod
      implicit none
#include <arch_specific.hf>

!author 
!     Gravel              - spring 2003 (after MC2 v_4.9.3)
!
!revision
! v3_11 - Gravel             - initial version
! v4_05 - Lepine M.          - VMM replacement with GMM
!


      type(gmm_metadata) :: mymeta
      integer err,i,j,k,istat
      integer ii, iin, jj, jin
!
!----------------------------------------------------------------------
!
      istat = gmm_get(gmmk_ut1_s,ut1,mymeta)
      if (GMM_IS_ERROR(istat)) print *,'mirror ERROR at gmm_get(ut1)'
      istat = gmm_get(gmmk_vt1_s,vt1,mymeta)
      if (GMM_IS_ERROR(istat)) print *,'mirror ERROR at gmm_get(vt1)'
      istat = gmm_get(gmmk_tt1_s,tt1,mymeta)
      if (GMM_IS_ERROR(istat)) print *,'mirror ERROR at gmm_get(tt1)'
      istat = gmm_get(gmmk_zdt1_s,zdt1,mymeta)
      if (GMM_IS_ERROR(istat)) print *,'mirror ERROR at gmm_get(zdt1)'
      istat = gmm_get(gmmk_qt1_s,qt1,mymeta)
      if (GMM_IS_ERROR(istat)) print *,'mirror ERROR at gmm_get(qt1)'
      istat = gmm_get(gmmk_st1_s,st1,mymeta)
      if (GMM_IS_ERROR(istat)) print *,'mirror ERROR at gmm_get(st1)'
      istat = gmm_get(gmmk_wt1_s,wt1,mymeta)
      if (GMM_IS_ERROR(istat)) print *,'mirror ERROR at gmm_get(wt1)'
!
      if (l_north) then
         do k=1,G_nk
            do i=1,l_ni
               vt1  (i,l_nj-pil_n,k) = 0.
            end do
            do j=1,pil_n-1
            jin = max(pil_s+1,l_nj-pil_n-j)
            jj = l_nj-pil_n+j
            do i=1,l_ni
               vt1  (i,jj,k) = - vt1  (i,jin,k)
            end do
            end do
            do j=1,pil_n
            jin = max(pil_s+1,l_nj-pil_n-j+1)
            jj  = l_nj-pil_n+j
            do i=1,l_ni
               tt1 (i,jj,k) = tt1 (i,jin,k)
               zdt1(i,jj,k) = zdt1(i,jin,k)
               wt1 (i,jj,k) = wt1 (i,jin,k)
               qt1 (i,jj,k) = qt1 (i,jin,k)
            end do
            do i=1,l_niu
               ut1   (i,jj,k) = ut1   (i,jin,k)
            end do
            end do
         end do
         do j=1,pil_n
         jin = max(pil_s+1,l_nj-pil_n-j+1)
         jj  = l_nj-pil_n+j
         do i=1,l_ni
            st1(i,jj)        = st1(i,jin)
            qt1(i,jj,G_nk+1) = qt1(i,jin,G_nk+1)
         end do
         end do
      endif
!
      if (l_east) then
         do k=1,G_nk
            do j=1,l_nj
               ut1  (l_ni-pil_e,j,k) = 0.
            end do
            do j=1,l_nj
            do i=1,pil_e-1
               iin = max(pil_w+1,l_ni-pil_e-i)
               ii  = l_ni-pil_e+i
               ut1  (ii,j,k) = - ut1  (iin,j,k)
            end do
            end do
            do j=1,l_nj
            do i=1,pil_e
               iin = max(pil_w+1,l_ni-pil_e-i+1)
               ii  = l_ni-pil_e+i
               tt1 (ii,j,k) = tt1 (iin,j,k)
               zdt1(ii,j,k) = zdt1(iin,j,k)
               wt1 (ii,j,k) = wt1 (iin,j,k)
               qt1 (ii,j,k) = qt1 (iin,j,k)
            end do
            end do
            do j=1,l_njv
            do i=1,pil_e
               iin = max(pil_w+1,l_ni-pil_e-i+1)
               ii  = l_ni-pil_e+i
               vt1  (ii,j,k) = vt1  (iin,j,k)
            end do
            end do
         end do
         do j=1,l_nj
         do i=1,pil_e
            iin = max(pil_w+1,l_ni-pil_e-i+1)
            ii  = l_ni-pil_e+i
            st1(ii,j)        = st1(iin,j)
            qt1(ii,j,G_nk+1) = qt1(iin,j,G_nk+1)
         end do
         end do
      endif
!
      if (l_south) then
         do k=1,G_nk
            do i=1,l_ni
               vt1  (i,pil_s,k) = 0.
            end do
            do j=1,pil_s-1
            jin = min(l_nj-pil_n-1,pil_s+j)
            jj  = pil_s-j
            do i=1,l_ni
               vt1  (i,jj,k) = - vt1 (i,jin,k)
            end do
            end do
            do j=1,pil_s
            jin = min(l_nj-pil_n,pil_s+j)
            jj  = pil_s-j+1
            do i=1,l_ni
               tt1 (i,jj,k) = tt1 (i,jin,k)
               zdt1(i,jj,k) = zdt1(i,jin,k)
               wt1 (i,jj,k) = wt1 (i,jin,k)
               qt1 (i,jj,k) = qt1 (i,jin,k)
            end do
            do i=1,l_niu
               ut1  (i,jj,k) = ut1  (i,jin,k)
            end do
            end do
         end do
         do j=1,pil_s
         jin = min(l_nj-pil_n,pil_s+j)
         jj  = pil_s-j+1
         do i=1,l_ni
            st1(i,jj)        = st1(i,jin)
            qt1(i,jj,G_nk+1) = qt1(i,jin,G_nk+1)
         end do
         end do
      endif
!
      if (l_west) then
         do k=1,G_nk
            do j=1,l_nj
               ut1  (pil_w,j,k) = 0.
            end do
            do j=1,l_nj
            do i=1,pil_w-1
               iin = min(l_ni-pil_e-1,pil_w+i)
               ii  = pil_w-i
               ut1  (ii,j,k) = - ut1  (iin,j,k)
            end do
            end do
            do j=1,l_nj
            do i=1,pil_w
               iin = min(l_ni-pil_e,pil_w+i)
               ii  = pil_w-i+1
               tt1 (ii,j,k) = tt1 (iin,j,k)
               zdt1(ii,j,k) = zdt1(iin,j,k)
               wt1 (ii,j,k) = wt1 (iin,j,k)
               qt1 (ii,j,k) = qt1 (iin,j,k)
            end do
            end do
            do j=1,l_njv
            do i=1,pil_w
               iin = min(l_ni-pil_e,pil_w+i)
               ii  = pil_w-i+1
               vt1  (ii,j,k) = vt1  (iin,j,k)
            end do
            end do
         end do
         do j=1,l_nj
         do i=1,pil_w
            iin = min(l_ni-pil_e,pil_w+i)
            ii  = pil_w-i+1
            st1(ii,j)        = st1(iin,j)
            qt1(ii,j,G_nk+1) = qt1(iin,j,G_nk+1)
         end do
         end do
      endif
!
!----------------------------------------------------------------------
!
	stop 'mirror'

      return
      end
