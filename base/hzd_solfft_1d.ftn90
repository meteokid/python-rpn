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

!**s/r hzd_solfft_1d - parallel direct solution of 1D high-order diffusion 
!                      equation with ffft8 (based on HZD_SOLFFT, A.Qaddouri) 


!
       subroutine hzd_solfft_1d (F_sol, F_Rhs_8, F_pri_8, F_deltai_8, &
                   minx1, maxx1, minx2, maxx2, nx1, nx2, nx3, F_pwr, &
                   minx,maxx,miny,maxy,gnk,gni,nil,njl,nkl, &
                   F_opsxp0_8, F_opsyp0_8,F_cdiff,F_npex,F_npey)
!
      implicit none
#include <arch_specific.hf>
!
      integer  minx1, maxx1, minx2, maxx2 , nx1, nx2, nx3, F_pwr, &
               minx , maxx , miny , maxy  , gnk, gni, &
               njl  , nkl  , nil  , F_npex, F_npey
      real*8  F_opsxp0_8(*), F_opsyp0_8(*), F_pri_8, &
              F_deltai_8(1:F_pwr,1:F_pwr,1:gni,nx3),F_Rhs_8
      real   F_cdiff, F_sol(minx:maxx,miny:maxy,gnk)
!
!author
!    M.Tanguay
!
!revision
! v3_20 - Tanguay M.        - initial version
! v3_21 - Tanguay M.        - Revision Openmp
!
!object
!    see id section 
!
!arguments
!  Name        I/O                 Description
!----------------------------------------------------------------
!  F_sol        I/O      r.h.s. and result of horizontal diffusion
!  F_Rhs_8      I        work vector
!----------------------------------------------------------------
!
!*
#include "ptopo.cdk"
      real*8   fdg1_8( miny:maxy, minx1:maxx1,gni+  F_npex), &
               fwft_8( miny:maxy ,minx1:maxx1,gni+2+F_npex)
!
      real*8   ZERO_8, fact_8
      parameter( ZERO_8 = 0.0 )      
!
      integer o1,o2,i,j,k,jw,offj,indy
!
!     __________________________________________________________________
!
!
! ----------
! Resolution
! ----------
!
      fact_8=((-1)**F_pwr)*dble(F_cdiff)
      call rpn_comm_transpose48 ( F_sol, Minx, Maxx, Gni,1, (Maxy-Miny+1), &
                                (Maxy-Miny+1),minx1, maxx1, Gnk, fdg1_8, 1, &
                                   fact_8,0.d0)
!
!$omp parallel private(indy)
!$omp do
      do i = 1, Gni
         do k = 1, Nkl
         do j= njl+1,maxy
            fwft_8(j,k,i) = ZERO_8
         enddo
         enddo
!
         do k= Nkl+1, maxx1
            do j= miny,maxy
               fwft_8(j,k,i) = ZERO_8
            enddo
         enddo
!
         do k=1,nkl
            do j= miny,0
               fwft_8(j,k,i) = ZERO_8
            enddo
            do j=1,njl
               fwft_8(j,k,i) = F_opsxp0_8(gni+i)*fdg1_8(j,k,i)
            enddo
         enddo
      enddo
!$omp enddo
!$omp do
      do i= 1,gni+2
         do k=minx1,0
            do j= miny,maxy
               fwft_8(j,k,i) = ZERO_8
            enddo
         enddo
      enddo
!$omp enddo
!
!     projection ( wfft = x transposed * g )
!     --------------------------------------
!
!$omp do
      do k=1,Nkl
         call ffft8 (fwft_8(miny,k,1),(Maxy-Miny+1)*(maxx1-minx1+1), &
                           1, (Maxy-Miny+1) , -1 )
      enddo
!$omp enddo
!
!$omp do
      do i = 0, (Gni)/2
         do k = 1, Nkl
!JJ         do jw = 1, (Maxy-Miny+1)
            do jw = 1, njl
               fwft_8(jw,k,2*i+1) = F_pri_8 * fwft_8(jw,k,2*i+1)
               fwft_8(jw,k,2*i+2) = F_pri_8 * fwft_8(jw,k,2*i+2)
            enddo
         enddo
      enddo
!$omp enddo
!$omp do
      do k = 1, Nkl
         do j = 1, (Maxy-Miny+1)
            fwft_8(j,k,Gni+2) = ZERO_8
            fwft_8(j,k,2)     = fwft_8(j,k,1)
         enddo
      enddo
!$omp enddo
!
!     Scaling only (WE DO NOT RESOLVE A TRIDIAGONAL MATRIX in Y)  
!     ----------------------------------------------------------
!
      offj = Ptopo_gindx(3,Ptopo_myproc+1)-1
!
!$omp do                                                                       
      do i = 1,gni
      do k = 1,nkl
      do j = 1,njl
         indy = offj + j
         fdg1_8(j,k,i)=F_deltai_8(F_pwr,1,i,indy)* &
                      (F_opsyp0_8(nx3+indy)*fwft_8(j,k,i+1))
      enddo
      enddo
      enddo
!$omp enddo
!
!     inverse projection ( r = x * w )
!     --------------------------------
!
!$omp do
      do i = 1, Gni
         do k = 1, nkl
            do j=1,njl
               fwft_8(j,k,i+1) = fdg1_8(j,k,i)
            enddo
         enddo
      enddo
!$omp enddo
!$omp do
      do k = 1, nkl
         do j = 1, (Maxy-Miny+1)
            fwft_8(j,k,1)     = fwft_8(j,k,2)
            fwft_8(j,k,2)     = ZERO_8
            fwft_8(j,k,Gni+2) = ZERO_8
         enddo
      enddo
!$omp enddo
!
!$omp do
      do k=1,Nkl
         call ffft8 (fwft_8(Miny,k,1),(Maxy-Miny+1)*(maxx1-minx1+1),1, &
                                     (Maxy-Miny+1), +1 )
      enddo
!$omp enddo
!$omp end parallel

      call rpn_comm_transpose48 ( F_sol, Minx, Maxx, Gni,1, (Maxy-Miny+1), &
                                (Maxy-Miny+1),minx1, maxx1, Gnk, fwft_8, -1, &
                                   1.d0,0.d0)
!
!     __________________________________________________________________
!
      return
      end
