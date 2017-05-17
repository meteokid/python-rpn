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

!**s/r hzd_solfft_lam - parallel direct sol_8ution of high-order diffusion 
!                   equation with fffts


       subroutine hzd_solfft_lam2(F_sol, F_Rhs_8                   , &
                              F_a_8, F_c_8, F_deltai_8             , &
                   minx1, maxx1, minx2, maxx2, nx1, nx2, nx3, F_pwr, &
                   minx,maxx,miny,maxy,gnk,Gni,nil,njl,nkl         , &
                   F_opsxp0_8, F_opsyp0_8,F_cdiff,F_npex,F_npey)
!
      use grid_options
      implicit none
#include <arch_specific.hf>
!
      integer  minx1, maxx1, minx2, maxx2 , nx1, nx2, nx3, F_pwr, &
               minx , maxx , miny , maxy  , gnk, Gni, &
               njl  , nkl  , nil  , F_npex, F_npey
      real*8  F_opsxp0_8(*), F_opsyp0_8(*), &
                  F_a_8(1:F_pwr,1:F_pwr,minx2:maxx2,nx3), &
                  F_c_8(1:F_pwr,1:F_pwr,minx2:maxx2,nx3), &
             F_deltai_8(1:F_pwr,1:F_pwr,minx2:maxx2,nx3), &
             F_Rhs_8(minx:maxx,miny:maxy,gnk)
      real   F_cdiff, F_sol(minx:maxx,miny:maxy,gnk)
!
!author
!     Abdessamad Qaddouri
!
!revision
! v2_10 - Qaddouri A.        - initial version
! v3_00 - Desgagne & Lee    - Lam configuration
! v3_10 - Corbeil & Desgagne & Lee - AIXport+Opti+OpenMP
! v3_30 - Lee/Qaddouri - openMP revision and bug correction
!                      - OMP single on LOOP is suspect
!
!object
!
!arguments
!  Name        I/O                 Description
!----------------------------------------------------------------
!  F_sol        I/O      r.h.s. and result of horizontal diffusion
!  F_Rhs_8         I        work vector
!
!----------------------------------------------------------------
!
#include "ptopo.cdk"
#include "glb_ld.cdk"
#include "glb_pil.cdk"

      character*4 type_fft
      integer o1,o2,i,j,k,l_pil_w,l_pil_e
      integer ki,kkii,ki0,kin,kilon,kitotal,pi0,pin
      real*8  fdg1_8(miny :maxy ,minx1:maxx1,Gni+F_npex  )
      real*8  fwft_8(miny :maxy ,minx1:maxx1,Gni+2+F_npex)
      real*8  fdg2_8(minx1:maxx1,minx2:maxx2,nx3+F_npey  )
      real*8  dn3_8 (minx1:maxx1,minx2:maxx2,F_pwr,nx3   )
      real*8  sol_8 (minx1:maxx1,minx2:maxx2,F_pwr,nx3   ), pri
      real*8, parameter :: ZERO_8=0.0
!     __________________________________________________________________
!
!  The I vector lies on the Y processor so, l_pil_w and l_pil_e will
!  represent the pilot region along I
      l_pil_w=0
      l_pil_e=0
      if (l_south) l_pil_w= Lam_pil_w
      if (l_north) l_pil_e= Lam_pil_e

      kilon = (maxx2-l_pil_e-minx2-l_pil_w +1 +Ptopo_npeOpenMP)/Ptopo_npeOpenMP

      type_fft = 'QCOS'
      if (Grd_yinyang_L) type_fft = 'SIN'

      call itf_fft_set ( G_ni-Lam_pil_w-Lam_pil_e, type_fft, pri )

!$omp parallel private(ki0,kin,pi0,pin)
!$&            shared (kilon)
!$omp do
      do k = 1, gnk
      do j = 1+pil_s, njl-pil_n
      do i = 1+pil_w,nil-pil_e
         F_Rhs_8(i,j,k) = ((-1)**F_pwr)*dble(F_cdiff)*dble(F_sol(i,j,k))
      enddo
      enddo
      enddo
!$omp enddo
!$omp single
!       
      call rpn_comm_transpose( F_Rhs_8, Minx, Maxx, Gni, (Maxy-Miny+1), &
                                      Minx1, Maxx1, gnk, fdg1_8,1, 2 )
!$omp end single
!
!$omp do
      do i = 1+Lam_pil_w, Gni-Lam_pil_e
         do k = 1, nkl
         do j = 1+pil_s, njl-pil_n
            fdg1_8(j,k,i) = F_opsxp0_8(Gni+i)*fdg1_8(j,k,i)
         enddo
         enddo
      enddo
!$omp enddo
!$omp do
      do i= 1,Gni
         do k = 1, nkl
         do j= njl+1-pil_n,maxy
            fwft_8(j,k,i) = ZERO_8
         enddo
         enddo
         do k = 1, nkl
         do j= miny,pil_s
            fwft_8(j,k,i) = ZERO_8
         enddo
         enddo
      enddo
!$omp enddo
!
!$omp do
      do i= 1,Gni
      do k= Nkl+1, maxx1
      do j= miny,maxy
         fwft_8(j,k,i) = ZERO_8
      enddo
      enddo
      enddo
!$omp enddo
!
!$omp do
      do i= 1,Gni
      do k= minx1, 0
      do j= miny,maxy
         fwft_8(j,k,i) = ZERO_8
      enddo
      enddo
      enddo
!$omp enddo
!
!$omp do
      do i = 1+Lam_pil_w, Gni-Lam_pil_e
         do k=1,nkl
         do j=1+pil_s,njl-pil_n
            fwft_8(j,k,i) = fdg1_8(j,k,i)
         enddo
         enddo
      enddo
!$omp enddo
!
!     projection ( wfft = x transposed * g )
!
!$omp do
      do k=1,Nkl
         call itf_fft_drv (fwft_8(1+pil_s,k,1+Lam_pil_w), &
                     (Maxy-Miny+1)*(maxx1-minx1+1),1    , &
                     (Maxy-Miny+1-pil_s-pil_n), -1 )
      enddo
!$omp enddo

!$omp do
      do i = 0+Lam_pil_w, Gni-1-Lam_pil_e
         do k = 1, Nkl
            do j = 1+pil_s, (Maxy-Miny+1)-pil_n
               fwft_8(j,k,i+1) = pri * fwft_8(j,k,i+1)
            enddo
         enddo
      enddo
!$omp enddo

!$omp single
!
      call rpn_comm_transpose  &
           (fwft_8,Miny,Maxy,nx3, (Maxx1-Minx1+1), &
                        minx2, maxx2, Gni, fdg2_8, 2, 2)
!$omp end single
!
! cote droit
!
!$omp do
      do j = 1, nx3
      do o1= 1, F_pwr
      do i = minx2, maxx2
      do k = minx1, maxx1
         sol_8(k,i,o1,j) = ZERO_8
         dn3_8(k,i,o1,j) = ZERO_8
      enddo
      enddo
      enddo
      enddo
!$omp enddo
!
!$omp do
      do j = 1+Lam_pil_s, nx3-Lam_pil_n
      do i = 1+l_pil_w, nx2-l_pil_e
      do k = 1, nx1
         dn3_8(k,i,1,j)= F_opsyp0_8(nx3+j)*fdg2_8(k,i,j)
      enddo
      enddo
      enddo
!$omp enddo
!
! resolution du systeme blok-tridiagonal
!
! aller
!
!$omp do
      do o1= 1,F_pwr
      do i = 1+l_pil_w, nx2-l_pil_e
      do k= 1, nx1
         sol_8(k,i,o1,1+Lam_pil_s) = dn3_8(k,i,o1,1+Lam_pil_s)
      enddo
      enddo
      enddo
!$omp enddo
!
!$omp do
      do kkii = 1, Ptopo_npeOpenMP
         ki0 = minx2+l_pil_w + kilon*(kkii-1)
         kin = min(ki0+kilon-1, maxx2-l_pil_e)
         pi0 = 1+l_pil_w + kilon*(kkii-1)
         pin = min(pi0+kilon-1, nx2-l_pil_e)
         do j = 2+Lam_pil_s, nx3-Lam_pil_n
            do o1= 1, F_pwr
            do o2= 1, F_pwr
            do i = ki0,kin
            do k = 1, nx1
               sol_8(k,i,o1,j)= sol_8(k,i,o1,j) &
                             + F_a_8(o1,o2,i,j)*sol_8(k,i,o2,j-1)
            enddo
            enddo
            enddo
            enddo
            do o1= 1,F_pwr
            do i = pi0,pin
            do k= 1, nx1
               sol_8(k,i,o1,j) = dn3_8(k,i,o1,j) - sol_8(k,i,o1,j)
            enddo
            enddo
            enddo
         enddo
      enddo
!$omp enddo
!
! scale le cote droit pour retour
!
!$omp do
      do j = 1, nx3
         do o1= 1, F_pwr
         do i = minx2, maxx2
         do k = minx1, maxx1
            dn3_8(k,i,o1,j) = ZERO_8
         enddo
         enddo
         enddo
      enddo
!$omp enddo
!$omp do
      do j = 1+Lam_pil_s, nx3-Lam_pil_n
         do o2=1,F_pwr
         do o1=1,F_pwr
         do i= minx2+l_pil_w,maxx2-l_pil_e
         do k= minx1,maxx1
            dn3_8(k,i,o1,j)= dn3_8(k,i,o1,j) &
                             + F_deltai_8(o1,o2,i,j)*sol_8(k,i,o2,j)
         enddo
         enddo
         enddo
         enddo
      enddo
!$omp enddo
!
! retour
!
!$omp do
      do j = 1, nx3
      do o1= 1, F_pwr
      do i = 1, nx2
         do k = 1, nx1
            sol_8(k,i,o1,j)=0.0
         enddo
      enddo
      enddo
      enddo
!$omp enddo

!
! Maybe better if OpenMP on i (LC), done (VL)
!
!$omp do
      do i = 1+l_pil_w, nx2-l_pil_e
      do o1= 1, F_pwr
      do k = 1, nx1
         sol_8(k,i,o1,nx3-Lam_pil_n)=dn3_8(k,i,o1,nx3-Lam_pil_n)
      enddo
      enddo
      enddo
!$omp enddo
!
!$omp do
      do kkii = 1, Ptopo_npeOpenMP
         ki0 = minx2+l_pil_w + kilon*(kkii-1)
         kin = min(ki0+kilon-1, maxx2-l_pil_e)
         pi0 = 1+l_pil_w + kilon*(kkii-1)
         pin = min(pi0+kilon-1, nx2-l_pil_e)
         do j = nx3-1-Lam_pil_n, 1+Lam_pil_s, -1
            do o1= 1, F_pwr
            do o2= 1, F_pwr
            do k = minx1, maxx1
            do i = ki0, kin
               sol_8(k,i,o1,j)= sol_8(k,i,o1,j) &
                          + F_c_8(o1,o2,i,j)*sol_8(k,i,o2,j+1)
            enddo
            enddo
            enddo
            enddo
!
            do o1= 1, F_pwr
            do i = pi0,pin
            do k = 1, nx1
               sol_8(k,i,o1,j)=dn3_8(k,i,o1,j)-sol_8(k,i,o1,j)
            enddo
            enddo
            enddo
         enddo
      enddo
!$omp enddo
!
!$omp do
      do j = 1+Lam_pil_s, nx3-Lam_pil_n
      do i = 1+l_pil_w, nx2-l_pil_e
      do k = 1, nx1
         fdg2_8(k,i,j)=sol_8(k,i,F_pwr,j)
      enddo
      enddo
      enddo
!$omp enddo
!
!     inverse projection ( r = x * w )
!
!$omp single
      call rpn_comm_transpose &
           ( fwft_8 , Miny, Maxy, nx3, (Maxx1-Minx1+1), &
                     minx2, maxx2,Gni, fdg2_8,- 2, 2 )
!$omp end single
!
!$omp do
      do k=1,Nkl
         call itf_fft_drv (fwft_8(1+pil_s,k,1+Lam_pil_w), &
                     (Maxy-Miny+1) * (maxx1-minx1+1), 1 , &
                     (Maxy-Miny+1-pil_s-pil_n), +1 )
      enddo
!$omp enddo

!$omp single
      call rpn_comm_transpose( F_Rhs_8,Minx, Maxx, Gni, (Maxy-Miny+1), &
                                    Minx1, Maxx1, gnk, fwft_8, -1, 2 )
!$omp end single
!$omp do
      do k = 1, gnk
      do j = 1+pil_s, njl-pil_n
      do i = 1+pil_w, nil-pil_e
         F_sol(i,j,k) = sngl(F_Rhs_8(i,j,k))
      enddo
      enddo
      enddo
!$omp enddo
!$omp end parallel
!     __________________________________________________________________
!
      return
      end

