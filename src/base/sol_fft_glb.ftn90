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

!**s/r sol_fft_glb - parallel direct solution of an elliptic problem using FFT

      subroutine sol_fft_glb ( sol, Rhs                            , &
                               F_t0nis, F_t0njs, F_t0nj            , &
                               F_t1nks, F_t1nk , F_t2nis, F_t2ni   , &
                               F_nk, F_gni, F_gnj, F_npex1, F_npey1, &
                               F_ai, F_bi, F_ci, F_dg2, F_dwfft )
      implicit none
#include <arch_specific.hf>

      integer F_t0nis, F_t0njs, F_t0nj, F_t2nis, F_t2ni
      integer F_t1nks, F_nk, F_t1nk, F_gni, F_gnj
      integer F_npex1, F_npey1

      real*8  Sol(1:F_t0nis,1:F_t0njs,F_nk), Rhs(1:F_t0nis,1:F_t0njs,F_nk)
      Real*8  F_ai(1:F_t1nks,1:F_t2nis,F_gnj), &
              F_bi(1:F_t1nks,1:F_t2nis,F_gnj), &
              F_ci(1:F_t1nks,1:F_t2nis,F_gnj)
      real*8  F_dwfft(1:F_t0njs,1:F_t1nks,F_gni+2+F_npex1)
      real*8  F_dg2  (1:F_t1nks,1:F_t2nis,F_gnj  +F_npey1)

!author    Abdessamad Qaddouri- JULY 1999

!revision
! v1_96 - alain patoine            - rename sol_fft8, sol_fft8_2 (calling sequence changed)
! v3_10 - Corbeil & Desgagne & Lee - AIXport+Opti+OpenMP
! v4_50 - Desgagne                 - major interface revision

      integer i, j, k
      real*8 pri
      real*8, parameter :: zero=0.0
!     __________________________________________________________________
!
      call itf_fft_set ( F_gni, 'PERIODIC', pri )

      call rpn_comm_transpose( Rhs, 1, F_t0nis, F_gni, (F_t0njs-1+1), &
                               1, F_t1nks, F_nk, F_dwfft, 1, 2 )
!     F_dwfft is (j,nk/npex,G_ni)

!     projection ( wfft = x transposed * g )

!$omp parallel

!$omp do
      do i= 1,F_gni
         F_dwfft(F_t0nj+1:F_t0njs,        1:F_t1nk ,i)= zero
         F_dwfft(       1:F_t0njs, F_t1nk+1:F_t1nks,i)= zero
      enddo
!$omp enddo

!$omp do
      do k=1,F_t1nk
         call itf_fft_drv (F_dwfft(1,k,1),(F_t0njs-1+1)*(F_t1nks-1+1), &
                                                1,(F_t0njs-1+1), -1 )
      enddo
!$omp enddo

!$omp do
      do k = 1, F_t1nk
         do i = 0, (F_gni)/2
            do j = 1, (F_t0njs-1+1)
               F_dwfft(j,k,2*i+1) = pri * F_dwfft(j,k,2*i+1)
               F_dwfft(j,k,2*i+2) = pri * F_dwfft(j,k,2*i+2)
            enddo
         enddo
         F_dwfft(1:F_t0njs-1+1,k,F_gni+2) = zero
         F_dwfft(1:F_t0njs-1+1,k,      2) = F_dwfft(1:F_t0njs-1+1,k,1)
      enddo
!$omp enddo

!$omp single
      call rpn_comm_transpose &
           ( F_dwfft(1,1,2), 1, F_t0njs, F_gnj, (F_t1nks-1+1), &
                             1, F_t2nis, F_gni, F_dg2, 2, 2 )
!     F_dg2 is (nk/npex,G_ni/npey,G_nj
!$omp end single

      call sol_diag (F_dg2, F_ai, F_bi, F_ci, &
           F_t1nks*F_t2nis, F_t1nks*F_t2ni, F_gnj, F_npey1)

!$omp single
      call rpn_comm_transpose &
           ( F_dwfft(1,1,2), 1, F_t0njs, F_gnj , (F_t1nks-1+1), &
                             1, F_t2nis, F_gni, F_dg2,- 2, 2 )
!$omp end single

!$omp do
      do k = 1, F_t1nk
         F_dwfft(1:F_t0njs-1+1,k,1)       = F_dwfft(1:F_t0njs-1+1,k,2)
         F_dwfft(1:F_t0njs-1+1,k,2)       = zero
         F_dwfft(1:F_t0njs-1+1,k,F_gni+2) = zero
      enddo
!$omp enddo

!     inverse projection ( r = x * w )

!$omp do
      do k=1, F_t1nk
         call itf_fft_drv (F_dwfft(1,k,1),(F_t0njs-1+1)*(F_t1nks-1+1), &
                                                1,(F_t0njs-1+1), +1 )
      enddo
!$omp enddo

!$omp end parallel

      call rpn_comm_transpose( Sol, 1, F_t0nis, F_gni, (F_t0njs-1+1), &
                                    1, F_t1nks, F_nk , F_dwfft, -1, 2 )
!     __________________________________________________________________
!
      return
      end

      subroutine sol_diag (F_dg2,F_ai,F_bi,F_ci,F_n1s,F_n1,F_nj,F_npey)
      implicit none

      integer F_n1s,F_n1,F_nj,F_npey
      Real*8  F_ai( F_n1s,F_nj), F_bi(F_n1s,F_nj), F_ci(F_n1s,F_nj)
      real*8  F_dg2(F_n1s,F_nj+F_npey)

#include "ptopo.cdk"

      integer kitotal,kilon,kkii,ki0,kin,ki,j

      kitotal = F_n1
      kilon = (kitotal + Ptopo_npeOpenMP)/Ptopo_npeOpenMP

!$omp do
      do kkii = 1,Ptopo_npeOpenMP

          ki0 = 1 + kilon*(kkii-1)
          kin = min(kitotal, kilon*kkii)
         
          do ki= ki0, kin
             F_dg2(ki,1) = F_bi(ki,1)*F_dg2(ki,1)
          enddo

          do j = 2, F_nj
          do ki= ki0, kin
             F_dg2(ki,j) = F_bi(ki,j) * F_dg2(ki,j  ) - F_ai(ki,j) &
                                      * F_dg2(ki,j-1)
          enddo
          enddo

          do j = F_nj-1, 1, -1
          do ki= ki0, kin
             F_dg2(ki,j) = F_dg2(ki,j) - F_ci(ki,j) * F_dg2(ki,j+1)
          enddo
          enddo

      enddo
!$omp enddo

      return
      end
