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
!---------------------------------- LICENCE END --------------------------------

!*s/r spn_fld - doing forward 2-D FFT, applying a filter, doing backward FFT 
!             - and applying nudging tendency

      subroutine spn_fld ( Minx, Maxx, Miny, Maxy, Nil, Njl, &
                           Minz, Maxz, Nk, Nkl, Gni, Gnj,    &
                           Minij, Maxij, L_nij, L_nij0,      &
                           F_npex1, F_npey1,Fld_S )
      use spn_work_mod
      implicit none
#include <arch_specific.hf>

      integer  Minx, Maxx, Miny, Maxy, Nil, Njl
      integer  Minz, Maxz, Nk, Nkl, Gni, Gnj
      integer  Minij, Maxij, L_nij, L_nij0
      integer  F_npex1, F_npey1
      character (len=1) Fld_S

!author
!     Minwei Qian (CCRD) & Bernard Dugas, Syed Husain  (MRB)  - summer 2015
!
!revision
! v4_80 - Qian, Dugas, Hussain            - initial version
! 
!arguments
!  Name        I/O                 Description
!----------------------------------------------------------------
! Minx         I    - minimum index on X (ldnh_nj)
! Maxx         I    - maximum index on X (ldnh_maxx)
! Miny         I    - minimum index on Y (ldnh_miny)
! Maxy         I    - maximum index on Y (ldnh_maxy)
! Nil          I    - number of points on local PEy for I (ldnh_ni)
! Njl          I    - number of points on local PEy for J (ldnh_nj)
! Minz         I    - minimum index on local PEx for K (trp_12smin)
! Maxz         I    - maximum index on local PEx for K (trp_12smax)
! Nk           I    - G_nk-1 points in Z direction globally
! Nkl          I    - number of points on local PEx for K (trp_12sn)
! Gni          I    - number of points in X direction globally (G_ni)
! Gnj          I    - number of points in Y direction globally (G_nj)
! Minij        I    - minimum index on local PEy for I (trp_22min)
! Maxij        I    - maximum index on local PEy for I (trp_22max)
! L_nij        I    - number of points on local PEy for I (trp_22n)
! L_nij0       I    - global offset of the first I element on PEy
! F_npex1      I    - number of processors in X
! F_npey1      I    - number of processors in Y
! Fld_S        I    - name of variable to treat (either of 't','u','v')

#include "gmm.hf"
#include "ptopo.cdk"
#include "glb_ld.cdk"
#include "glb_pil.cdk"
#include "vt1.cdk"
#include "nest.cdk"
#include "lctl.cdk"
#include "dcst.cdk"
#include "cstv.cdk"
#include "step.cdk"
#include "lam.cdk"
#include "spn.cdk"

      external ffft8, rpn_comm_transpose

      real(8) fdwfft(Miny:Maxy,Minz :Maxz ,Gni+2+F_npex1)
      real(8)   fdg2(Minz:Maxz,Minij:Maxij,Gnj+2+F_npey1)
      real*8  pri

      integer  err(3),key(2),nvar
      integer gmmstat
      type(gmm_metadata):: metadata
      real, dimension(:,:,:), pointer :: fld3d=>null(), fld_nest3d=>null()
      integer i,  j, k
      integer ii,jj,kk
      real fld_r(Minx:Maxx,Miny:Maxy)
      real fld_rG(G_ni,G_nj)

      integer no_steps, tmdt
      real spn_wt
!
!----------------------------------------------------------------------
!
      tmdt    = int(Cstv_dt_8)
      no_steps= Step_nesdt/tmdt
      spn_wt  = sqrt((cos(Dcst_pi_8*(float(Lctl_step)/float(no_steps))))**2)**Spn_wt_pwr

      if (Spn_weight_L) spn_wt=1.0

      if (Fld_S.eq.'t') then

         gmmstat = gmm_getmeta (gmmk_tt1_s, metadata)
         gmmstat = gmm_get (gmmk_tt1_s, fld3d, metadata)
         gmmstat = gmm_get (gmmk_nest_t_s, fld_nest3d, metadata)

         Ldiff3D (Minx:Maxx,Miny:Maxy,1:Nk)= &
              fld_nest3d(Minx:Maxx,Miny:Maxy,1:Nk) - &
                   fld3d(Minx:Maxx,Miny:Maxy,1:Nk)
      endif

      if (Fld_S.eq.'u') then

         gmmstat = gmm_getmeta (gmmk_ut1_s, metadata)
         gmmstat = gmm_get (gmmk_ut1_s, fld3d, metadata)
         gmmstat = gmm_get (gmmk_nest_u_s, fld_nest3d, metadata)

         Ldiff3D (Minx:Maxx,Miny:Maxy,1:Nk)= &
              fld_nest3d(Minx:Maxx,Miny:Maxy,1:Nk) - &
                   fld3d(Minx:Maxx,Miny:Maxy,1:Nk)

      endif


      if (Fld_S.eq.'v') then

         gmmstat = gmm_getmeta (gmmk_vt1_s, metadata)
         gmmstat = gmm_get (gmmk_vt1_s, fld3d, metadata)
         gmmstat = gmm_get (gmmk_nest_v_s, fld_nest3d, metadata)

         Ldiff3D (Minx:Maxx,Miny:Maxy,1:Nk)= &
              fld_nest3d(Minx:Maxx,Miny:Maxy,1:Nk) - &
                   fld3d(Minx:Maxx,Miny:Maxy,1:Nk)

      endif


! do transpose from (i,j,k) to (j,k,i)
      call rpn_comm_transpose                         &
           ( Ldiff3D, Minx, Maxx, Gni, (Maxy-Miny+1), &
                      Minz, Maxz, Nk,  fdwfft, 1, 2 )

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! projection ( wfft = x transposed * g )
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!$omp parallel
!$omp do
      do i= 1,Gni               ! trimming in X
         do k= Minz, nkl
            do j= njl+1-pil_n,Maxy
               fdwfft(j,k,i)=0.
            enddo
         enddo
         do k= Minz, nkl
            do j= Miny, pil_s
               fdwfft(j,k,i)=0.
            enddo
         enddo
         do k= Nkl+1,Maxz
            do j= Miny,Maxy
               fdwfft(j,k,i)=0.
            enddo
         enddo

         do k= Minz, 0
            do j= Miny,Maxy
               fdwfft(j,k,i)=0.
            enddo
         enddo
      enddo
!$omp enddo


!$omp single
      call itf_fft_set( Gni-Lam_pil_w-Lam_pil_e,'QCOS',pri )
!$omp end single

!$omp do
      do k=1,Nkl                ! do forward fft in X direction
         call itf_fft_qcos( fdwfft(1+pil_s,k,1+Lam_pil_w), &
         (Maxy-Miny+1)*(Maxz-Minz+1),1,       &
         (Maxy-Miny+1-pil_s-pil_n), -1 )
      enddo
!$omp enddo

!$omp single
! do transpose from (j,k,i) to (k,i,j)
      call rpn_comm_transpose                          &
      ( fdwfft, Miny,  Maxy,  Gnj, (Maxz-Minz+1), &
      Minij, Maxij, Gni, fdg2, 2, 2 )
      call itf_fft_set( Gnj-Lam_pil_s-Lam_pil_n,'QCOS',pri )
!$omp end single

!$omp do
      do k=1,L_nij              ! do forward fft in Y direction
         call itf_fft_qcos( fdg2(1,k,1+Lam_pil_s),     &
         (Maxz-Minz+1)*(Maxij-Minij+1),1, &
         (Maxz-Minz+1), -1 )
      enddo
!$omp enddo

!$omp do
      do jj=1,G_nj+2            ! do filter in X-Y direction
         do ii=1,L_nij
            do kk=Minz,Maxz
               fdg2(kk,ii,jj)=fdg2(kk,ii,jj)*fxy(ii+L_nij0,jj)
            enddo
         enddo
      enddo
!$omp enddo

!$omp do
      do k=1,L_nij              ! do backward fft in Y direction
         call itf_fft_qcos( fdg2(1,k,1+Lam_pil_s),     &
         (Maxz-Minz+1)*(Maxij-Minij+1),1, &
         (Maxz-Minz+1), +1 )
      enddo
!$omp enddo

!$omp single
! do backward transpose from (k,i,j) to (j,k,i)
      call rpn_comm_transpose                          &
      ( fdwfft, Miny,  Maxy,  Gnj, (Maxz-Minz+1), &
      Minij, Maxij, Gni, fdg2, -2, 2 )
      call itf_fft_set( Gni-Lam_pil_w-Lam_pil_e,'QCOS',pri )
!$omp end single

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! inverse projection ( r = x * w )
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!$omp do
      do k=1, Nkl               ! do backward fft in X direction
         call itf_fft_qcos( fdwfft(1+pil_s,k,1+Lam_pil_w), &
         (Maxy-Miny+1)*(Maxz-Minz+1),1,       &
         (Maxy-Miny+1-pil_s-pil_n), +1 )
      enddo
!$omp enddo
!$omp end parallel

! do backward transpose from (j,k,i) to (i,j,k)
      call rpn_comm_transpose                         &
           ( Ldiff3D, Minx, Maxx, Gni, (Maxy-Miny+1), &
                     Minz, Maxz, Nk,  fdwfft, -1, 2 )

      do kk=2,Nk
         fld3d(1:l_ni,1:l_nj,kk) = &
         fld3d(1:l_ni,1:l_nj,kk) + &
         prof(kk)*SNGL(Ldiff3D(1:l_ni,1:l_nj,kk))*spn_wt
      enddo
!     
!----------------------------------------------------------------------
!
      return
      end subroutine spn_fld

