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

!**s/r hzd_abc -- Prepares matrices aix,bix,cix,dix,aiy,biy,ciy
!
      subroutine hzd_abc (F_aix_8, F_bix_8, F_cix_8, F_dix_8 , &
                          F_aiy_8, F_biy_8, F_ciy_8, F_coef_8, &
                          F_cy2_8, F_xp0_8, F_xp2_8, F_yp0_8 , &
                          F_yp2_8, Gni, Gnj, lnjs_nh, lnj,  &
                          ni22s, ni22, Miny,Maxy, F_gnjv, lnjv)
!
      implicit none
#include <arch_specific.hf>
!
      integer Gni,Gnj,lnjs_nh,lnj,ni22s,ni22,Miny,Maxy, &
              F_gnjv, lnjv
      real*8 F_aix_8(*), F_bix_8(*), F_cix_8(*), F_dix_8(*),  &
             F_aiy_8(*), F_biy_8(*), F_ciy_8(*), F_coef_8, &
             F_cy2_8(Miny:Maxy) , F_xp0_8(Gni,3), F_xp2_8(Gni,3),  &
             F_yp0_8(Gnj,3), F_yp2_8(Gnj,3)
!
!author    
!     J.P. Toviessi
!
!revision
! v2_00 - Desgagne M.       - initial MPI version 
! v2_10 - J.P. Toviessi     - reset V grid operator 
! v2_10                       latitudinal modulation of hor. diffusion
! v2_11 - Desgagne M.       - remove vertical modulation
! v3_00 - Desgagne & Lee    - replace geomd.cdk by grd.cdk
! v3_10 - Corbeil & Desgagne & Lee - AIXport+Opti+OpenMP
!
!object
!       
!arguments
!  Name        I/O                 Description
!----------------------------------------------------------------
!  F_aix_8
!----------------------------------------------------------------
! 

#include "glb_ld.cdk"
#include "hzd.cdk"
#include "grd.cdk"
#include "trp.cdk"
!     
      integer i, j, k, ii, jj
      real*8 ax_8(lnjs_nh,G_ni), bx_8(lnjs_nh,G_ni), cx_8(lnjs_nh,G_ni), &
             ay_8(  ni22s,G_nj), by_8(  ni22s,G_nj), cy_8(  ni22s,G_nj), &
             difvx_8(G_ni,l_nj), difvy_8(ni22,G_nj), diy_8
      real*8 ZERO_8,ONE_8,HALF_8
      parameter ( ZERO_8 = 0.0 , ONE_8 = 1.0 , HALF_8 = 0.5 )
!*
!     ---------------------------------------------------------------
!
!     Calcul le long de X
!
!$omp parallel private(jj,ii)
      if (Hzd_difva_L) then
         if (Grd_uniform_L) then
!$omp do
            do j = 1, l_nj
            do i = 1, G_ni
               difvx_8 (i,j) = 0.0
            end do
            end do
!$omp enddo
            if (l_south) then
!$omp do
               do i = 1, G_ni
                  difvx_8 (i,1) =  ONE_8
                  difvx_8 (i,2) = HALF_8
               end do
!$omp enddo
            endif
            if (l_north) then
!$omp do
               do i = 1, G_ni
                  difvx_8 (i,lnjv)   =  ONE_8
                  difvx_8 (i,lnjv-1) = HALF_8
               end do
!$omp enddo
            end if
         else
!$omp do
            do j = 1, l_nj
            jj = j + l_j0 - 1
            do i = 1, G_ni
               difvx_8 (i,j) =  &
               ( (G_xg_8(i+1)-G_xg_8(i-1))*(G_yg_8(jj+1)-G_yg_8(jj-1))/ &
               ( (G_xg_8(G_ni/2+2)-G_xg_8(G_ni/2)) &
                *(G_yg_8(G_nj/2+2)-G_yg_8(G_nj/2)) ) ) **HALF_8
            end do
            end do
!$omp enddo
         endif
      else
!$omp do
         do j = 1, l_nj
         do i = 1, G_ni
            difvx_8 (i,j) = 1.0
         end do
         end do
!$omp enddo
      endif
!
!$omp do
      do j = 1, lnjv
         do i = 1, G_ni
            ax_8(j,i) = F_xp0_8(i,1) -  difvx_8(i,j)* &
                        F_xp2_8(i,1)*F_coef_8/F_cy2_8(j)
            bx_8(j,i) = F_xp0_8(i,2) -  difvx_8(i,j)* &
                        F_xp2_8(i,2)*F_coef_8/F_cy2_8(j)
            cx_8(j,i) = F_xp0_8(i,3) -  difvx_8(i,j)* &
                        F_xp2_8(i,3)*F_coef_8/F_cy2_8(j)
            enddo
      enddo
!$omp enddo
!
!$omp do
      do j = lnjv+1, lnjs_nh
      do i = 1, G_ni
         bx_8(j,i)=  ONE_8
         cx_8(j,i)= ZERO_8
         ax_8(j,i)= ZERO_8
      enddo
      enddo
!$omp enddo

!
!$omp single
      call set_trig21 (F_aix_8,F_bix_8,F_cix_8,F_dix_8, ax_8,bx_8,cx_8,  &
                       lnjs_nh, 1, G_ni, lnjs_nh, .true.)
!$omp end single
!
!     Calcul le long de Y
!
      if (Hzd_difva_L) then
         if (Grd_uniform_L) then
!$omp do
            do i = 1, ni22
            do j = 3, G_nj
               difvy_8 (i,j) = 0.0
            end do
            end do
!$omp enddo
!$omp do
            do i = 1, ni22
               difvy_8 (i,1) =  ONE_8
               difvy_8 (i,2) = HALF_8
               difvy_8 (i,F_gnjv)   =  ONE_8
               difvy_8 (i,F_gnjv-1) = HALF_8
            end do
!$omp enddo
         else
!$omp do
            do i = 1, ni22
            ii = i + trp_22n0 - 1
            do j = 1, G_nj
               difvy_8 (i,j) =  &
               ( (G_xg_8(ii+1)-G_xg_8(ii-1))*(G_yg_8(j+1)-G_yg_8(j-1))/ &
               ( (G_xg_8(G_ni/2+2)-G_xg_8(G_ni/2)) &
                *(G_yg_8(G_nj/2+2)-G_yg_8(G_nj/2)) ) ) **HALF_8
            end do
            end do
!$omp enddo
         endif
      else
!$omp do
         do i = 1, ni22
         do j = 1, G_nj
            difvy_8 (i,j) = 1.0
         end do
         end do
!$omp enddo
      endif
!
!$omp do
      do i= 1, ni22
         do j= 1, F_gnjv
            ay_8(i,j) = F_yp0_8(j,1) - difvy_8(i,j)* &
                        F_yp2_8(j,1) * F_coef_8 
            by_8(i,j) = F_yp0_8(j,2) - difvy_8(i,j)* &
                        F_yp2_8(j,2) * F_coef_8 
            cy_8(i,j) = F_yp0_8(j,3) - difvy_8(i,j)* &
                        F_yp2_8(j,3) * F_coef_8 
         enddo
         ay_8(i,1) = ZERO_8
      enddo
!$omp enddo
!
!$omp do
      do i = ni22+1, ni22s
      do j = 1, F_gnjv
         by_8(i,j)=  ONE_8
         cy_8(i,j)= ZERO_8
         ay_8(i,j)= ZERO_8
      enddo
      enddo
!$omp enddo
!$omp end parallel
!
      call set_trig21 (F_aiy_8,F_biy_8,F_ciy_8,diy_8, ay_8,by_8,cy_8,  &
                       ni22s, 1, F_gnjv, ni22s, .false.)
!
!     ---------------------------------------------------------------
!
      return
      end

