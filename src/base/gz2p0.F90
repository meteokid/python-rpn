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

!**s/r gz2p02 - Compute P0 from GZ from pressure coordinate
!
      subroutine gz2p02 ( F_ps, F_gz, F_topo, F_rna ,&
                          Mminx,Mmaxx,Mminy,Mmaxy,Nk,&
                          F_i0,F_in,F_j0,F_jn )
      implicit none
#include <arch_specific.hf>

      integer Nk,Mminx,Mmaxx,Mminy,Mmaxy,F_i0,F_in,F_j0,F_jn
      real F_gz  (Mminx:Mmaxx,Mminy:Mmaxy,Nk), &
           F_ps  (Mminx:Mmaxx,Mminy:Mmaxy)   , &
           F_topo(Mminx:Mmaxx,Mminy:Mmaxy)   , F_rna(Nk)

!arguments
!  Name        I/O                 Description
!----------------------------------------------------------------
! F_ps         O    - ln(pi_s/z_s)
! F_gz         I    - geopotential height
! F_topo       I    - topography
! F_rna        I    - pressure levels from pressure analyse
! NN           I    - number of points on the plane
! Nk           I    - number of levels from the pressure analyse

#include "lun.cdk"
#include "dcst.cdk"
#include "glb_ld.cdk"
#include "geomg.cdk"

      integer i,j,k,m,NN
      real, allocatable, dimension(:  ) :: guess,a,topo
      real, allocatable, dimension(:,:) :: zcol,tcol
      real lna(Nk),sdd(Nk),conv,acc
!
!     ---------------------------------------------------------------
!
      acc = .1 * Dcst_grav_8
      conv = alog(100.)
!     Convert millibar to log of pascal unit - Pressure Analysis
      do k=1,Nk
         lna(k) = alog(F_rna(k))
      enddo
      do k=1,Nk-1
         sdd(k) = 1./(lna(k+1)-lna(k))
      enddo

      NN=(F_jn-F_j0+1)*(F_in-F_i0+1)
      allocate(guess(NN),a(NN),topo(NN))
      allocate(zcol(NN,Nk),tcol(NN,Nk))

      m=0
      do j=F_j0,F_jn
      do i=F_i0,F_in
         m=m+1
         topo(m)=F_topo(i,j)
      enddo
      enddo
      do k=1,Nk
         m=0
         do j=F_j0,F_jn
         do i=F_i0,F_in
            m=m+1
            zcol(m,k) = Dcst_grav_8*F_gz(i,j,k)
         enddo
         enddo
      enddo
!
!     Compute derivative of geopotential (from vdfds)
!
      do k=1,Nk-1
      do i=1,NN
         tcol(i,k+1) = sdd(k)*(zcol(i,k+1)-zcol(i,k))
      enddo
      enddo
!
      do i=1,NN
        a(i) = tcol(i,2)
      enddo
!
      do k=2,Nk-1
      do i=1,NN
         tcol(i,k) = (sdd(k)*tcol(i,k+1)+sdd(k-1)*tcol(i,k)) &
                     /(sdd(k)+sdd(k-1))
      enddo
      enddo
!
!     BOUNDARIES
      do i=1,NN
         tcol(i,1) = a(i)
         tcol(i,Nk) = tcol(i,Nk)
      enddo
!     Derived VT can be used here for input if non-existent in the analysis
!     do k=1,Nk
!     do i=1,NN
!        Ind_t(i,k) = -tcol(i,k)/Dcst_rgasd_8
!     enddo
!     enddo
!
!     Compute pressure at the surface (PS)
      do i=1,NN
         guess(i) = lna(Nk)-topo(i)/(Dcst_rgasd_8*250.)
      enddo
!
      call vterp1 (guess,topo,zcol,tcol,lna,acc,NN,Nk)
      m=0
      do j=F_j0,F_jn
      do i=F_i0,F_in
         m=m+1
         F_ps(i,j) = guess(m) + conv
      enddo
      enddo

      deallocate(guess,a,topo,zcol,tcol)
!
!     ---------------------------------------------------------------
!
      return
      end
