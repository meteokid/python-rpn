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

!**S/R hzd_areav - AREAL AVERAGE on a given area
!
!
      subroutine hzd_areav(F_wk1,F_xp0_8,F_yp0_8,Minx,Maxx,Miny,Maxy,Gni,Gnj,Nk)
!
      implicit none
#include <arch_specific.hf>
!
!
      integer  Minx,Maxx,Miny,Maxy
      integer  Gni,Gnj,Nk
      real     F_wk1(Minx:Maxx,Miny:Maxy,Nk)
      real*8   F_xp0_8(Gni*3),F_yp0_8(Gnj*3)
!
!author
!      Abdessamad Qaddouri - May 2000
!
!arguments
! 

!
      real*8 zero
      parameter( zero = 0.0 )

      integer i, j, k
      real*8 avg(Nk),area1,area2,area
      real*8 ssq,var(Nk)
      real*8, dimension(:,:,:), allocatable :: xdphi,wkphi
!*
!     __________________________________________________________________
!
      allocate( xdphi(Gni,Gnj,Nk), wkphi(Gni,Gnj,Nk) )

      area1=0.0

      do i=1,Gni
         area1 = area1+F_xp0_8(Gni+i)
      enddo
      area2=0.0
      do j=1,Gnj-1
         area2 = area2+F_yp0_8(Gnj+j)
      enddo
      area = area1*area2
      print*,'hzd_areav: area=', area,'Gni=',Gni,'Gnj=',Gnj-1,'Nk=',Nk

      do k = 1,Nk
        do j=1,Gnj-1
         do i=1,Gni
            xdphi(i,j,k)= F_wk1(i,j,k)
!           print *,'xdphi(',i,',',j,',',k,')=',xdphi(i,j,k)
            wkphi(i,j,k)= F_xp0_8(Gni+i)*xdphi(i,j,k)*F_yp0_8(Gnj+j)
         enddo
        enddo
      enddo
        
      do k=1,Nk
         avg(k) = 0.0
        do j=1,Gnj-1
         do i=1,Gni
            avg(k)= avg(k)+wkphi(i,j,k)
         enddo
        enddo
        avg(k)=avg(k)/area
      enddo
!
      do k = 1,Nk
        do j=1,Gnj-1
         do i=1,Gni
            wkphi(i,j,k)= F_xp0_8(Gni+i)*xdphi(i,j,k)* &
                                xdphi(i,j,k)*F_yp0_8(Gnj+j)
        enddo
       enddo
      enddo

       do k=1,Nk
          ssq = 0.0
          do j=1,Gnj-1
             do i=1,Gni
                ssq= ssq+wkphi(i,j,k)
             enddo
          enddo
          ssq=ssq/area
          var(k)=sqrt((ssq-avg(k)*avg(k)))
       enddo

!      print *,'XPO,YP0 operators, GNI,GNJ=',Gni,Gnj
!      do i=1,Gni*3
!      print *,'F_xp0_8(',i,')=',F_xp0_8(i)
!      enddo
!      do j=1,Gnj*3
!      print *,'F_yp0_8(',j,')=',F_yp0_8(j)
!      enddo
       do k=1,nk
          print *,'k=',k,'avg=',avg(k),'var=',var(k)
       enddo

       deallocate(xdphi,wkphi)
!     __________________________________________________________________
      return
      end
