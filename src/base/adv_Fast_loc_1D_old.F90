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

!**s/p adv_Fast_loc_1D - Localize F_x2 in F_x1 

      subroutine adv_Fast_loc_1D_old (ia,ib,F_x2_8,F_n2,F_x1_8,F_n1)

      implicit none

      integer F_n2,F_n1

      real*8 F_x2_8(0:F_n2),F_x1_8(0:F_n1)
      integer ia(F_n2),ib(F_n2)

      !author Tanguay/Qaddouri
      !
      !revision
      ! v4_80 - Tanguay/Qaddouri - SLICE
      !
      !arguments
      !---------------------------------------------------------------------
      !INPUT :  F_x2, F_x1: boundaries of two sets of cells
      !Output:  ia,ib     : locations of each cell edgesof F_x2 in F_x1
      !---------------------------------------------------------------------

      integer i,i0,pnx
      real*8 x00_8,hxmn_8,whx_8(F_n1),ovdx_8,rri_8,fi_8,bsx_8(F_n1+1)
      real*8, parameter :: LARGE_8 = 1.D20

      integer,pointer, dimension(:) :: lcx

      !---------------------------------------------------------------------

      !Prepare localization of F2 in F1
      !--------------------------------
      x00_8 = F_x1_8(0)

      hxmn_8 = LARGE_8

      do i = 1,F_n1
         whx_8(i) = F_x1_8(i) - F_x1_8(i-1)
         hxmn_8 = min(whx_8(i), hxmn_8)
      enddo

      ovdx_8 = 1.0d0/hxmn_8

      pnx = int (1.0+(F_x1_8(F_n1)-x00_8) * ovdx_8)

      allocate (lcx(pnx))

      i0 = 1
      do i=1,pnx
         fi_8 = F_x1_8(0) + (i-1) * hxmn_8
         if (fi_8 > F_x1_8(i0)) i0 = min((F_n1+1)-1,i0+1)
         lcx(i) = i0
      enddo

      do i = 1,F_n1+1
         bsx_8(i) = F_x1_8(i-1)
      enddo

      do i = 1,F_n2

         !Find indice such as x_1(ia(i)-1) .le. x_2(i-1) .le. x_1(ia(i))
         !--------------------------------------------------------------
         rri_8 = F_x2_8(i-1)
         ia(i) = (rri_8 - x00_8) * ovdx_8
         ia(i) = lcx(ia(i)+1) + 1
         if (rri_8 < bsx_8(ia(i))) ia(i) = ia(i) - 1
         ia(i) = max(1,min(ia(i),F_n1))

         !Find indice such as x_1(ib(i)-1) .le. x_2(i) .le. x_1(ib(i))
         !------------------------------------------------------------
         rri_8 = F_x2_8(i)
         ib(i) = (rri_8 - x00_8) * ovdx_8
         ib(i) = lcx(ib(i)+1) + 1
         if (rri_8 < bsx_8(ib(i))) ib(i) = ib(i) - 1
         ib(i) = max(1,min(ib(i),F_n1))

         if (.NOT.(F_x1_8(ia(i)-1)<=F_x2_8(i-1).and.F_x2_8(i-1)<=F_x1_8(ia(i)))) then
            print *,'EN KO X2(I-1)=',F_x1_8(ia(i)-1),F_x2_8(i-1),F_x1_8(ia(i))
            call flush(6)
            STOP
         endif
         if (.NOT.(F_x1_8(ib(i)-1)<=F_x2_8(i).and.F_x2_8(i)<=F_x1_8(ib(i)))) then
            print *,'EN KO X2(I)=',F_x1_8(ib(i)-1),F_x2_8(i),F_x1_8(ib(i))
            call flush(6)
            STOP
         endif

      enddo

      deallocate (lcx)

      return
      end
