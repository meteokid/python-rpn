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
!**s/r grid_area_mask - Evaluate area and mask 

      subroutine grid_area_mask (F_area_8,F_mask_8,Ni,Nj) 
      use grid_options
      implicit none
#include <arch_specific.hf>

      integer,                   intent(in)  :: Ni,Nj   
      real*8 , dimension(Ni,Nj), intent(out) :: F_area_8
      real*8 , dimension(Ni,Nj), intent(out) :: F_mask_8
 
!author
!     Author Qaddouri/Tanguay -- Summer 2014
!
!revision
! v4_70 - Qaddouri/Tanguay     - initial version
! v4_73 - Lee V.  - optimization for MPI

#include "glb_ld.cdk"
#include "glb_pil.cdk"
#include "geomg.cdk"
#include "ptopo.cdk"

      real*8, external :: yyg_weight
      integer i,j,k,np_subd,ierr
      real*8, parameter :: HALF_8 = 0.5
      real*8 poids(l_ni,l_nj), area_4(l_ni,l_nj), &
             dx,dy,x_a_4,y_a_4,sf(2),sp(2)
!
!     ---------------------------------------------------------------
!
      do j = 1,l_nj
         do i = 1,l_ni
            F_area_8(i,j)= ( Geomg_x_8(i+1)-Geomg_x_8(i-1) )   * HALF_8 * &
                           (sin((Geomg_y_8(j+1)+Geomg_y_8(j  ))* HALF_8)- &
                            sin((Geomg_y_8(j  )+Geomg_y_8(j-1))* HALF_8))
         enddo
      enddo

      !---------
      !GU or LAM
      !---------
      if (.not.Grd_yinyang_L) F_mask_8 = 1.0d0

      !-------------
      !Yin-Yang grid 
      !-------------
      if (Grd_yinyang_L) then

         !1) Find out where YIN lat lon points are in (YAN) grid with call to smat.
         !2) If they are not outside of Yin grid, put area to zero for those points.
         !--------------------------------------------------------------------------
         np_subd = 4*(G_ni-Lam_pil_e-Lam_pil_w)

         sp    = 0.d0
         sf    = 0.d0

         do j = 1+pil_s, l_nj-pil_n

            y_a_4 = Geomg_y_8(j)

            do i = 1+pil_w, l_ni-pil_e

               x_a_4 = Geomg_x_8(i)-acos(-1.d0)
               dx    = ( Geomg_x_8(i+1)-Geomg_x_8(i-1) ) * HALF_8
               dy    = (sin((Geomg_y_8(j+1)+Geomg_y_8(j  ))* HALF_8) -  &
                        sin((Geomg_y_8(j  )+Geomg_y_8(j-1))* HALF_8))

               area_4(i,j) = dx*dy
               poids (i,j) = yyg_weight (x_a_4,y_a_4,dx,dy,np_subd)

               !Check if poids <0
               !-----------------
               if (poids(i,j)*(1.d0-poids(i,j)) .gt. 0.d0) then
                   sp(1) = sp(1) + poids(i,j)*area_4(i,j)
               elseif (abs(poids(i,j)-1.d0) .lt. 1.d-14) then
                   sp(2) = sp(2) + poids(i,j)*area_4(i,j)
               endif

            enddo

         enddo

!MPI reduce to get the global value for sp into sf
         call RPN_COMM_allreduce(sp,sf,2,"MPI_DOUBLE_PRECISION","MPI_SUM","GRID",ierr)

         !Correct and scale poids
         !-----------------------
         sp = 0.d0

         do j = 1+pil_s, l_nj-pil_n
         do i = 1+pil_w, l_ni-pil_e

            x_a_4 = poids(i,j)*(2.d0*acos(-1.d0) - sf(2))/sf(1)

            if (poids(i,j)*(1.d0-poids(i,j)) .gt. 0.d0) then
                poids(i,j) = min( 1.0d0, x_a_4 )
            endif
            if (poids(i,j)*(1.0-poids(i,j)) .gt. 0.d0) then
                sp(1) = sp(1) + poids(i,j)*area_4(i,j)
            elseif (abs(poids(i,j)-1.d0) .lt. 1.d-14) then
                sp(2) = sp(2) + poids(i,j)*area_4(i,j)
            endif

         enddo
         enddo

!MPI reduce to get the global value for sp into sf
         call RPN_COMM_allreduce(sp,sf,2,"MPI_DOUBLE_PRECISION","MPI_SUM","GRID",ierr)

         !Correct
         !-------
         do j = 1+pil_s, l_nj-pil_n
         do i = 1+pil_w, l_ni-pil_e
            x_a_4 = poids(i,j)*(2.d0*acos(-1.d0) - sf(2))/sf(1)

            if (poids(i,j)*(1.d0-poids(i,j)) .gt. 0.d0) then
                poids(i,j) = min( 1.d0, x_a_4 )
            endif
 
         enddo
         enddo

         F_mask_8 = 0.d0
         do j=1+pil_s,l_nj-pil_n
            do i = 1+pil_w,l_ni-pil_e
               F_mask_8(i,j) = poids(i,j)
            enddo
         enddo

      endif
!
!     ---------------------------------------------------------------
!
      return
      end
