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

!**s/p adv_remapping - 1D-Remapping based on Zerroukat, 2012, SLICE-3D  

      subroutine adv_remapping (F_m2_8,F_x2_8,F_n2,F_m1_8,F_x1_8,F_n1)

      implicit none
#include <arch_specific.hf>

      integer F_n2,F_n1  

      real*8 F_m2_8(F_n2),F_x2_8(0:F_n2),F_m1_8(F_n1),F_x1_8(0:F_n1)

      !@author Monique Tanguay

      !@revisions
      !v4_XX - Tanguay M.        - GEM4 Mass-Conservation

      !arguments
      !---------------------------------------------------------------------
      !INPUT : F_m_1(i) = mass(F_x_1(i-1),F_x_1(i))
      !OUTPUT: F_m_2(i) = mass(F_x_2(i-1),F_x_2(i)) = INT(psi)(F_x_2(i-1),F_x_2(i)) with INT(psi)(i-1,i) = F_m_1(i)
      !---------------------------------------------------------------------
#include "adv_nml.cdk"

      !---------------------------------------------------------------------

      integer i,zz,ia,ib,n1,n2,ii,i0,pnx
      real*8 psi_abc_8(F_n1,3),rbr_8,x00_8,hxmn_8,whx_8(F_n1),ovdx_8,rri_8,fi_8,bsx_8(F_n1+1), &
             zet_8,zet1_8,zet2_8
      real*8, parameter :: LARGE_8 = 1.D20, INV_1_8 = 1.d0, INV_2_8 = 0.5d0, INV_3_8 = 1.d0/3.d0

      integer,pointer, dimension(:) :: lcx

      real*8, parameter :: EPS_8 = 1.0D-06

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

      !Build psi such that INT(psi)(i,i+1) = F_m_1(i) 
      !----------------------------------------------

      if (Adw_reconstruction==1) call adv_reconstruction_LP (psi_abc_8,F_x1_8,F_m1_8,F_n1) 

      if (Adw_reconstruction==2) call adv_reconstruction_CW (psi_abc_8,F_x1_8,F_m1_8,F_n1,Adw_PPM_mono) 

      do i = 1,F_n2 

         !Find indice such as x_1(ia-1) .le. x_2(i-1) .le. x_1(ia) 
         !--------------------------------------------------------
         rri_8 = F_x2_8(i-1)
         ia = (rri_8 - x00_8) * ovdx_8
         ia = lcx(ia+1) + 1
         if (rri_8 < bsx_8(ia)) ia = ia - 1
         ia = max(1,min(ia,F_n1))

         !Find indice such as x_1(ib-1) .le. x_2(i) .le. x_1(ib)
         !------------------------------------------------------
         rri_8 = F_x2_8(i)
         ib = (rri_8 - x00_8) * ovdx_8
         ib = lcx(ib+1) + 1
         if (rri_8 < bsx_8(ib)) ib = ib - 1
         ib = max(1,min(ib,F_n1))

         if (.NOT.(F_x1_8(ia-1)<=F_x2_8(i-1).and.F_x2_8(i-1)<=F_x1_8(ia))) then 
            print *,'EN KO X2(I-1)=',F_x1_8(ia-1),F_x2_8(i-1),F_x1_8(ia) 
            call flush(6)
            STOP 
         endif
         if (.NOT.(F_x1_8(ib-1)<=F_x2_8(i).and.F_x2_8(i)<=F_x1_8(ib))) then 
            print *,'EN KO X2(I)=',F_x1_8(ib-1),F_x2_8(i),F_x1_8(ib) 
            call flush(6)
            STOP 
         endif

         rbr_8 = 0.0d0

         if (ia/=ib) then 

            !Residual at Left on (ia-1,ia)
            !-----------------------------
            zet_8 = (F_x2_8(i-1) - F_x1_8(ia-1))/(F_x1_8(ia) - F_x1_8(ia-1))

            rbr_8 = (psi_abc_8(ia,1) * (1.0d0**1  - zet_8**1 )*INV_1_8 + &
                     psi_abc_8(ia,2) * (1.0d0**2  - zet_8**2 )*INV_2_8 + &
                     psi_abc_8(ia,3) * (1.0d0**3  - zet_8**3 )*INV_3_8) * (F_x1_8(ia) - F_x1_8(ia-1)) + rbr_8

            !Central on (ia,ib-1)
            !--------------------
            do zz = ia+1,ib-1

               rbr_8 = F_m1_8(zz) + rbr_8

            enddo

            !Residual at Right on (ib-1,ib)
            !------------------------------
            zet_8 = (F_x2_8(i) - F_x1_8(ib-1))/(F_x1_8(ib) - F_x1_8(ib-1))

            rbr_8 = (psi_abc_8(ib,1) * (zet_8**1 -  0.0d0**1 )*INV_1_8 + &
                     psi_abc_8(ib,2) * (zet_8**2 -  0.0d0**2 )*INV_2_8 + &
                     psi_abc_8(ib,3) * (zet_8**3 -  0.0d0**3 )*INV_3_8) * (F_x1_8(ib) - F_x1_8(ib-1)) + rbr_8

         else

            zet1_8 = (F_x2_8(i-1) - F_x1_8(ia-1))/(F_x1_8(ia) - F_x1_8(ia-1))
            zet2_8 = (F_x2_8(i  ) - F_x1_8(ia-1))/(F_x1_8(ia) - F_x1_8(ia-1))

            rbr_8 = (psi_abc_8(ia,1) * (zet2_8**1 - zet1_8**1)*INV_1_8 + &
                     psi_abc_8(ia,2) * (zet2_8**2 - zet1_8**2)*INV_2_8 + &
                     psi_abc_8(ia,3) * (zet2_8**3 - zet1_8**3)*INV_3_8) * (F_x1_8(ia) - F_x1_8(ia-1)) + rbr_8

         endif 

         F_m2_8(i) = rbr_8

      enddo

      deallocate (lcx)
      
      return
      end
