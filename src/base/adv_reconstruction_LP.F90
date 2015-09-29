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

!**s/p adv_reconstruction_LP - 1D-Reconstruction based on Laprise and Plante 1995 (PPM1)  

      subroutine adv_reconstruction_LP (F_psi_8,F_xu_8,F_m_8,F_n)

      implicit none
#include <arch_specific.hf>

      integer F_n 

      real*8 F_psi_8(F_n,3),F_xu_8(0:F_n),F_m_8(F_n)

      !@author Monique Tanguay

      !@revisions
      !v4_XX - Tanguay M.        - GEM4 Mass-Conservation

      !arguments
      !---------------------------------------------------------------------
      !F_psi_8(i) = INT(psi)(F_xu(i-1),F_xu(i)) = F_m(i) 
      !---------------------------------------------------------------------

      integer i,shift
      real*8 i_12_8,i_13_8,i_22_8,i_23_8,i_32_8,i_33_8,j_23_8,j_33_8,r_i1_8,r_i2_8,r_i3_8,r_j2_8,r_j3_8,r_k3_8,inv_j2_8,inv_j3_8, &
             L1_8,L2_8,L3_8,R1_8,R2_8,R3_8,d2_8(F_n),d3_8(F_n),i1_8(F_n),x_a_8,x_b_8,x_c_8
      real*8, parameter :: INV_1_8 = 1.d0, INV_2_8 = 0.5d0, INV_3_8 = 1.d0/3.d0

      !---------------------------------------------------------------------

      do i=1,F_n

         L1_8 = F_xu_8(i-1)
         L2_8 = F_xu_8(i-1) * F_xu_8(i-1)
         L3_8 = F_xu_8(i-1) * L2_8

         R1_8 = F_xu_8(i)
         R2_8 = F_xu_8(i) * F_xu_8(i)
         R3_8 = F_xu_8(i) * R2_8

         d2_8(i) = (R2_8 - L2_8)*INV_2_8
         d3_8(i) = (R3_8 - L3_8)*INV_3_8

         i1_8(i) = 1.0d0/((R1_8 - L1_8)*INV_1_8)

      enddo

      !Cubic reconstruction with extrapolation at the boundaries: We resolve a linear system
      !-------------------------------------------------------------------------------------
      do i=1,F_n 

         shift = 0
         if (i==1  ) shift =  1 
         if (i==F_n) shift = -1 

         i_12_8 = d2_8(i+(1)-2+shift)*i1_8(i+(1)-2+shift)
         i_13_8 = d3_8(i+(1)-2+shift)*i1_8(i+(1)-2+shift)
         i_22_8 = d2_8(i+(2)-2+shift)*i1_8(i+(2)-2+shift)
         i_23_8 = d3_8(i+(2)-2+shift)*i1_8(i+(2)-2+shift)
         i_32_8 = d2_8(i+(3)-2+shift)*i1_8(i+(3)-2+shift)
         i_33_8 = d3_8(i+(3)-2+shift)*i1_8(i+(3)-2+shift)

         inv_j2_8 = 1.0d0/(i_22_8-i_12_8)
         inv_j3_8 = 1.0d0/(i_32_8-i_12_8)

         j_23_8 = (i_23_8-i_13_8)*inv_j2_8
         j_33_8 = (i_33_8-i_13_8)*inv_j3_8

         r_i1_8 = F_m_8(i+1-2+shift) * i1_8(i+(1)-2+shift)
         r_i2_8 = F_m_8(i+2-2+shift) * i1_8(i+(2)-2+shift)
         r_i3_8 = F_m_8(i+3-2+shift) * i1_8(i+(3)-2+shift)

         r_j2_8 = (r_i2_8-r_i1_8)*inv_j2_8
         r_j3_8 = (r_i3_8-r_i1_8)*inv_j3_8

         r_k3_8 = (r_j3_8-r_j2_8)/(j_33_8-j_23_8)

         x_c_8 = r_k3_8
         x_b_8 = r_j2_8 - j_23_8 * x_c_8 
         x_a_8 = r_i1_8 - i_12_8 * x_b_8 - i_13_8 * x_c_8 

         !Convertion from parabola in x to parabola in zeta=[x-x(i-1)]/[x(i)-x(i-1)] 
         !--------------------------------------------------------------------------
         F_psi_8(i,1) = x_a_8 + x_b_8*F_xu_8(i-1) + x_c_8*F_xu_8(i-1)**2 
         F_psi_8(i,2) = (F_xu_8(i)-F_xu_8(i-1))*(x_b_8+2.0d0*x_c_8*F_xu_8(i-1)) 
         F_psi_8(i,3) = x_c_8*(F_xu_8(i)-F_xu_8(i-1))**2

      enddo

      return
      end
