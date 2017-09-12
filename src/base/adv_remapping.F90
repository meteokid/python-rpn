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

      use tracers
      implicit none
#include <arch_specific.hf>

      integer F_n2,F_n1

      real*8 F_m2_8(F_n2),F_x2_8(0:F_n2),F_m1_8(F_n1),F_x1_8(0:F_n1)

      !@author Monique Tanguay

      !@revisions
      !v4_80 - Tanguay M.        - GEM4 Mass-Conservation
      !v4_80 - Qaddouri A.       - Fast_loc_1D/PPM2 Z4 Z41 Z412

      !arguments
      !---------------------------------------------------------------------
      !INPUT : F_m_1(i) = mass(F_x_1(i-1),F_x_1(i))
      !OUTPUT: F_m_2(i) = mass(F_x_2(i-1),F_x_2(i)) = INT(psi)(F_x_2(i-1),F_x_2(i)) with INT(psi)(i-1,i) = F_m_1(i)
      !---------------------------------------------------------------------

      !---------------------------------------------------------------------

      integer i,zz,ia,ib,iaa(F_n2),ibb(F_n2)
      real*8 psi_abc_8(F_n1,4),rbr_8, &
             zet_8,zet1_8,zet2_8
      real*8, parameter :: INV_1_8 = 1.d0, INV_2_8 = 0.5d0, INV_3_8 = 1.d0/3.d0, INV_4_8 = 1.d0/4.d0

      !---------------------------------------------------------------------

      !Zeroing psi_abc_8
      !-----------------
      psi_abc_8 = 0.0d0

      !Prepare localization of F2 in F1
      !--------------------------------
      call adv_Fast_loc_1D_old (iaa,ibb,F_x2_8,F_n2,F_x1_8,F_n1)

      !Build psi such that INT(psi)(i,i+1) = F_m_1(i)
      !----------------------------------------------

      if (Tr_SLICE_rebuild==1) call adv_rebuild_LP  (psi_abc_8,F_x1_8,F_m1_8,F_n1)
      if (Tr_SLICE_rebuild==2) call adv_rebuild_CW  (psi_abc_8,F_x1_8,F_m1_8,F_n1,Tr_SLICE_mono)
      if (Tr_SLICE_rebuild==3) call adv_rebuild     (psi_abc_8,F_x1_8,F_m1_8,F_n1,Tr_SLICE_rebuild)
      if (Tr_SLICE_rebuild==4) call adv_rebuild     (psi_abc_8,F_x1_8,F_m1_8,F_n1,Tr_SLICE_rebuild)

      do i = 1,F_n2

         ia = iaa(i)
         ib = ibb(i)

         rbr_8 = 0.0d0

         if (ia/=ib) then

            !Residual at Left on (ia-1,ia)
            !-----------------------------
            zet_8 = (F_x2_8(i-1) - F_x1_8(ia-1))/(F_x1_8(ia) - F_x1_8(ia-1))

            rbr_8 = (psi_abc_8(ia,1) * (1.0d0**1  - zet_8**1 )*INV_1_8 + &
                     psi_abc_8(ia,2) * (1.0d0**2  - zet_8**2 )*INV_2_8 + &
                     psi_abc_8(ia,3) * (1.0d0**3  - zet_8**3 )*INV_3_8 + &
                     psi_abc_8(ia,4) * (1.0d0**4  - zet_8**4 )*INV_4_8) * (F_x1_8(ia) - F_x1_8(ia-1)) + rbr_8

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
                     psi_abc_8(ib,3) * (zet_8**3 -  0.0d0**3 )*INV_3_8+ &
                     psi_abc_8(ib,4) * (zet_8**4 -  0.0d0**4 )*INV_4_8) * (F_x1_8(ib) - F_x1_8(ib-1)) + rbr_8

         else

            zet1_8 = (F_x2_8(i-1) - F_x1_8(ia-1))/(F_x1_8(ia) - F_x1_8(ia-1))
            zet2_8 = (F_x2_8(i  ) - F_x1_8(ia-1))/(F_x1_8(ia) - F_x1_8(ia-1))

            rbr_8 = (psi_abc_8(ia,1) * (zet2_8**1 - zet1_8**1)*INV_1_8 + &
                     psi_abc_8(ia,2) * (zet2_8**2 - zet1_8**2)*INV_2_8 + &
                     psi_abc_8(ia,3) * (zet2_8**3 - zet1_8**3)*INV_3_8+ &
                     psi_abc_8(ia,4) * (zet2_8**4 - zet1_8**4)*INV_4_8) * (F_x1_8(ia) - F_x1_8(ia-1)) + rbr_8

         endif

         F_m2_8(i) = rbr_8

      enddo

      return
      end
