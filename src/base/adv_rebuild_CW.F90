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

!**s/p adv_rebuild_CW - 1D-Rebuild based on Piecewise Parabolic Method (Colella and Woodward, 1984)  

      subroutine adv_rebuild_CW (F_psi_8,F_xu_8,F_m_8,F_n,F_mono)

      implicit none
#include <arch_specific.hf>

      integer F_n,F_mono 

      real*8 F_psi_8(F_n,3),F_xu_8(0:F_n),F_m_8(F_n)

      !@author Monique Tanguay

      !@revisions
      !v4_80 - Tanguay M.        - GEM4 Mass-Conservation

      !arguments
      !---------------------------------------------------------------------
      !F_psi_8  = A,B,C of rho(x) = A + Bx + C x**2  
      !
      !F_m_8(i) = INT(rho) from (xu_i to xu_i+1)
      !---------------------------------------------------------------------

      integer i,i1,i2,shift

      real*8 f1_8,f2_8,f3_8,dd_8,r1_8,r2_8,p1_8,pa_8,pb_8,p2_8,alpha_8,der0_8,der1_8,der0_rev_8,der1_rev_8, &
             rhoL_8(F_n),rhoR_8(F_n),rhoA_8(F_n),Xrho_8(F_n),Dx_8(F_n) 

      real*8 i_12_8,i_13_8,i_22_8,i_23_8,i_32_8,i_33_8,j_23_8,j_33_8,r_i1_8,r_i2_8,r_i3_8,r_j2_8,r_j3_8,r_k3_8,inv_j2_8,inv_j3_8, &
             L1_8,L2_8,L3_8,R3_8,d2_8(F_n),d3_8(F_n),i1_8(F_n),x_a_8,x_b_8,x_c_8

      real*8, parameter :: ONE_8=1.d0, INV_1_8 = 1.d0, INV_2_8 = 0.5d0, INV_3_8 = 1.d0/3.d0

      !---------------------------------------------------------------------

      i1 = 3
      i2 = F_n-2 

      do i=1,F_n

         Dx_8  (i) = F_xu_8(i) - F_xu_8(i-1)

         rhoA_8(i) = F_m_8(i)/Dx_8(i)

      enddo
      
      do i=i1-1,i2+1

         f1_8 = Dx_8(i)/(Dx_8(i-1)+Dx_8(i)+Dx_8(i+1)) 

         f2_8 = (2.0d0*Dx_8(i-1)+Dx_8(i))/(Dx_8(i+1)+Dx_8(i))

         f3_8 = (Dx_8(i)+2.0d0*Dx_8(i+1))/(Dx_8(i-1)+Dx_8(i)) 

         Xrho_8(i) = f1_8*( f2_8*(rhoA_8(i+1)-rhoA_8(i)) + f3_8*(rhoA_8(i)-rhoA_8(i-1)) ) 

      enddo

      do i=i1-1,i2

         f1_8 = Dx_8(i)/(Dx_8(i)+Dx_8(i+1)) 

         f2_8 = (2.0d0*Dx_8(i+1)*Dx_8(i))/(Dx_8(i)+Dx_8(i+1))

         dd_8 = Dx_8(i-1)+Dx_8(i)+Dx_8(i+1)+Dx_8(i+2) 

         r1_8 = (Dx_8(i-1)+Dx_8(i))/(2.0d0*Dx_8(i)+Dx_8(i+1))

         r2_8 = (Dx_8(i+2)+Dx_8(i+1))/(2.0d0*Dx_8(i+1)+Dx_8(i))

         p1_8 = f1_8*(rhoA_8(i+1)-rhoA_8(i))

         pa_8 = f2_8*(r1_8 - r2_8)*(rhoA_8(i+1)-rhoA_8(i))  

         pb_8 = -Dx_8(i)*r1_8*Xrho_8(i+1)+Dx_8(i+1)*r2_8*Xrho_8(i)    

         p2_8 = (pa_8 + pb_8)/dd_8   

         rhoR_8(i) = rhoA_8(i) + p1_8 + p2_8

         if (F_mono==4.and.rhoR_8(i)<0.0d0) rhoR_8(i) = 0.0d0

         !------------------------------------------------------------------

      enddo

      do i=i1,i2

         rhoL_8(i) = rhoR_8(i-1)

      enddo

      !--------------------------------------------------------------------------------------
      !Grid-scale violation: Check for spurious rhoL: Zerroukat et al. (2005) Eqs.(3.1)-(3.7) 
      !--------------------------------------------------------------------------------------
      if (F_mono==3) then

         do i=i1,i2

            if ( (rhoL_8(  i)-rhoA_8(i-1))*(rhoA_8(i  )-rhoL_8(i  )) <  0.0d0 .and. & 
                ((rhoA_8(i-1)-rhoA_8(i-2))*(rhoA_8(i+1)-rhoA_8(i  )) >= 0.0d0 .or.  &
                 (rhoA_8(i-1)-rhoA_8(i-2))*(rhoA_8(i-2)-rhoA_8(i-3)) <= 0.0d0 .or.  &
                 (rhoA_8(i+1)-rhoA_8(i  ))*(rhoA_8(i+2)-rhoA_8(i+1)) <= 0.0d0 .or.  &
                 (rhoL_8(i  )-rhoA_8(i-1))*(rhoA_8(i-1)-rhoA_8(i-2)) <= 0.0d0) ) then

                 alpha_8 = 0.0d0 

                 if (abs(rhoL_8(i)-rhoA_8(i)) - abs(rhoL_8(i)-rhoA_8(i-1)) >= 0.0d0) alpha_8 = 1.0d0 

                 rhoL_8(i) = alpha_8 * rhoA_8(i-1) + (1.0d0 - alpha_8) * rhoA_8(i) 

                 if (rhoL_8(i)<0.0d0) rhoL_8(i) = 0.0d0 

                 if (i>=i1+1) rhoR_8(i-1) = rhoL_8(i)

            endif 

         enddo

      !--------------------------------------------------------------------------------------
      endif
      !--------------------------------------------------------------------------------------

      do i=i1,i2

         F_psi_8(i,1) =  1.0d0 * rhoL_8(i)
         F_psi_8(i,2) = -4.0d0 * rhoL_8(i) + 6.0d0 * rhoA_8(i) - 2.0d0 * rhoR_8(i)
         F_psi_8(i,3) =  3.0d0 * rhoL_8(i) + 3.0d0 * rhoR_8(i) - 6.0d0 * rhoA_8(i)

      enddo  

      !---------------------------------------------------------------------------------------------------------
      !Subgrid-scale violation: Check for spurious extrema in [i-1,i]: Zerroukat et al. (2005) Eqs.(3.9)-(3.13) 
      !together with Zerroukat et al. (2006) Eqs.(22)-(23) 
      !---------------------------------------------------------------------------------------------------------
      if (F_mono==3) then

         do i=i1,i2

            der0_8 = F_psi_8(i,2)
            der1_8 = F_psi_8(i,2)+2.0d0*F_psi_8(i,3) 

            if (der0_8 * der1_8 < 0.0d0                                       .and. &
                ((rhoL_8(i  )-rhoL_8(i-1))*(rhoL_8(i+2)-rhoL_8(i+1)) >= 0.0d0 .or.  &
                 (rhoL_8(i-1)-rhoL_8(i-2))*(rhoL_8(i  )-rhoL_8(i-1)) <= 0.0d0 .or.  &
                 (rhoL_8(i+2)-rhoL_8(i+1))*(rhoL_8(i+3)-rhoL_8(i+2)) <= 0.0d0 .or.  &
                 (rhoL_8(i  )-rhoL_8(i-1))*F_psi_8(i,2)              <= 0.0d0) ) then

               if ((rhoL_8(i)<=rhoR_8(i).and.der0_8<0.0d0) .or. &
                   (rhoL_8(i)>=rhoR_8(i).and.der0_8>0.0d0) ) then

                   F_psi_8(i,1) =  1.0d0 * rhoL_8(i)
                   F_psi_8(i,2) =  0.0d0
                   F_psi_8(i,3) =  3.0d0 * rhoA_8(i) - 3.0d0 * rhoL_8(i)

                   der1_rev_8 = F_psi_8(i,2)+2.0d0*F_psi_8(i,3) 

                   if (sign(ONE_8,der1_rev_8)/=sign(ONE_8,der1_8)) then

                      F_psi_8(i,1) =  1.0d0 * rhoA_8(i)
                      F_psi_8(i,2) =  0.0d0
                      F_psi_8(i,3) =  0.0d0

                   endif

               else

                   F_psi_8(i,1) = -2.0d0 * rhoR_8(i) + 3.0d0 * rhoA_8(i)
                   F_psi_8(i,2) =  6.0d0 * rhoR_8(i) - 6.0d0 * rhoA_8(i)
                   F_psi_8(i,3) = -3.0d0 * rhoR_8(i) + 3.0d0 * rhoA_8(i)

                   der0_rev_8 = F_psi_8(i,2)

                   if (sign(ONE_8,der0_rev_8)/=sign(ONE_8,der0_8)) then

                      F_psi_8(i,1) =  1.0d0 * rhoA_8(i)
                      F_psi_8(i,2) =  0.0d0
                      F_psi_8(i,3) =  0.0d0

                   endif

               endif

            endif

         enddo

      !----------------------------------------------------------------------------------------------------------
      endif
      !----------------------------------------------------------------------------------------------------------

      !Use PPM1 (Laprise and Plante 1995) when PPP2 not applicable:
      !Cubic rebuild with extrapolation at the boundaries: We resolve a linear system
      !------------------------------------------------------------------------------
      do i=1,F_n

         if (.NOT.(i<=i1.or.i>=i2)) cycle 

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

      do i=1,F_n

         if (.NOT.(i<i1.or.i>i2)) cycle 

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
