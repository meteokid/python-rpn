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

!**s/r adv_rebuild - Integral based on CODE Zerroukat et al(2002)/Mahidjiba et al(2008) (INTEGRAL_WITHIN_CV_PCM)  

      subroutine adv_rebuild (F_psi,F_x_left,F_mass_cv,F_n,F_rebuild)

      implicit none          

      integer F_n,F_rebuild
      real*8 F_psi(F_n,4),F_x_left(0:F_n),F_mass_cv(F_n)

      !author Tanguay/Qaddouri
      !
      !revision
      ! v4_80 - Tanguay/Qaddouri - SLICE

      !Local variables 
      !---------------
      integer i
      real*8 rho_left_all(0:F_n),slope(F_n),x1,x2,dx,rho_left,rho_right, &
             mass_transformed,slope_transformed  
               
      !Compute RHO at LEFT boundary and First Derivate of RHO at CENTER cell 
      !---------------------------------------------------------------------
      if (F_rebuild==3.or.F_rebuild==4) & 
      call adv_compute_cv_LEFT_boundary_values (F_mass_cv,slope,F_x_left,rho_left_all,F_n,F_rebuild) 

      F_psi(:,:) = 0.0

      do i = 1,F_n

         !Compute transformed variables
         !-----------------------------
         rho_left  = rho_left_all(i-1)
         rho_right = rho_left_all(i)
         dx = F_x_left(i)-F_x_left(i-1)

           mass_transformed = F_mass_cv(i) / dx
          slope_transformed = slope(i) * dx

         !Evaluate the cubic coefficients
         !-------------------------------
         if (F_rebuild==3) then !TO BE REMOVED WITH slope_transformed = rho_right - rho_left

         F_psi(i,1) = rho_left 
         F_psi(i,2) = 2.0*(+3.*mass_transformed - 2.*rho_left - rho_right)
         F_psi(i,3) = 3.0*(-2.*mass_transformed + rho_left+rho_right)

         elseif (F_rebuild==4) then

         F_psi(i,1) = rho_left
         F_psi(i,2) = 2.0* (-3.*rho_left + 3.*mass_transformed &
                      - slope_transformed )
         F_psi(i,3) = 3.0* (+3.*rho_left - rho_right  &
                      - 2.*mass_transformed + 2.*slope_transformed)
         F_psi(i,4) = 4.0*( -1.*rho_left + rho_right - slope_transformed )

         endif

      enddo

      return
      end
