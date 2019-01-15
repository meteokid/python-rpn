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

!**s/p adv_compute_cv_boundary_values: Estimate RHO at LEFT boundary of the control volume CV
!                                      together with the slope of RHO at the center of CV.
!                                      Based on CODE Zerroukat et al(2002)/Mahidjiba et al(2008)

      subroutine adv_compute_cv_LEFT_boundary_values (mass_8,slope_rho_8,x_left_cv_8,rho_left_cv_8,n,F_rebuild)

      implicit none

      integer ::  n,F_rebuild
      real*8, dimension(n)   ::  mass_8,slope_rho_8
      real*8, dimension(n+1) :: x_left_cv_8,rho_left_cv_8

      !author Tanguay/Qaddouri
      !
      !revision
      ! v4_80 - Tanguay/Qaddouri - SLICE

      !------------------------------------------------------
      !CAUTION: BOUNDARY AND CENTER INDEXES DIFFERED FROM GEM
      !         X_LEFT(I) < X_CENTER(I) < X_RIGHT(I+1)
      !------------------------------------------------------

      !Local variables
      !---------------
      real*8, dimension(n) :: x_centre_cv_8
      real*8, dimension(:), allocatable :: cumulative_mass_8,mass_of_4cvs_8,x_left_of_4cvs_8
      real*8, dimension(2) :: xp_i_8,slope_i_8,slope_previous_8
      integer :: i,im,ip,cv_start,cv_finish
      integer, parameter :: standard_number_of_cvs = 4

      do i = 1,n
         x_centre_cv_8(i) = 0.5*(x_left_cv_8(i)+x_left_cv_8(i+1))
      enddo

      allocate( cumulative_mass_8(standard_number_of_cvs+1), &
                   mass_of_4cvs_8(standard_number_of_cvs),   &
                 x_left_of_4cvs_8(standard_number_of_cvs+1) )

      do i = 1,n+1

         !Compute Cumulative mass
         !-----------------------
         cv_start  = max(i-standard_number_of_cvs/2,1)    ! cv at which cumulative mass starts
         cv_finish = cv_start+(standard_number_of_cvs-1)  ! cv at which cumulative mass finish

         if (cv_finish > n) then
             cv_finish = n
             cv_start  = cv_finish-(standard_number_of_cvs-1)
         endif

           mass_of_4cvs_8 =      mass_8(cv_start:cv_finish)
         x_left_of_4cvs_8 = x_left_cv_8(cv_start:cv_finish+1)

         call discrete_cumulative_mass (mass_of_4cvs_8,cumulative_mass_8,standard_number_of_cvs)

         !Compute First Derivate of Cumulative mass at LEFT boundary
         !----------------------------------------------------------
         call diff_polynome (x_left_cv_8(i),rho_left_cv_8(i),1,x_left_of_4cvs_8,cumulative_mass_8, &
                             standard_number_of_cvs+1)

         if (F_rebuild==4) then

            !Compute Second Derivative of Cumulative mass at CENTER cell
            !-----------------------------------------------------------
            im = max(1,i-1)               ! CV index of CV(i-1)
            ip = min(i,n)                 ! CV index at CV(i)
            xp_i_8(1) = x_centre_cv_8(im) !   centre of CV(i-1)
            xp_i_8(2) = x_centre_cv_8(ip) !   centre of CV(i)


            call diff2_polynome (xp_i_8,slope_i_8,2,x_left_of_4cvs_8,cumulative_mass_8, &
                                 standard_number_of_cvs+1)

            if (i > 1) then
               slope_rho_8(i-1) = 0.5*(slope_i_8(1)+slope_previous_8(2))
            endif

            slope_previous_8(1) = slope_i_8(1)
            slope_previous_8(2) = slope_i_8(2)

         elseif (F_rebuild==3) then

            if (i > 1) then
               slope_rho_8(i-1) = (rho_left_cv_8(i)-rho_left_cv_8(i-1))/(x_left_cv_8(i)-x_left_cv_8(i-1))
            endif

         endif

      enddo

      deallocate( cumulative_mass_8,mass_of_4cvs_8,x_left_of_4cvs_8 )

      return
      end

!-------------------------------------------------------
!**s/r discrete_cumulative_mass - Return cumulative mass
!-------------------------------------------------------

      subroutine discrete_cumulative_mass (mass_8,cumulative_mass_8,n)

      implicit none

      !Input arguments
      !---------------
      integer n
      real*8 mass_8(n),cumulative_mass_8(n+1)

      !Local variables
      !---------------
      integer i

      cumulative_mass_8(1) = 0.d0

      do i = 2,n+1
         cumulative_mass_8(i) = cumulative_mass_8(i-1)+mass_8(i-1)
      enddo

      return
      end

!-------------------------------------------------------------------------------------------------------
!**s/r diff_polynome - Return the 1st derivatives (dp/dx)(x(i)),i=1,n of the
!                      polynomial p(x) of degree (np-1) that fits the given points (xp(i),yp(i)), i=1,np
!-------------------------------------------------------------------------------------------------------

      subroutine diff_polynome (x_8,y_8,n,xp_8,yp_8,np)

      implicit none

      !Input arguments
      !---------------
      integer n,np
      real*8 x_8(n),y_8(n),xp_8(np),yp_8(np)

      !Local variables
      !---------------
      integer i,j
      real*8 w_8(np)

      call polynomial_coefficients (xp_8,yp_8,np,w_8)

      do i = 1,n
         y_8(i) = 0.d0
         do j = 2,np
            y_8(i) = y_8(i)+(j-1)*w_8(j)*(x_8(i)**(j-2))
         enddo
      enddo

      return
      end

!--------------------------------------------------------------------------------------------------------
!**s/r diff2_polynome - Return the 2nd derivatives (d^2p/dx^2)(x(i)),i=1,n of the
!                       polynomial p(x) of degree (np-1) that fits the given points (xp(i),yp(i)), i=1,np
!--------------------------------------------------------------------------------------------------------

      subroutine diff2_polynome (x_8,y_8,n,xp_8,yp_8,np)

      implicit none

      !Input arguments
      !---------------
      integer n,np
      real*8 x_8(n),y_8(n),xp_8(np),yp_8(np)

      !Local variables
      !---------------
      integer i,j
      real*8 w_8(np)

      if (np < 3) then
         do i = 1,n
            y_8(i) = 0.d0
         enddo
      else
         call polynomial_coefficients (xp_8,yp_8,np,w_8)
         do i = 1,n
            y_8(i) = 0.d0
            do j = 3,np
               y_8(i) = y_8(i)+(j-2)*(j-1)*w_8(j)*(x_8(i)**(j-3))
            enddo
         enddo
      endif

      return
      end

!-----------------------------------------------------------------------------------------------------------------
!**s/r polynomial_coefficients - Return the array {coeff} which are the coefficients
!                                of the polynomial p(x) of degree (n-1) that fits the n points (x(i),y(i)),i=1.n).
!                                The polynomial p(x) = coeff(1)+coff(2)*x +.....+ coeff(n)*x^{n-1}.
!                                EXTRACTED FROM NUMERICAL RECIPES (POLCOE)
!-----------------------------------------------------------------------------------------------------------------

      subroutine polynomial_coefficients (x_8,y_8,n,coeff_8)

      implicit none

      !Input arguments
      !---------------
      integer n
      real*8 coeff_8(n),x_8(n),y_8(n)

      !Local variables
      !---------------
      integer i,j,k
      real*8 b_8,ff_8,phi_8,s_8(n)

      do i = 1,n
         s_8(i) = 0.0d0
         coeff_8(i) = 0.d0
      enddo
      s_8(n) = -x_8(1)
      do i = 2,n
         do j = n+1-i,n-1
            s_8(j) = s_8(j)-x_8(i)*s_8(j+1)
         enddo
         s_8(n) = s_8(n)-x_8(i)
      enddo
      do j = 1,n
         phi_8 = n
         do k = n-1,1,-1
            phi_8 = k*s_8(k+1)+x_8(j)*phi_8
         enddo
         ff_8 = y_8(j)/phi_8
         b_8 = 1.0d0
         do k = n,1,-1
            coeff_8(k) = coeff_8(k)+b_8*ff_8
            b_8 = s_8(k)+x_8(j)*b_8
         enddo
      enddo

      return
      end
