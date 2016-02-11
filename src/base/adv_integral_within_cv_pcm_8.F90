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

!**s/r adv_integral_within_cv_pcm_8 - Integral based on PCM of Zerroukat et al(2002).
!                                     Based on CODE Zerroukat et al(2002)/Mahidjiba et al(2008)

      function adv_integral_within_cv_pcm_8 ( x1_8, x2_8, mass_cv_8, dx_8, slope_8, & 
                                              x_left_8, x_right_8, rho_left_8, rho_right_8)

      implicit none          

      real*8 mass_cv_8,x1_8,x2_8,dx_8,slope_8,x_left_8,x_right_8
      real*8 rho_left_8,rho_right_8,adv_integral_within_cv_pcm_8

      !author Tanguay/Qaddouri
      !
      !revision
      ! v4_80 - Tanguay/Qaddouri - SLICE
      !
      !object
      !-------------------------------------------------------------------------------
      ! Given the mass and the values of rho (rho_left, rho_right) at the borders
      ! (x_left, x_right) of a control volume this function computes the integral:
      !
      !                   y=x2
      !                 INT  {rho(y) dy} ..........................................(1)
      !                   y=x1
      !
      ! where rho(y) is a continuous function for y E[x1,x2] which is a piece
      ! wise cubic defined as:
      !
      !                       rho(y) = a0 + a1 y + a2 y^2 + a3 y^3.................(2)
      !
      !              where a0, a1, a2, a3 are coefficents such that:
      !
      !                        rho(x_left ) = rho_left ..........................(2.1)
      !                        rho(x_right) = rho_right .........................(2.2)
      !
      !                         y=x_right
      !                       INT   {rho(y) dy} = mass_cv .......................(2.3)
      !                         y=x_left
      !
      !                        d(rho)/dy = slope at y=(x_left+x_right)/2 ........(2.4)
      !
      ! Note that this function uses a transformed space s = y-x_left/(x_right-x_left)
      !-------------------------------------------------------------------------------

      !Local variables
      !---------------
      real*8 a0_8,a1_8,a2_8,a3_8,s1_8,s2_8,m1_8,m2_8
      real*8 mass_transformed_8, slope_transformed_8  
               
      !Check if the born of integrations x1 & x2 are within the cv
      !-----------------------------------------------------------
      if ( x1_8 < x_left_8  .or. x2_8 < x_left_8 &
           .or. x1_8 > x_right_8 .or. x2_8 > x_right_8 ) then

         write(*,*)' borns of integration out off CV '
         write(*,*)' in INTEGRAL_WITHIN_CV_PCM function'
         write(*,*)' x_left =',x_left_8
         write(*,*)' x1 =',x1_8
         write(*,*)' x2 =',x2_8
         write(*,*)' x_right =',x_right_8

         stop

      endif      
       
      !Compute transformed variables
      !----------------------------- 
      s1_8 = (x1_8-x_left_8) / dx_8
      s2_8 = (x2_8-x_left_8) / dx_8

       mass_transformed_8 = mass_cv_8 / dx_8
      slope_transformed_8 = slope_8 * dx_8
               
      !Evaluate the cubic coefficients
      !-------------------------------
      a0_8 = rho_left_8
      a1_8 = -3.*rho_left_8 + 3.*mass_transformed_8 & 
                              - slope_transformed_8          
      a2_8 = +3.*rho_left_8 - rho_right_8 & 
             -2.*mass_transformed_8 + 2.*slope_transformed_8
      a3_8 = -1.*rho_left_8 + rho_right_8 - slope_transformed_8               

      !Compute definite integral between x1 and x2 
      !-------------------------------------------
      m1_8 = a0_8*s1_8 + a1_8*s1_8**2. + a2_8*s1_8**3. + a3_8*s1_8**4.
      m2_8 = a0_8*s2_8 + a1_8*s2_8**2. + a2_8*s2_8**3. + a3_8*s2_8**4.          
               
      adv_integral_within_cv_pcm_8 = (m2_8 - m1_8) * dx_8

      return 
      end 
