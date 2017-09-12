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

!**s/r adv_mass_of_area_8 - Calculates the mass (integral) of an area limited between X_Start et X_Finish located
!                           inside the control volumes Cv_Start and Cv_Finish respectively.
!                           Based on CODE Zerroukat et al(2002)/Mahidjiba et al(2008)

      function adv_mass_of_area_8 (  x_start_8,  x_finish_8,         &
                                    cv_start  , cv_finish  ,         &
                                      x_left_8, rho_left_8 , mass_8, dx_8, slope_8,n)

      implicit none

      integer n,cv_start,cv_finish
      real*8 x_start_8,x_finish_8,adv_mass_of_area_8
      real*8 x_left_8(n+1),rho_left_8(n+1),mass_8(n),dx_8(n),slope_8(n)

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
      real*8 x1_8,x2_8,m1_8,m2_8,m3_8
      integer i,cv,piecewise_option

      real*8   adv_integral_within_cv_pcm_8, adv_integral_within_cv_ppm_8
      external adv_integral_within_cv_pcm_8, adv_integral_within_cv_ppm_8

      !--------------------------------------------------------------
      ! piecewise_option=0   ! piecewise constant
      ! piecewise_option=1   ! piecewise linear
      ! piecewise_option=2   ! piecewise parabolic Colella & Woodward
      ! piecewise_option=4   ! piecewise parabolic Laprise & Plante
      ! piecewise_option=3   ! piecewise cubic
      !--------------------------------------------------------------

      piecewise_option=2

      m1_8 = 0.0d0
      m2_8 = 0.0d0
      m3_8 = 0.0d0

      x1_8 = x_start_8
      x2_8 = x_left_8(cv_start+1)
        cv = cv_start

      if (x2_8 < x1_8-epsilon(x1_8)) then

         write(*,*)' Error: x2 < x1 inside MASS_OF_AREA '
         write(*,*)' x1 / x2 =',x1_8,' / ',x2_8
         write(*,*)' CV start / finish =',cv_start,'/',cv_finish
         stop

      elseif (x2_8 >= x_finish_8) then  ! both x1 and x2 are within the same CV

         x2_8 = x_finish_8

         select case(piecewise_option)

            case(2)

            m1_8 = adv_integral_within_cv_ppm_8( x1_8, x2_8,                      &
                                                     mass_8(cv), dx_8(cv),        &
                                                   x_left_8(cv),  x_left_8(cv+1), &
                                                 rho_left_8(cv), rho_left_8(cv+1) )


            case(3)

            m1_8 = adv_integral_within_cv_pcm_8 ( x1_8, x2_8,                           &
                                                     mass_8(cv), dx_8(cv), slope_8(cv), &
                                                   x_left_8(cv),   x_left_8(cv+1),      &
                                                 rho_left_8(cv), rho_left_8(cv+1) )
         end select

      else ! General case: a distinct cv_start

         select case(piecewise_option) ! from cv_finish with a couple in between

            case(2)

            m1_8 = adv_integral_within_cv_ppm_8( x1_8, x2_8,                       &
                                                     mass_8(cv), dx_8(cv),         &
                                                   x_left_8(cv),   x_left_8(cv+1), &
                                                 rho_left_8(cv), rho_left_8(cv+1) )

            case(3)

            m1_8 = adv_integral_within_cv_pcm_8 ( x1_8, x2_8,                           &
                                                     mass_8(cv), dx_8(cv), slope_8(cv), &
                                                   x_left_8(cv),   x_left_8(cv+1),      &
                                                 rho_left_8(cv), rho_left_8(cv+1) )
         end select

         x1_8 =  x_left_8(cv_finish)
         x2_8 =  x_finish_8
          cv  = cv_finish

         select case(piecewise_option)

            case(2)

            m3_8 = adv_integral_within_cv_ppm_8( x1_8, x2_8,                       &
                                                     mass_8(cv), dx_8(cv),         &
                                                   x_left_8(cv),   x_left_8(cv+1), &
                                                 rho_left_8(cv), rho_left_8(cv+1) )

            case(3)

            m3_8 = adv_integral_within_cv_pcm_8 ( x1_8, x2_8,                           &
                                                     mass_8(cv), dx_8(cv), slope_8(cv), &
                                                   x_left_8(cv),   x_left_8(cv+1),      &
                                                 rho_left_8(cv), rho_left_8(cv+1) )


         end select


      endif

      if (cv_finish-1-cv_start > 0) then

         do i = cv_start+1,cv_finish-1
            m2_8 = m2_8 + mass_8(i)
         enddo

      endif

      adv_mass_of_area_8 = m1_8 + m2_8 + m3_8

      return
      end
