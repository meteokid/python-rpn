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

!**s/r adv_mass_of_area_1D - Calculates the mass (integral) of an area limited between X_Start et X_Finish located
!                            inside the control volumes Cv_Start and Cv_Finish respectively.
!                            Based on CODE Zerroukat et al(2002)/Mahidjiba et al(2008)

      subroutine adv_mass_of_area_1D ( Ni, N_Mass, Mass_8, Dx_8, Slope_8, Nxu, Cv_Start, X_Start_8, X_Left_8, Rho_Left_8, Mass_Cv_8 )

      implicit none

      integer  :: Ni, Nxu, N_Mass
      integer, Dimension(Nxu) :: Cv_Start

      real*8,  Dimension(N_Mass) :: Mass_8
      real*8,  Dimension(Ni)     :: Mass_Cv_8
      real*8,  Dimension(Ni)     :: Slope_8, Dx_8
      real*8,  Dimension(Nxu)    :: X_Left_8, Rho_Left_8
      real*8,  Dimension(Nxu)    :: X_Start_8

      !author Tanguay/Qaddouri
      !
      !revision
      ! v4_80 - Tanguay/Qaddouri - SLICE
      !
      !object
      !--------------------------------------------------------------------------------------------------------------------------
      !     The steps are as follows:
      !
      !     Given the densities (Rho_Left, Rho_Right) at the boundaries (X_left, X_right) of the control volume, we calculate:
      !
      !     y = int {rho(y) dy} between x1 and x2....................(1)
      !
      !     where rho(y) represents a continuous function over the interval [x1, x2]  and is defined by the following expression:
      !
      !     rho(y) = a0 + a1 y + a2 y^2 + a3 y^3.....................(2)
      !
      !     where a0, a1, a2, a3 are the coefficients calculated with:
      !
      !     rho(x_left ) = rho_left  ................................(2.1)
      !     rho(x_right) = rho_right ................................(2.2)
      !
      !     y = mass_cv = int {rho(y) dy} between x_left and x_right.(2.3)
      !
      !     d(rho)/dy = slope at y = (x_left + x_right)/2 ...........(2.4)
      !
      !     Note that this subroutine uses the following spatial transformation
      !
      !     s = y - x_left/(x_right - x_left)
      !--------------------------------------------------------------------------------------------------------------------------

      !------------------------------------------------------
      !CAUTION: BOUNDARY AND CENTER INDEXES DIFFERED FROM GEM
      !         X_LEFT(I) < X_CENTER(I) < X_RIGHT(I+1)
      !------------------------------------------------------

      !Local variables
      !---------------
      integer  :: Cv, i
      real*8,  Dimension(Ni)     :: XX_8
      real*8             	 :: a1_8, a2_8, a3_8, ddx_8, Diff_8, s1_8, s2_8
      real*8                     :: RRho_Left_8, m3_8

      !Initialize to ZERO
      !------------------
      Mass_Cv_8 = 0.d0
      XX_8      = 0.d0

      do i = 1, Ni

         Cv          = Cv_Start(i)
         XX_8(i)     = Min( X_Left_8(Cv + 1), X_Start_8(i+1) )
         ddx_8       = dx_8(Cv)
         RRho_Left_8 = Rho_Left_8(Cv)
	 Diff_8      = Rho_Left_8(Cv+1) - RRho_left_8
         s1_8        = ( X_Start_8(i)   - X_Left_8 (Cv) ) / ddx_8
         s2_8        = ( XX_8(i)        - X_Left_8 (Cv) ) / ddx_8
         a2_8        = Mass_8(Cv) / ddx_8 - RRho_Left_8
         a3_8        = Slope_8(Cv) * ddx_8
         a1_8        = + 3.D0 * a2_8 - a3_8
         a2_8        = - 2.D0 * a2_8 + 2.D0 * a3_8 - Diff_8
         a3_8        = - a3_8 + Diff_8

         Mass_Cv_8(i) = s2_8 * s2_8 + s1_8 * s1_8
         Mass_Cv_8(i) = ( RRho_Left_8 + a2_8 * ( Mass_Cv_8(i) + s2_8 * s1_8 ) + &
                        ( s2_8 + s1_8 ) * ( a1_8 + a3_8 * Mass_Cv_8(i) ) ) * ( XX_8(i) - X_Start_8(i) ) + &
		        sum( Mass_8( Cv_Start(i) + 1 : Cv_Start(i+1) - 1 ) )

      enddo

      do i = 1, Ni

         if ( XX_8(i) < X_Start_8(i+1) ) then

            Cv          = Cv_Start(i+1)
            ddx_8       = dx_8(Cv)
            RRho_Left_8 = Rho_Left_8(Cv)
            Diff_8      = Rho_Left_8(Cv+1) - RRho_Left_8
            s1_8        = ( X_Left_8(Cv)   - X_Left_8(Cv) ) / ddx_8
            s2_8        = ( X_Start_8(i+1) - X_Left_8(Cv) ) / ddx_8
            a2_8        = Mass_8(Cv) / ddx_8 - RRho_Left_8
            a3_8        = Slope_8(Cv) * ddx_8
            a1_8        = + 3.D0 * a2_8 - a3_8
            a2_8        = - 2.D0 * a2_8 + 2.D0 * a3_8 - Diff_8
            a3_8        = - a3_8 + Diff_8
            m3_8        = s2_8 * s2_8 + s1_8 * s1_8

            Mass_Cv_8(i) = Mass_Cv_8(i) + ( RRho_Left_8  + a2_8 * ( m3_8 + s2_8 * s1_8 ) + &
                           ( s2_8 + s1_8 ) * ( a1_8 + a3_8 * m3_8 ) ) * &
	                   ( X_Start_8(i+1) - X_Left_8(cv) )

         endif

      enddo

      end subroutine adv_mass_of_area_1D
