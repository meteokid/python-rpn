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

!**s/r coriol8 - compute coriolis FACTOR multiplied by a constant
!		 on the U and V grids
!

!
      subroutine coriol8 ( F_u_8, F_v_8, F_x_8, F_y_8,  &
                           F_xu_8, F_yv_8, F_ct_8, F_rot_8, Minx,Maxx,Miny,Maxy)
!
      implicit none
#include <arch_specific.hf>
!
      integer Minx,Maxx,Miny,Maxy
      real*8 F_u_8(Minx:Maxx,Miny:Maxy), F_v_8(Minx:Maxx,Miny:Maxy),  &
             F_x_8(Minx:Maxx), F_y_8(Miny:Maxy), &
             F_xu_8(Minx:Maxx), F_yv_8(Miny:Maxy), &
             F_ct_8,               F_rot_8(3,3)
!
!author
!     michel roch/jean cote - august 1995 - from coriol3
!
!revision
! v2_00 - Desgagne/Lee      - initial MPI version (from coriol v1_03)
! v3_11 - Gravel S          - theoretical case
! v4_04 - Tanguay M.        - Williamson's cases   
! v4_40 - Qaddouri A.       - adjustment for Yang grid
!
!object
!       See above id.
!
!arguments
!  Name        I/O                 Description
!----------------------------------------------------------------
! F_u_8        O    - coriolis FACTOR on U grid
! F_v_8        O    - coriolis FACTOR on V grid
! F_x_8        I    - longitudes in radians PHI grid
! F_y_8        I    - latitudes in radians PHI grid
! F_xu_8       I    - longitudes in radians U grid
! F_yv_8       I    - latitudes in radians V grid
! F_ct_8       I    - multiplicative constant
! F_rot_8      I    - rotation matrix of the grid
!
#include "glb_ld.cdk"
#include "lun.cdk"
#include "dcst.cdk"
#include "grd.cdk"
#include "schm.cdk"
#include "wil_williamson.cdk"

      real*8 ZERO, ONE, TWO
      parameter( ZERO = 0.0 )
      parameter( ONE  = 1.0 )
      parameter( TWO  = 2.0 )
!
      integer i, j, pn
      real*8  c0, sa, ca, s0, ang, coef_8, s(2,2), Y_lat,Y_lon
!
!-------------------------------------------------------------------
!
      if ( Schm_theoc_L ) then
         F_u_8 = ZERO ; F_v_8 = ZERO
         return
      endif
!
!     Set rotation parameters (standard mode)
!     ---------------------------------------
      if ( .not.Schm_autobar_L .or. &

       (Schm_autobar_L .and. Williamson_alpha.eq.0.0)) then
!
         s0 = F_rot_8(3,3)

         if ( abs( (abs(s0)-ONE) ).gt.1.0e-10 ) then
            if (Lun_out.gt.0)  &
                  write( Lun_out, '(''rotation OF CORIOLIS FACTOR'')')
            ang = atan2( F_rot_8(2,3), F_rot_8(1,3) )
         else
            if (Lun_out.gt.0) &
                   write( Lun_out, '(''NO rotation OF CORIOLIS FACTOR'')')
            s0 = sign( ONE, s0 )
            ang = ZERO
         endif
!
         c0 = sqrt( max( ZERO, ONE - s0 ** 2 ) )
!
!	processing coriolis FACTOR on V grid
!       ____________________________________
!
         do j=1-G_haloy,l_nj+G_haloy
            sa = ( TWO * Dcst_omega_8 * F_ct_8 ) * s0 * sin(F_yv_8(j))
            ca = ( TWO * Dcst_omega_8 * F_ct_8 ) * c0 * cos(F_yv_8(j))
            do i=1-G_halox,l_ni+G_halox
               F_v_8(i,j) = ca * cos(F_x_8(i)-ang) + sa
            enddo
         enddo

!	processing coriolis FACTOR on U grid
!       ____________________________________
!
         do j=1-G_haloy,l_nj+G_haloy
            sa = ( TWO * Dcst_omega_8 * F_ct_8 ) * s0 * sin(F_y_8(j))
            ca = ( TWO * Dcst_omega_8 * F_ct_8 ) * c0 * cos(F_y_8(j))
            do i=1-G_halox,l_ni+G_halox
               F_u_8(i,j) = ca * cos(F_xu_8(i) - ang) + sa
            enddo
         enddo

!
      else
!
!     ------------------------------------------------------------------------
!     Use Coriolis based on alpha when 
!     AUTOBAROTROPE and Williamson's cases 2 and 3
!     ------------------------------------------------------------------------

         coef_8 = TWO * Dcst_omega_8 * F_ct_8
!
         if (trim(Grd_yinyang_S) .eq. 'YAN') then
!
!       processing coriolis FACTOR on V grid
!       ____________________________________
!
            do j=1-G_haloy,l_nj+G_haloy
            do i=1-G_halox,l_ni+G_halox
               call smat (S,Y_lon,Y_lat,F_x_8(i),F_yv_8(j))
               F_v_8(i,j) = coef_8 *   &
                    (-cos( Y_lon)*cos(Y_lat)*sin(Williamson_alpha)+ &
                            sin(Y_lat)* cos(Williamson_alpha))
            enddo
            enddo
!
!       processing coriolis FACTOR on U grid
!       ____________________________________
!
            do j=1-G_haloy,l_nj+G_haloy
            do i=1-G_halox,l_ni+G_halox
               call smat (S,Y_lon,Y_lat,F_xu_8(i),F_y_8(j))
               F_u_8(i,j) = coef_8 *   &
                    (-cos( Y_lon)*cos(Y_lat)*sin(Williamson_alpha)+ &
                            sin(Y_lat)* cos(Williamson_alpha))
            enddo
            enddo

         else
            write(Lun_out,*) ''
            write(Lun_out,*) 'Coriolis evaluation using alpha when Williamson cases 2 and 3 '
            write(Lun_out,*) ''
!
!       processing coriolis FACTOR on V grid
!       ____________________________________
!
            do j=1-G_haloy,l_nj+G_haloy
            do i=1-G_halox,l_ni+G_halox
               F_v_8(i,j) = coef_8*   &
                    (-cos( F_x_8(i))* cos(F_yv_8(j))*sin(Williamson_alpha)+ &
                            sin(F_yv_8(j))* cos(Williamson_alpha))
            enddo
            enddo
!
!       processing coriolis FACTOR on U grid
!       ____________________________________
!
            do j=1-G_haloy,l_nj+G_haloy
            do i=1-G_halox,l_ni+G_halox
               F_u_8(i,j) = coef_8*   &
                    (-cos( F_xu_8(i))* cos(F_y_8(j))*sin(Williamson_alpha)+ &
                            sin(F_y_8(j))* cos(Williamson_alpha))
            enddo
            enddo
         endif
!
      endif
!
!-------------------------------------------------------------------
!
      return
      end
