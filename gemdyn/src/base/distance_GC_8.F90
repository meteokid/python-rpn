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

!**s/r distance_GC_8: Distance between two points based on Great Circle in horizontal
 
      function distance_GC_8 (lon_a_8,lat_a_8,lon_b_8,lat_b_8)
 
      implicit none
#include <arch_specific.hf>
 
      real*8 distance_GC_8,lon_a_8,lat_a_8,lon_b_8,lat_b_8

      !@author Monique Tanguay

      !@revisions
      !v4_XX - Tanguay M.        - GEM4 Mass-Conservation

      !---------------------------------------------------------------------

      real*8 cx_a_8,cy_a_8,cz_a_8,cx_b_8,cy_b_8,cz_b_8,chord_8,angle_8

      !----------------
      !Treat Horizontal
      !----------------

         !Conversion to Cartesian coordinates
         !-----------------------------------
         cx_a_8 = cos(lon_a_8)*cos(lat_a_8)
         cy_a_8 = sin(lon_a_8)*cos(lat_a_8)
         cz_a_8 = sin(lat_a_8)

         cx_b_8 = cos(lon_b_8)*cos(lat_b_8)
         cy_b_8 = sin(lon_b_8)*cos(lat_b_8)
         cz_b_8 = sin(lat_b_8)

         !Evaluate chord
         !--------------
         chord_8 = sqrt ( (cx_b_8-cx_a_8)**2 + (cy_b_8-cy_a_8)**2 + (cz_b_8-cz_a_8)**2 ) 

         !Evaluate angle 
         !--------------
         angle_8 = 2. * asin (0.5*chord_8)

         distance_GC_8 = angle_8

      return
      end
