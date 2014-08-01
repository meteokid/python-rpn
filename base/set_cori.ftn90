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

!**s/r set_cori - allocate and compute coriolis factor
!
      subroutine set_cori
      implicit none
#include <arch_specific.hf>

!author
!     michel roch - rpn - aug 95
!
!revision
! v2_00 - Desgagne/Lee      - initial MPI version (from setcori v1_03)
! v3_00 - Desgagne & Lee    - Lam configuration

#include "glb_ld.cdk"
#include "lun.cdk"
#include "grd.cdk"
#include "geomg.cdk"
#include "cori.cdk"

      real*8 ONE
      parameter ( ONE = 1.0 )
!
!     ---------------------------------------------------------------
!
      if (Lun_out.gt.0) write(lun_out,1000)

      allocate (cori_fcoru_8(l_minx:l_maxx,l_miny:l_maxy),&
                cori_fcorv_8(l_minx:l_maxx,l_miny:l_maxy))

      call coriol8 ( cori_fcoru_8, cori_fcorv_8, geomg_x_8, geomg_y_8,&
                     geomg_xu_8, geomg_yv_8, ONE, Grd_rot_8, &
                     l_minx,l_maxx,l_miny,l_maxy)

 1000 format( &
      /,'ALLOCATE AND COMPUTE CORIOLIS FACTOR (S/R SET_CORI)', &
      /,'==================================================')
!
!     ---------------------------------------------------------------
!
      return
      end

