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

!**s/r set_intuv - computes (u,v) cubic lagrange interpolation coefficients
!
      subroutine set_intuv
      implicit none
#include <arch_specific.hf>

!author
!       jean cote/andre methot - rpn/cmc - sept 96
!
!revision
! v2_00 - Desgagne M.       - initial MPI version (from setinuvl v1_03)
! v3_00 - Desgagne & Lee    - Lam configuration

#include "glb_ld.cdk"
#include "lun.cdk"
#include "geomg.cdk"
#include "intuv.cdk"
#include "inuvl.cdk"
#include "dcst.cdk"
#include "ptopo.cdk"

      real*8 zero, half, one, two, three, alpha1, alpha2
      parameter( zero   = 0.0 )
      parameter( half   = 0.5 )
      parameter( two    = 2.0 )
      parameter( alpha1 = -1.d0/16.d0 )
      parameter( alpha2 = 9.d0/16.d0 )


      integer  i, j, indx, offi, offj, err

!   * Statement functions
      real*8 hh
      real*8 lag2, lag3, x, x1, x2, x3, x4
     
      lag2( x, x1, x2, x3 ) = &
       ( ( x  - x2 ) * ( x  - x3 ) )/ &
       ( ( x1 - x2 ) * ( x1 - x3 ) )
      lag3( x, x1, x2, x3, x4 ) = &
       ( ( x  - x2 ) * ( x  - x3 ) * ( x  - x4 ) )/ &
       ( ( x1 - x2 ) * ( x1 - x3 ) * ( x1 - x4 ) )
!
!     ---------------------------------------------------------------
!
      if (Lun_out.gt.0) write ( Lun_out, 1000 )
!
      offi = Ptopo_gindx(1,Ptopo_myproc+1)-1
      offj = Ptopo_gindx(3,Ptopo_myproc+1)-1
!
      allocate ( inuvl_wyyv3_8(l_miny:l_maxy,4),inuvl_wyvy3_8(l_miny:l_maxy,4))
      
!           
!
     
!
      do j = 1-G_haloy,l_nj+G_haloy
    
         inuvl_wyyv3_8(j,1) = alpha1 !-1/16  =alpha1
         inuvl_wyyv3_8(j,2) = alpha2 ! 9/16 =alpha2
         inuvl_wyyv3_8(j,3) = alpha2 ! 9/16 =alpha2
         inuvl_wyyv3_8(j,4) = alpha1 !-1/16  =alpha1
!
         inuvl_wyvy3_8(j,1) = alpha1
         inuvl_wyvy3_8(j,2) = alpha2
         inuvl_wyvy3_8(j,3) = alpha2
         inuvl_wyvy3_8(j,4) = alpha1
!
      end do

      if (l_north) then
         x2 = geomg_yv_8(l_nj-2)
         x3 = geomg_yv_8(l_nj-1)
         x4 = Dcst_pi_8/two
         inuvl_wyvy3_8(l_nj,1) = lag2(geomg_y_8(l_nj), x2, x3, x4 )
         inuvl_wyvy3_8(l_nj,2) = lag2(geomg_y_8(l_nj), x3, x2, x4 )
         inuvl_wyvy3_8(l_nj,3) = lag2(geomg_y_8(l_nj), x4, x2, x3 )
         inuvl_wyvy3_8(l_nj,4) = 0.0
         indx = offj + l_njv
         hh = (G_yg_8(indx+1)+ G_yg_8(indx)) * HALF
         x1 = G_yg_8(indx-1)
         x2 = G_yg_8(indx)
         x3 = G_yg_8(indx+1)
         x4 = Dcst_pi_8/two
         inuvl_wyyv3_8(l_njv,1) = lag3( hh, x1, x2, x3, x4 )
         inuvl_wyyv3_8(l_njv,2) = lag3( hh, x2, x1, x3, x4 )
         inuvl_wyyv3_8(l_njv,3) = lag3( hh, x3, x1, x2, x4 )
         inuvl_wyyv3_8(l_njv,4) = lag3( hh, x4, x1, x2, x3 )
      endif
      if (l_south) then
         indx = offj + 2
         x2 = (G_yg_8(indx-1)+ G_yg_8(indx-2)) * HALF
         x3 = (G_yg_8(indx  )+ G_yg_8(indx-1)) * HALF
         x4 = (G_yg_8(indx+1)+ G_yg_8(indx  )) * HALF
         inuvl_wyvy3_8(1,1) = 0.0
         inuvl_wyvy3_8(1,2) = lag2(geomg_y_8(1), x2, x3, x4 )
         inuvl_wyvy3_8(1,3) = lag2(geomg_y_8(1), x3, x2, x4 )
         inuvl_wyvy3_8(1,4) = lag2(geomg_y_8(1), x4, x2, x3 )
         indx = offj + 1
         hh = (G_yg_8(indx+1)+ G_yg_8(indx)) * HALF
         x1 = -Dcst_pi_8/two
         x2 = G_yg_8(indx)
         x3 = G_yg_8(indx+1)
         x4 = G_yg_8(indx+2)
         inuvl_wyyv3_8(1,1) = lag3( hh, x1, x2, x3, x4 )
         inuvl_wyyv3_8(1,2) = lag3( hh, x2, x1, x3, x4 )
         inuvl_wyyv3_8(1,3) = lag3( hh, x3, x1, x2, x4 )
         inuvl_wyyv3_8(1,4) = lag3( hh, x4, x1, x2, x3 )
      endif
!
 1000 format( &
       /,'COMPUTE (U,V) INTERPOLATION COEFFICIENTS (S/R SET_INTUV)', &
       /,'========================================================')
!
!     ---------------------------------------------------------------
!
      return
      end
