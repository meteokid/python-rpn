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

!**s/r hzd_exp_geom
!
      subroutine hzd_exp_geom
      use hzd_mod
      use gem_options
      use grid_options
      implicit none
#include <arch_specific.hf>
!
!author    
!    Abdessamad Qaddouri - summer 2015
!
!revision
! v4_80 - Qaddouri, Desgagne, Lee      - Initial version
!
#include "glb_ld.cdk"
#include "geomg.cdk"

      integer i,j
      real*8  aaa,bbb,ccc,ddd,dx_8
!
!     ---------------------------------------------------------------
!
      dx_8= Grd_dx * acos( -1.0d0 )/180.0d0

      allocate ( Hzd_geom_q (l_miny:l_maxy,5),&
                 Hzd_geom_u (l_miny:l_maxy,5),&
                 Hzd_geom_v (l_miny:l_maxy,5) )

      do j= 2-G_haloy, l_nj+G_haloy-1

         aaa = (sin(Geomg_yv_8(j))-sin(Geomg_yv_8(j-1))) * Geomg_invcy2_8(j)
         bbb = (Geomg_sy_8(j  ) - Geomg_sy_8(j-1)) / Geomg_cyv2_8(j-1)
         ccc = (Geomg_sy_8(j+1) - Geomg_sy_8(j  )) / Geomg_cyv2_8(j  )
         ddd = 1.0d0 / (dx_8 * (sin(Geomg_yv_8 (j)) - sin(Geomg_yv_8 (j-1))))

         Hzd_geom_q(j,2)= ddd*aaa/dx_8
         Hzd_geom_q(j,3)= Hzd_geom_q(j,2)
         Hzd_geom_q(j,4)= ddd*dx_8/bbb
         Hzd_geom_q(j,5)= ddd*dx_8/ccc
         Hzd_geom_q(j,1)=-(Hzd_geom_q(j,2)+Hzd_geom_q(j,3)+ &
                             Hzd_geom_q(j,4)+Hzd_geom_q(j,5))

         aaa= (Geomg_sy_8(j+1) - Geomg_sy_8(j)) * Geomg_invcy2_8(j)
         bbb= Geomg_cy2_8(j  ) / (sin(Geomg_yv_8 (j)) - sin(Geomg_yv_8 (j-1)))
         ccc= Geomg_cy2_8(j+1) / (sin(Geomg_yv_8 (j+1)) - sin(Geomg_yv_8 (j)))
         ddd= 1.d0 / ( dx_8 * (Geomg_sy_8(j+1) - Geomg_sy_8(j)) )

         Hzd_geom_v(j,2)=ddd*aaa/dx_8
         Hzd_geom_v(j,3)=Hzd_geom_v(j,2)
         Hzd_geom_v(j,4)=ddd*dx_8*bbb
         Hzd_geom_v(j,5)=ddd*dx_8*ccc
         Hzd_geom_v(j,1)=-(Hzd_geom_v(j,2)+Hzd_geom_v(j,3)+ &
                             Hzd_geom_v(j,4)+Hzd_geom_v(j,5))

      end do

      Hzd_geom_u = Hzd_geom_q
!
!     ---------------------------------------------------------------
!
      return
      end
