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

!** s/r ens_uvduv - Computes sqrt((u*du)**2 +(v*dv)**2) of wind-like fields 
!
      subroutine ens_uvduv ( F_duv, F_uu, F_vv, Minx,Maxx,Miny,Maxy, Nk )
!
      implicit none
#include <arch_specific.hf>
!
      integer  Minx,Maxx,Miny,Maxy, Nk
      real     F_duv(Minx:Maxx,Miny:Maxy,Nk), &
               F_uu (Minx:Maxx,Miny:Maxy,Nk), &
               F_vv (Minx:Maxx,Miny:Maxy,Nk)

#include "glb_ld.cdk"
      integer i, j, k
!
!     __________________________________________________________________
!
! this routine is not usefull and should be inlined in the caller ens_filter.F90 ; same also with ens_uvgdw.F90
      do k = 1, Nk
      do j = 0, l_nj
      do i = 0, l_ni
         F_duv(i,j,k) = 0.5*sqrt( (F_uu(i,j  ,k)+F_uu(i,j-1  ,k))**2 &
                                + (F_vv(i,j-1,k)+F_vv(i-1,j-1,k))**2 )
      end do
      end do
      end do
!
!     __________________________________________________________________
!
      return
      end
