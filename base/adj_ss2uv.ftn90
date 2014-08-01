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

!**s/r adj_ss2uv- Returns U,V surface pressures using the incoming
!                 pressure
!
      subroutine adj_ss2uv(F_ssq0,NI,NJ,F_ssu0,NIU,NJU,F_ssv0,NIV,NJV, &
                           vcoord)
      implicit none
#include <arch_specific.hf>
!
      integer ni,nj,niu,nju,niv,njv,vcoord
      real F_ssq0(ni,nj),F_ssu0(niu,nju),F_ssv0(niv,njv)
!
!author V.Lee
!
!revision
! v4_03 - Lee V         - initial version
! v4_05 - Lee V.        - F_ssq0 is assumed now in Pascals,not milli-bars
!
!object
!       computes hydrostatic surface pressure for U, V grid


#include "glb_ld.cdk"
!*
      integer i,j
      real p0_temp(l_minx:l_maxx,l_miny:l_maxy)
!
!     ---------------------------------------------------------------
!
!Pressure coordinate
      if (vcoord.eq.0) then
          F_ssu0 = 0.0
          F_ssv0 = 0.0
          return
      endif
!     ---------------------------------------------------------------
!V stag   coordinate
      if (vcoord.eq.6) then
!         F_ssq0 is log of p0
!         Transfer to vector with halos
          do j=1,l_nj
          do i=1,l_ni
             p0_temp(i,j) = F_ssq0(i,j)
          enddo
          enddo
          call rpn_comm_xch_halo ( p0_temp, l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj,1, &
                    G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
          do j=1,nju
          do i=1,niu
             F_ssu0(i,j) = (p0_temp(i,j) + p0_temp(i+1,j)) *.5
          enddo
          enddo
          do j=1,njv
          do i=1,niv
             F_ssv0(i,j) = (p0_temp(i,j) + p0_temp(i,j+1)) *.5
          enddo
          enddo
      else
!     ---------------------------------------------------------------
!Assume p0 is in Pascals so convert to log of Pascals before averaging
          do j=1,l_nj
          do i=1,l_ni
             p0_temp(i,j) = log(F_ssq0(i,j))
          enddo
          enddo
          call rpn_comm_xch_halo ( p0_temp, l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj,1, &
                    G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
          do j=1,nju
          do i=1,niu
             F_ssu0(i,j) = exp((p0_temp(i,j) + p0_temp(i+1,j)) *.5)
          enddo
          enddo
          do j=1,njv
          do i=1,niv
             F_ssv0(i,j) = exp((p0_temp(i,j) + p0_temp(i,j+1)) *.5)
          enddo
          enddo
      endif
!
!     ---------------------------------------------------------------
!
      return
      end
