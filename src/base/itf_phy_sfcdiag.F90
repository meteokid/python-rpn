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
!/@*
      subroutine itf_phy_sfcdiag (F_dest, minx, maxx, miny, maxy, &
                                  F_var_S, F_status, F_quiet_L)
      use phy_itf, only: phy_get
      use grid_options
      implicit none
#include <arch_specific.hf>

   !@objective
   !@arguments
      character(len=*) :: F_var_S
      logical :: F_quiet_L
      integer :: minx, maxx, miny, maxy, F_status
      real, dimension(minx:maxx,miny:maxy,1), target :: F_dest
   !@author  Michel Desgagne  -  Summer 2013
   !*@/

#include "glb_ld.cdk"

      real, dimension(:,:,:), pointer :: ptr3d
!
!-----------------------------------------------------------------
!
      ptr3d => F_dest(Grd_lphy_i0:Grd_lphy_in,Grd_lphy_j0:Grd_lphy_jn,:)

      F_status = phy_get (ptr3d, F_var_S, F_npath='VO', F_bpath='D'    ,&
                       F_start=(/-1,-1,l_nk+1/), F_end=(/-1,-1,l_nk+1/),&
                       F_quiet=F_quiet_L)
!
!-----------------------------------------------------------------
!
      return
      end subroutine itf_phy_sfcdiag
