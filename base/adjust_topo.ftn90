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

!** s/r adjust_topo -
!

!
      subroutine adjust_topo2(F_topo, F_target, F_lowres, F_blend_L,&
                                                   Minx,Maxx,Miny,Maxy, ni, nj )
      use nest_blending, only: nest_blend
      implicit none
#include <arch_specific.hf>
!
      logical F_blend_L
      integer Minx,Maxx,Miny,Maxy, ni, nj
      real    F_topo(Minx:Maxx,Miny:Maxy), F_target(ni,nj), F_lowres(ni,nj)
!
!authors
!      M.Desgagne   -  Spring 2010
!
!revision
! v4_13 - Plante A.   - initial version 
! v4_40 - Plante A.   - clean up

#include "gmm.hf"
#include "glb_ld.cdk"
#include "schm.cdk"
#include "p_geof.cdk"
#include "vtopo.cdk"

      type(gmm_metadata) :: mymeta
      integer istat
!
!     ________________________________________________________________
!
      if (.not. Schm_topo_L) F_target = 0.
!
      if (G_lam .and. F_blend_L) &
        call nest_blend (F_target,F_lowres,1,l_ni,1,l_nj,'M',level=G_nk+1)
!
      F_topo (1:l_ni,1:l_nj) = F_target(1:l_ni,1:l_nj)

      if (Vtopo_L) then
         istat = gmm_get(gmmk_topo_low_s , topo_low , mymeta)
         istat = gmm_get(gmmk_topo_high_s, topo_high, mymeta)
         topo_low (1:l_ni,1:l_nj) = F_lowres (1:l_ni,1:l_nj)
         topo_high(1:l_ni,1:l_nj) = F_target (1:l_ni,1:l_nj)
         F_topo   (1:l_ni,1:l_nj) = F_lowres (1:l_ni,1:l_nj)
      endif

      call rpn_comm_xch_halo ( F_topo, l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj,1, &
                    G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
!     ________________________________________________________________
!
      return
      end

