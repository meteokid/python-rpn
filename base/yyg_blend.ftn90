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
!**s/r yyg_blend - for blending Yin-Yang wind boundary conditions
!
      subroutine yyg_blend (F_apply_L)
      implicit none
#include <arch_specific.hf>

      logical F_apply_L

!author 
!     Vivian Lee   - Spring 2014
!
!revision
! v4_70 - Lee V.        - Initial version

#include "gmm.hf"
#include "glb_ld.cdk"
#include "vt1.cdk"
#include "lun.cdk"

      integer istat
      real tempzd (l_minx:l_maxx,l_miny:l_maxy, l_nk)
!
!----------------------------------------------------------------------
!
      if (.not. F_apply_L) return

      if (Lun_debug_L) write (Lun_out,1001)

      istat = gmm_get(gmmk_ut1_s , ut1)
      istat = gmm_get(gmmk_vt1_s , vt1)
      istat = gmm_get(gmmk_zdt1_s,zdt1)

      tempzd = zdt1

      call rpn_comm_xch_halo( tempzd, l_minx,l_maxx,l_miny,l_maxy, &
                              l_ni,l_nj,G_nk,G_halox,G_haloy     , &
                              G_periodx,G_periody,l_ni,0 )
      call yyg_blenbc2 (zdt1, tempzd, l_minx,l_maxx,l_miny,l_maxy,G_nk)

      call yyg_blenuv (ut1, vt1, l_minx,l_maxx,l_miny,l_maxy,G_nk)

      call pw_update_UV
!
!----------------------------------------------------------------------
 1001 format(3X,'BLEND YY Boundary ConditionS: (S/R yyg_blend)')
!
      return
      end
