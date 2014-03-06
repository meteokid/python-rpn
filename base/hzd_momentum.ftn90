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

!**s/r hzd_momentum - applies horizontal diffusion on zdt and possibly on u and v
!                     to smooth momentum components for trajectory calculations after 
!                     a Crank-Nicholson step avoiding pole problems
!
      subroutine hzd_momentum
      implicit none
#include <arch_specific.hf>
!
!author
!     Claude Girard - spring 2012
!
!revision
! v4_40 - Girard C.      - initial version
!
!object
!	
!arguments
!     none
!
#include "gmm.hf"
#include "glb_ld.cdk"
#include "schm.cdk"
#include "vt0.cdk"
#include "grd.cdk"

      logical, save :: switch_on_hzd= .true.
      integer istat
!     _________________________________________________________________
!
      if (Schm_hzdadw_L .and. switch_on_hzd) then

         istat= gmm_get(gmmk_zdt0_s,zdt0)
         istat= gmm_get(gmmk_ut0_s ,ut0 )
         istat= gmm_get(gmmk_vt0_s ,vt0 )

         call hzd_ctrl3(ut0 , 'U', G_nk)
         call hzd_ctrl3(vt0 , 'V', G_nk)
         call hzd_ctrl3(zdt0, 'S', G_nk)
         
         if (Grd_yinyang_L) then
            call yyg_xchng (zdt0, l_minx,l_maxx,l_miny,l_maxy, &
                            G_nk, .false., 'CUBIC')
            call yyg_nestuv(ut0 , vt0, l_minx,l_maxx,l_miny,l_maxy,G_nk)
         endif

      endif
!     _________________________________________________________________
!
      return
      end
