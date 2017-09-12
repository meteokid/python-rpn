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

!**s/r nest_gwa

      subroutine nest_gwa()
      use nest_blending, only: nest_blend
      use gem_options
      use mtn_options
      use theo_dif
      use theo_options
      implicit none
#include <arch_specific.hf>

!author
!     Michel Desgagne   - Spring 2006
!
!revision
! v3_30 - Lee V.          - initial version
! v4_05 - Plante A.       - top blending
! v4_05 - Lepine M.       - VMM replacement with GMM
! v4_40 - Lee V.          - no blending for Yin-Yang
!
!----------------------------------------------------------------------
!
      call nest_HOR_gwa()

      if ( Schm_theoc_L ) then

         if ( hdif_lnr > 0. ) call theo_hdif_main()
         if ( mtn_zblen_thk > 0. ) call height_sponge()

         call slabsym()
         if( Theo_case_S == 'BUBBLE' ) call mirror()

      endif
!
!----------------------------------------------------------------------
!
      return
      end
