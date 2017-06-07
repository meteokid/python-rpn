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

!  s/r prgenab- To print out all the values set in set_zeta
!               Only active under Out_DEBUG_L mode
!

!
      subroutine prgenab ()
      use grid_options
      use gem_options
      implicit none
#include <arch_specific.hf>
!
! author 
!     Lee V - December 2007
!
! revision
!
! object
!    See above
! arguments
! none
!
#include "glb_ld.cdk"
#include "lun.cdk"
#include "pres.cdk"
#include "type.cdk"
#include "ver.cdk"
#include "cstv.cdk"
!

      integer k
!     __________________________________________________________________
!
!
      if (Lun_debug_L) then
      print *,'prgenab Ver_a_8%m(k),Ver_a_8%t(k),k=1,G_nk+1'
      do k=1,G_nk+1
         print *,k,Ver_a_8%m(k),Ver_a_8%t(k)
      enddo
      print *,'prgenab Ver_b_8%m(k),Ver_b_8%t(k),k=1,G_nk+1'
      do k=1,G_nk+1
         print *,k,Ver_b_8%m(k),Ver_b_8%t(k)
      enddo
      print *,'prgenab Ver_z_8%m(k),Ver_z_8%t(k),k=0,G_nk'
      do k=0,G_nk
         print *,k,Ver_z_8%m(k),Ver_z_8%t(k)
      enddo
      print *,'prgenab Ver_z_8%t(k),Ver_z_8%x(k),k=0,G_nk'
      do k=0,G_nk
         print *,k,Ver_z_8%t(k),Ver_z_8%x(k)
      enddo
      print *,'prgenab Ver_dz_8%m(k),Ver_dz_8%t(k),k=1,G_nk'
      do k=1,G_nk
         print *,k,Ver_dz_8%m(k),Ver_dz_8%t(k)
      enddo
      print *,'prgenab Ver_idz_8%m(k),Ver_idz_8%t(k),k=1,G_nk'
      do k=1,G_nk
         print *,k,Ver_idz_8%m(k),Ver_idz_8%t(k)
      enddo
      print *,'prgenab Ver_dbdz_8%m(k),Ver_dbdz_8%t(k),k=1,G_nk'
      do k=1,G_nk
         print *,k,Ver_dbdz_8%m(k),Ver_dbdz_8%t(k)
      enddo
      print *,'prgenab Ver_wp_8%m(k),Ver_wm_8%m(k),k=1,G_nk'
      do k=1,G_nk
         print *,k,Ver_wp_8%m(k),Ver_wm_8%m(k)
      enddo 
      print *,'prgenab Ver_wp_8%t(k),Ver_wm_8%t(k),k=1,G_nk'
      do k=1,G_nk
         print *,k,Ver_wp_8%t(k),Ver_wm_8%t(k)
      enddo 
      print *,'prgenab Ver_wpM_8(k),Ver_wmM_8(k),k=1,G_nk'
      do k=1,G_nk
         print *,k,Ver_wpM_8(k),Ver_wmM_8(k)
      enddo 
      print *,'prgenab Ver_wpC_8(k),Ver_wmC_8(k),k=1,G_nk'
      do k=1,G_nk
         print *,k,Ver_wpC_8(k),Ver_wmC_8(k)
      enddo 
      print *,'prgenab Ver_wpA_8(k),Ver_wmA_8(k),k=1,G_nk'
      do k=1,G_nk
         print *,k,Ver_wpA_8(k),Ver_wmA_8(k)
      enddo 
      print *,'prgenab Ver_bzz_8(k),Ver_wpstar_8(k),k=1,G_nk'
      do k=1,G_nk
         print *,k,Ver_bzz_8(k),Ver_wpstar_8(k)
      enddo
      print *,'prgenab Ver_hyb%m(k),Ver_hyb_%t(k),k=1,G_nk+1 for Output Levels'
      do k=1,G_nk+1
         print *,k,Ver_hyb%m(k),Ver_hyb%t(k)
      enddo 
      endif

      return
      end
