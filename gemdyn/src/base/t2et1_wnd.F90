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

!**s/r t2et1_wnd - T2 = T1 (Winds only)

      subroutine t2et1_wnd
      use gmm_vt1
      use gmm_vt2
      use geomh
      use glb_ld
      use gmm_itf_mod
      implicit none


   type(gmm_metadata) :: dummy_gmm_meta
   integer :: istat
!
!  __________________________________________________________________
!
   istat = GMM_OK
   istat = min(gmm_get(gmmk_ut1_s ,ut1 ,dummy_gmm_meta),istat)
   istat = min(gmm_get(gmmk_vt1_s ,vt1 ,dummy_gmm_meta),istat)
   istat = min(gmm_get(gmmk_zdt1_s,zdt1,dummy_gmm_meta),istat)

   istat = min(gmm_get(gmmk_ut2_s ,ut2 ,dummy_gmm_meta),istat)
   istat = min(gmm_get(gmmk_vt2_s ,vt2 ,dummy_gmm_meta),istat)
   istat = min(gmm_get(gmmk_zdt2_s,zdt2,dummy_gmm_meta),istat)

    ut2 =  ut1
    vt2 =  vt1
   zdt2 = zdt1

   return
   end
