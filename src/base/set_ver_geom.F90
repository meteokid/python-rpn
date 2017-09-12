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
!**s/r set_ver_geom - initialize model vertical geometry

      subroutine set_ver_geom()
      use gem_options
      use glb_ld
      use lun
      use ver
      implicit none
#include <arch_specific.hf>


      character(len=8) :: dumc
      integer :: k, pnip1
      real :: height,heightp1
!
!     ---------------------------------------------------------------
!
      if (Lun_out > 0) write (Lun_out,1000)

      if (Lun_out > 0) then
         write (Lun_out,1005) G_nk,Hyb_rcoef
         do k=1,G_nk
            height  =-16000./log(10.)*log(Ver_hyb%m(k))

            if (k < G_nk) then
               heightp1 = -16000./log(10.)*log(Ver_hyb%m(k+1))
            end if

            if (k == G_nk) then
               heightp1 = 0.
            end if

            call convip(pnip1,Ver_hyb%m(k),5,1,dumc,.false.)
            write (Lun_out,1006) k,Ver_hyb%m(k),height, &
                                 height-heightp1,pnip1
         end do
      endif

      call canonical_cases ("SET_GEOM")

 1000 format(/,'INITIALIZATION OF MODEL VERTICAL GEOMETRY (S/R set_ver_geom)', &
             /'===============================================')
 1005 format (/'STAGGERED VERTICAL LAYERING ON',I4,' MOMENTUM HYBRID LEVELS WITH ', &
               'Grd_rcoef= ',2f7.2,':'/ &
               2x,'level',10x,'HYB',8x,'~HEIGHTS',5x,'~DELTA_Z',7x,'IP1')
 1006 format (1x,i4,3x,es15.5,2(6x,f6.0),4x,i10)

!
!     ---------------------------------------------------------------
!
      return
      end subroutine set_ver_geom
