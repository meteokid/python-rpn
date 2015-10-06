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

!**s/r inp_get - Read variable F_var_S valid at F_datev, perform hor. 
!                interpolation to F_hgrid_S hor. grid and perform
!                vertical interpolation to F_vgrid_S vertical grid

      integer function inp_get ( F_var_S, F_hgrid_S, F_ver_ip1, &
                                 F_vgd_src, F_vgd_dst, &
                                 F_sfc_src, F_sfc_dst, F_dest,&
                                 Minx,Maxx,Miny,Maxy, F_nk )
      use vGrid_Descriptors
      implicit none
#include <arch_specific.hf>

      character*(*)          , intent(IN)  :: F_var_S,F_hgrid_S
      integer                , intent(IN)  :: Minx,Maxx,Miny,Maxy, F_nk
      integer                , intent(IN)  :: F_ver_ip1(F_nk)
      type(vgrid_descriptor) , intent(IN)  :: F_vgd_src, F_vgd_dst
      real, dimension(Minx:Maxx,Miny:Maxy     ), target, &
                                   intent(IN) :: F_sfc_src, F_sfc_dst
      real, dimension(Minx:Maxx,Miny:Maxy,F_nk), intent(OUT):: F_dest

Interface
      integer function inp_read ( F_var_S, F_hgrid_S, F_dest,&
                                  F_ip1, F_nka )
      implicit none
      character*(*)                     , intent(IN)  :: F_var_S,F_hgrid_S
      integer                           , intent(OUT) :: F_nka
      integer, dimension(:    ), pointer, intent(OUT) :: F_ip1
      real   , dimension(:,:,:), pointer, intent(OUT) :: F_dest
      End function inp_read
End Interface

#include "cstv.cdk"
#include "glb_ld.cdk"

      integer nka,istat
      integer, dimension (:    ), pointer :: ip1_list
      real   , dimension (:,:,:), allocatable, target :: srclev
      real   , dimension (:,:,:), pointer :: wrkr,ptr3d
      real   , dimension (:,:  ), pointer :: p0
      real, target :: dstlev(l_minx:l_maxx,l_miny:l_maxy,G_nk)
!
!---------------------------------------------------------------------
!
      nullify (ip1_list, wrkr)

      inp_get= inp_read ( F_var_S, F_hgrid_S, wrkr, ip1_list, nka )

      if (inp_get .lt. 0) then
         if (associated(ip1_list)) deallocate (ip1_list)
         if (associated(wrkr    )) deallocate (wrkr    )
         return
      endif

      allocate ( srclev(l_minx:l_maxx,l_miny:l_maxy,nka ))

      p0    => F_sfc_src(1:l_ni,1:l_nj)
      ptr3d =>    srclev(1:l_ni,1:l_nj,1:nka)
      istat= vgd_levels (F_vgd_src, ip1_list, ptr3d, p0, in_log=.true.)

      p0    => F_sfc_dst(1:l_ni,1:l_nj)
      ptr3d =>    dstlev(1:l_ni,1:l_nj,1:G_nk)
      istat= vgd_levels (F_vgd_dst, F_ver_ip1,ptr3d, p0, in_log=.true.)

      call vertint ( F_dest,dstlev,G_nk, wrkr,srclev,nka, &
                     l_minx,l_maxx,l_miny,l_maxy, 1,l_ni,1,l_nj,&
                     'cubic', .false. )

      deallocate (ip1_list,wrkr,srclev)
!
!---------------------------------------------------------------------
!
      return
      end
