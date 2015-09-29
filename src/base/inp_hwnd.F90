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

!**s/r inp_hwnd - Read horizontal winds UU,VV F valid at F_datev
!                 and perform vectorial horizontal interpolation
!                 and  vertical interpolation to momentum levels

      subroutine inp_hwnd ( F_u, F_v, F_vgd_src, F_vgd_dst,&
                            F_ssur, F_ssvr, F_ssu0, F_ssv0,&
                            Minx,Maxx,Miny,Maxy, F_nk )
      use vGrid_Descriptors
      implicit none
#include <arch_specific.hf>

      integer                , intent(IN)  :: Minx,Maxx,Miny,Maxy, F_nk
      type(vgrid_descriptor) , intent(IN)  :: F_vgd_src, F_vgd_dst
      real, dimension(Minx:Maxx,Miny:Maxy     ), target, &
                          intent(IN) :: F_ssur, F_ssvr, F_ssu0, F_ssv0
      real, dimension(Minx:Maxx,Miny:Maxy,F_nk), intent(OUT):: F_u, F_v

Interface
      subroutine inp_read_uv ( F_u, F_v, F_ip1, F_nka )
      implicit none
      integer                           , intent(OUT) :: F_nka
      integer, dimension(:    ), pointer, intent(OUT) :: F_ip1
      real   , dimension(:,:,:), pointer, intent(OUT) :: F_u, F_v
      End Subroutine inp_read_uv
End Interface

#include "glb_ld.cdk"
#include "ver.cdk"

      integer nka,istat
      integer, dimension (:    ), pointer :: ip1_list
      real   , dimension (:,:,:), allocatable, target :: srclev
      real   , dimension (:,:,:), pointer :: ur,vr,ptr3d
      real   , dimension (:,:  ), pointer :: p0
      real, target :: dstlev(l_minx:l_maxx,l_miny:l_maxy,G_nk)
!
!---------------------------------------------------------------------
!
      nullify (ip1_list, ur, vr, p0, ptr3d)

      call inp_read_uv ( ur, vr, ip1_list, nka )

      if (nka .lt. 1) then
         if (associated(ip1_list)) deallocate (ip1_list)
         if (associated(ur      )) deallocate (ur      )
         if (associated(vr      )) deallocate (vr      )
         return
      endif

      allocate ( srclev(l_minx:l_maxx,l_miny:l_maxy,nka ))

      p0    => F_ssur(1:l_ni,1:l_nj)
      ptr3d => srclev(1:l_ni,1:l_nj,1:nka)
      istat= vgd_levels (F_vgd_src, ip1_list, ptr3d, p0, in_log=.true.)
      nullify (p0, ptr3d)

      p0    => F_ssu0(1:l_ni,1:l_nj)
      ptr3d =>    dstlev(1:l_ni,1:l_nj,1:G_nk)
      istat= vgd_levels  (F_vgd_dst, Ver_ip1%m(1:G_nk), ptr3d, p0, &
                          in_log=.true. )
      nullify (p0, ptr3d)

      call vertint ( F_u,dstlev,G_nk, ur,srclev,nka, &
                     l_minx,l_maxx,l_miny,l_maxy, 1,l_ni,1,l_nj,&
                     'cubic', .false. )

      p0    => F_ssvr(1:l_ni,1:l_nj)
      ptr3d =>    srclev(1:l_ni,1:l_nj,1:nka)
      istat= vgd_levels (F_vgd_src, ip1_list, ptr3d, p0, in_log=.true.)

      p0    => F_ssv0(1:l_ni,1:l_nj)
      ptr3d =>    dstlev(1:l_ni,1:l_nj,1:G_nk)
      istat= vgd_levels ( F_vgd_dst, Ver_ip1%m(1:G_nk), ptr3d, p0, &
                          in_log=.true. )

      call vertint ( F_v,dstlev,G_nk, vr,srclev,nka, &
                     l_minx,l_maxx,l_miny,l_maxy, 1,l_ni,1,l_nj,&
                     'cubic', .false. )

      deallocate (ip1_list,ur,vr,srclev)
      nullify (ip1_list, ur, vr, p0, ptr3d)
!
!---------------------------------------------------------------------
!
      return
      end
