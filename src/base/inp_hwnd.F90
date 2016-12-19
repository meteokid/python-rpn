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

      subroutine inp_hwnd ( F_u,F_v, F_vgd_src,F_vgd_dst, F_stag_L    ,&
                            F_ssqr,F_ssur,F_ssvr, F_ssq0,F_ssu0,F_ssv0,&
                            Minx,Maxx,Miny,Maxy, F_nk )
      use vertical_interpolation, only: vertint2
      use vGrid_Descriptors
      use inp_base, only: inp_read_uv
      implicit none
#include <arch_specific.hf>

      logical                , intent(IN)  :: F_stag_L
      integer                , intent(IN)  :: Minx,Maxx,Miny,Maxy, F_nk
      type(vgrid_descriptor) , intent(IN)  :: F_vgd_src, F_vgd_dst
      real, dimension(Minx:Maxx,Miny:Maxy     ), target, intent(IN) :: &
                             F_ssqr,F_ssur,F_ssvr, F_ssq0,F_ssu0,F_ssv0
      real, dimension(Minx:Maxx,Miny:Maxy,F_nk), intent(OUT) :: F_u, F_v

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

      if (F_stag_L) then
         call inp_read_uv ( ur, vr, 'UV' , ip1_list, nka )
      else
         call inp_read_uv ( ur, vr, 'Q ' , ip1_list, nka )
      endif

      allocate ( srclev(l_minx:l_maxx,l_miny:l_maxy,nka) )

      if (F_stag_L) then

         p0    => F_ssur(1:l_ni,1:l_nj)
         ptr3d => srclev(1:l_ni,1:l_nj,1:nka)
         istat= vgd_levels ( F_vgd_src, ip1_list(1:nka), ptr3d,&
                             p0, in_log=.true. )
         nullify (p0, ptr3d)

         p0    => F_ssu0(1:l_ni,1:l_nj)
         ptr3d =>    dstlev(1:l_ni,1:l_nj,1:G_nk)
         istat= vgd_levels  (F_vgd_dst, Ver_ip1%m(1:G_nk), ptr3d, p0,&
                             in_log=.true. )
         nullify (p0, ptr3d)

         call vertint2 ( F_u,dstlev,G_nk, ur,srclev,nka,&
                         l_minx,l_maxx,l_miny,l_maxy, 1,l_ni,1,l_nj )

         p0    => F_ssvr(1:l_ni,1:l_nj)
         ptr3d =>    srclev(1:l_ni,1:l_nj,1:nka)
         istat= vgd_levels ( F_vgd_src, ip1_list(1:nka), ptr3d,&
                             p0, in_log=.true. )

         p0    => F_ssv0(1:l_ni,1:l_nj)
         ptr3d =>    dstlev(1:l_ni,1:l_nj,1:G_nk)
         istat= vgd_levels ( F_vgd_dst, Ver_ip1%m(1:G_nk), ptr3d, p0,&
                             in_log=.true. )

         call vertint2 ( F_v,dstlev,G_nk, vr,srclev,nka,&
                         l_minx,l_maxx,l_miny,l_maxy, 1,l_ni,1,l_nj )

      else

         p0    => F_ssqr(1:l_ni,1:l_nj)
         ptr3d => srclev(1:l_ni,1:l_nj,1:nka)
         istat= vgd_levels ( F_vgd_src, ip1_list(1:nka), ptr3d,&
                             p0, in_log=.true. )
         nullify (p0, ptr3d)

         p0    => F_ssq0(1:l_ni,1:l_nj)
         ptr3d =>    dstlev(1:l_ni,1:l_nj,1:G_nk)
         istat= vgd_levels  (F_vgd_dst, Ver_ip1%m(1:G_nk), ptr3d, p0,&
                            in_log=.true. )
         nullify (p0, ptr3d)

         call vertint2 ( F_u,dstlev,G_nk, ur,srclev,nka,&
                         l_minx,l_maxx,l_miny,l_maxy, 1,l_ni,1,l_nj )
         call vertint2 ( F_v,dstlev,G_nk, vr,srclev,nka,&
                         l_minx,l_maxx,l_miny,l_maxy, 1,l_ni,1,l_nj )

      endif

      deallocate (ip1_list,ur,vr,srclev)
      nullify (ip1_list, ur, vr, p0, ptr3d)
!
!---------------------------------------------------------------------
!
      return
      end
