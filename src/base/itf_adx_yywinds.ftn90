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

subroutine itf_adx_yywinds2 (F_u,F_v,i0u,j0u,inu,jnu,i0v,j0v,inv,jnv, &
                             minx,maxx,miny,maxy,nk)
  implicit none
#include <arch_specific.hf>
  !@objective Exchange U and V between Yin and Yang
  !@arguments

  integer :: i0u,j0u,inu,jnu,i0v,j0v,inv,jnv,&
              minx,maxx,miny,maxy,nk
  real F_u(minx:maxx,miny:maxy,l_nk), F_v(minx:maxx,miny:maxy,l_nk)

  !@revisions
  ! v4_40 - Qaddouri/Lee      - Yin-Yang uses global range
  ! v4.7  - Gaudreault S.     - Removing wind images
  !*@/

#include "glb_ld.cdk"

  integer :: i,j,k
  real temp_u (l_minx:l_maxx,l_miny:l_maxy,l_nk)
  real temp_u1(l_minx:l_maxx,l_miny:l_maxy,l_nk)
  real temp_v (l_minx:l_maxx,l_miny:l_maxy,l_nk)
  real temp_v1(l_minx:l_maxx,l_miny:l_maxy,l_nk)

  !---------------------------------------------------------------------

  temp_u=0.
  temp_v=0.

  do k=1,l_nk
     do j= j0u, jnu
        do i= i0u, inu
           temp_u (i,j,k)= F_u(i,j,k)
        enddo
     enddo
     do j= j0v, jnv
        do i= i0v, inv
           temp_v (i,j,k)= F_v(i,j,k)
        enddo
     enddo
  end do

  call rpn_comm_xch_halo(temp_u,l_minx,l_maxx,l_miny,l_maxy,&
       l_ni,l_nj,G_nk, G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
  call rpn_comm_xch_halo(temp_v,l_minx,l_maxx,l_miny,l_maxy,&
       l_ni,l_nj,G_nk, G_halox,G_haloy,G_periodx,G_periody,l_ni,0)

  temp_u1=temp_u
  temp_v1=temp_v

  call  yyg_scaluv ( temp_u1,temp_u,temp_v1,temp_v,&
                      l_minx,l_maxx,l_miny,l_maxy,G_nk )

  if (l_west) then
     do k= 1, G_nk
        do j= 1,l_nj
           do i= 1, 2
              F_u(i,j,k)= temp_u1 (i,j,k)
           enddo
        enddo
     enddo
  endif
  if (l_east) then
     do k= 1, G_nk
        do j= 1,l_nj
           F_u(l_niu,j,k)= temp_u1 (l_niu,j,k)
           F_u(l_ni ,j,k)= temp_u1 (l_ni ,j,k)
        enddo
     enddo
  endif
  if (l_south) then
     do k= 1, G_nk
        do i= 1, l_ni
           do j= 1, 2
              F_v(i,j,k)=temp_v1 (i,j,k)
           enddo
        enddo
     enddo
  endif
  if (l_north) then
     do k= 1, G_nk
        do i= 1, l_ni
           F_v(i,l_njv,k)= temp_v1(i,l_njv,k)
           F_v(i,l_nj ,k)= temp_v1(i,l_nj ,k)
        enddo
     enddo
  endif

  !---------------------------------------------------------------------
  return
end subroutine itf_adx_yywinds2
