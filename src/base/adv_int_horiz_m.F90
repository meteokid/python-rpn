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
!

  subroutine adv_int_horiz_m ( F_xmu, F_ymu, F_zmu, F_xmv, F_ymv, F_zmv,&
                               F_xm , F_ym , F_zm                      ,&
                               F_ni,F_nj, F_nk,F_k0,i0,in,j0,jn)
!
  implicit none

#include <arch_specific.hf>
!
   integer :: F_ni,F_nj,F_nk
   integer :: F_k0,i0,in,j0,jn
	real, dimension(F_ni,F_nj,F_nk),intent(out) :: F_xmu,F_ymu,F_zmu
	real, dimension(F_ni,F_nj,F_nk),intent(out) :: F_xmv,F_ymv,F_zmv
	real, dimension(F_ni,F_nj,F_nk),intent(in) :: F_xm,F_ym,F_zm
!
!authors
!     A. Plante & C. Girard
!
!revision
!
!object
!
!arguments
!______________________________________________________________________
!              |                                                 |     |
! NAME         | DESCRIPTION                                     | I/O |
!--------------|-------------------------------------------------|-----|
!              |                                                 |     |
! F_xt         | upwind longitudes for themodynamic level        |  o  |
! F_yt         | upwind latitudes for themodynamic level         |  o  |
! F_zt         | upwind height for themodynamic level            |  o  |
! F_xm         | upwind longitudes for momentum level            |  i  |
! F_ym         | upwind latitudes for momentum level             |  i  |
! F_zm         | upwind height for momentum level                |  i  |
!______________|_________________________________________________|_____|
!                      |
!----------------------------------------------------------------------

#include "glb_ld.cdk"
#include "adv_interp.cdk"


     integer i,j,k,i0u,inu,j0v,jnv
     real*8 aa, bb, cc, dd
	  real, pointer, dimension(:,:,:) :: pxh,pyh,pzh
	  logical,save :: done = .false.

!----------------------------------------------------------------------
     
     nullify (pxh,pyh,pzh)

    allocate(pxh(-1:F_ni+2,-1:F_nj+2,F_nk))
    allocate(pyh(-1:F_ni+2,-1:F_nj+2,F_nk))
    allocate(pzh(-1:F_ni+2,-1:F_nj+2,F_nk))

    do k=F_k0,F_nk
    do j=1, F_nj
    do i=1, F_ni
       pxh(i,j,k)=F_xm(i,j,k)
       pyh(i,j,k)=F_ym(i,j,k)
       pzh(i,j,k)=F_zm(i,j,k)
    enddo
    enddo
    enddo


    call rpn_comm_xch_halo(pxh,-1,F_ni+2,-1,F_nj+2,F_ni,F_nj,F_nk,2,2,.false.,.false.,F_ni,0)
    call rpn_comm_xch_halo(pyh,-1,F_ni+2,-1,F_nj+2,F_ni,F_nj,F_nk,2,2,.false.,.false.,F_ni,0)
    call rpn_comm_xch_halo(pzh,-1,F_ni+2,-1,F_nj+2,F_ni,F_nj,F_nk,2,2,.false.,.false.,F_ni,0)

    if(.not.done) then
    F_xmu=F_xm;F_ymu=F_ym;F_zmu=F_zm;F_xmv=F_xm;F_ymv=F_ym;F_zmv=F_zm
    done = .true.
    endif

    i0u=i0
    inu=in
    j0v=j0
    jnv=jn
    if(l_west) i0u=i0+1
    if(l_east) inu=in-2
    if(l_south) j0v=j0+1
    if(l_north) jnv=jn-2

    aa=-0.0625d0
    bb=+0.5625d0
    cc=adv_dlx_8(F_ni/2)*0.5d0

    do k=F_k0,F_nk
      do j=j0,jn
      do i=i0u,inu
          F_xmu(i,j,k) =  aa*(pxh(i-1,j,k)+pxh(i+2,j,k)) &
                        + bb*(pxh(i  ,j,k)+pxh(i+1,j,k)) - cc
          F_ymu(i,j,k) =  aa*(pyh(i-1,j,k)+pyh(i+2,j,k)) &
                        + bb*(pyh(i  ,j,k)+pyh(i+1,j,k))
          F_zmu(i,j,k) =  aa*(pzh(i-1,j,k)+pzh(i+2,j,k)) &
                        + bb*(pzh(i  ,j,k)+pzh(i+1,j,k))
      end do
      end do
    
      do j=j0v,jnv
      do i=i0,in

          F_xmv(i,j,k) =  aa*(pxh(i,j-1,k)+pxh(i,j+2,k)) &
                         + bb*(pxh(i,j  ,k)+pxh(i,j+1,k))

          F_ymv(i,j,k) =  aa*(pyh(i,j-1,k)+pyh(i,j+2,k)) &
                         + bb*(pyh(i,j  ,k)+pyh(i,j+1,k)) - cc

          F_zmv(i,j,k) =  aa*(pzh(i,j-1,k)+pzh(i,j+2,k)) &
                         + bb*(pzh(i,j  ,k)+pzh(i,j+1,k))
      enddo
      enddo

    enddo

 deallocate(pxh,pyh,pzh)

end subroutine adv_int_horiz_m
