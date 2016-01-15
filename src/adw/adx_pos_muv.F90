!-------------------------------------- LICENCE BEGIN ------------------------------------
!Environment Canada - Atmospheric Science and Technology License/Disclaimer,
!                     version 3; Last Modified: May 7, 2008.
!This is free but copyrighted software; you can use/redistribute/modify it under the terms
!of the Environment Canada - Atmospheric Science and Technology License/Disclaimer
!version 3 or (at your option) any later version that should be found at:
!http://collaboration.cmc.ec.gc.ca/science/rpn.comm/license.html
!
!This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
!without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
!See the above mentioned License/Disclaimer for more details.
!You should have received a copy of the License/Disclaimer along with this software;
!if not, you can write to: EC-RPN COMM Group, 2121 TransCanada, suite 500, Dorval (Quebec),
!CANADA, H9P 1J3; or send e-mail to service.rpn@ec.gc.ca
!-------------------------------------- LICENCE END --------------------------------------
!**s/r adx_pos_muv - horizontal interpolation of upstream momentum position
!
#include "constants.h"
!
subroutine adx_pos_muv ( F_xmu, F_ymu, F_zmu, F_xmv, F_ymv, F_zmv, &
                         F_xm, F_ym, F_zm, &
                         F_ni,F_nj,F_k0,F_nk,i0,in,j0,jn)
!
   implicit none
#include <arch_specific.hf>
!
   integer :: F_ni,F_nj,F_k0,F_nk,i0,in,j0,jn
   real, dimension(F_ni,F_nj,F_nk) :: F_xmu,F_ymu,F_zmu
   real, dimension(F_ni,F_nj,F_nk) :: F_xmv,F_ymv,F_zmv
   real, dimension(F_ni,F_nj,F_nk) :: F_xm,F_ym,F_zm
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
!
!implicits
#include "adx_dims.cdk"
#include "adx_grid.cdk"
#include "adx_dyn.cdk"
#include "adx_interp.cdk"
#include "inuvl.cdk"
#include "ver.cdk"
!***********************************************************************
      integer i,j,k,i0u,inu,j0v,jnv
      real*8 aa, bb, cc, dd
      real :: ztop_bound, zbot_bound
      real, pointer, dimension(:,:,:) :: pxh,pyh,pzh
      logical,save :: done = .false.

!
      nullify (pxh,pyh,pzh)
!***********************************************************************

      ztop_bound=Ver_z_8%m(0)
      zbot_bound=Ver_z_8%m(F_nk+1)
      
      allocate(pxh(-1:adx_mlni+2,-1:adx_mlnj+2,adx_lnk)) 
      allocate(pyh(-1:adx_mlni+2,-1:adx_mlnj+2,adx_lnk))
      allocate(pzh(-1:adx_mlni+2,-1:adx_mlnj+2,adx_lnk)) 

      do k=F_k0,F_nk
      do j=1, adx_mlnj
      do i=1, adx_mlni
         pxh(i,j,k)=F_xm(i,j,k)
         pyh(i,j,k)=F_ym(i,j,k)
         pzh(i,j,k)=F_zm(i,j,k)
      enddo
      enddo
      enddo
      call rpn_comm_xch_halo(pxh,-1,adx_mlni+2,-1,adx_mlnj+2,adx_mlni,adx_mlnj,adx_lnk,2,2,.false.,.false.,adx_mlni,0)
      call rpn_comm_xch_halo(pyh,-1,adx_mlni+2,-1,adx_mlnj+2,adx_mlni,adx_mlnj,adx_lnk,2,2,.false.,.false.,adx_mlni,0)
      call rpn_comm_xch_halo(pzh,-1,adx_mlni+2,-1,adx_mlnj+2,adx_mlni,adx_mlnj,adx_lnk,2,2,.false.,.false.,adx_mlni,0)

      if(.not.done) then
         F_xmu=F_xm;F_ymu=F_ym;F_zmu=F_zm;F_xmv=F_xm;F_ymv=F_ym;F_zmv=F_zm
         done = .true.
      endif

      i0u=i0
      inu=in
      j0v=j0
      jnv=jn
      if(adx_is_west) i0u=i0+1
      if(adx_is_east) inu=in-2
      if(adx_is_south) j0v=j0+1
      if(adx_is_north) jnv=jn-2

      aa=-0.0625d0
      bb=+0.5625d0
      cc=adx_dlx_8(adx_mlni/2)*0.5d0

      do k=F_k0,F_nk
         do j=j0,jn
         do i=i0u,inu
            F_xmu(i,j,k) =  aa*(pxh(i-1,j,k)+pxh(i+2,j,k)) &
                          + bb*(pxh(i  ,j,k)+pxh(i+1,j,k)) - cc
            F_ymu(i,j,k) =  aa*(pyh(i-1,j,k)+pyh(i+2,j,k)) &
                          + bb*(pyh(i  ,j,k)+pyh(i+1,j,k))
            F_zmu(i,j,k) =  aa*(pzh(i-1,j,k)+pzh(i+2,j,k)) &
                          + bb*(pzh(i  ,j,k)+pzh(i+1,j,k))
	    F_zmu(i,j,k) =  min(zbot_bound,max(F_zmu(i,j,k),ztop_bound))
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
           F_zmu(i,j,k) =  min(zbot_bound,max(F_zmu(i,j,k),ztop_bound))
         enddo
         enddo

      enddo

      deallocate(pxh,pyh,pzh)
!
      
return
end subroutine adx_pos_muv
