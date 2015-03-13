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

!**s/r e_target3df - defines destination grids for 3df files
!
      subroutine e_target3df (F_grda_S, F_nia, F_nja, &
                              F_ig1a, F_ig2a, F_ig3a, F_ig4a)
      implicit none
#include <arch_specific.hf>
!
      character*1 F_grda_S
      integer F_nia,F_nja,F_ig1a, F_ig2a, F_ig3a, F_ig4a
!author 
!    Michel Desgagne -   Winter 2012
!
!revision
! v4_50 - Desgagne M.      - initial version
!
#include "e_anal.cdk"
#include "e_grids.cdk"
#include "e_fu.cdk"
#include "pilot1.cdk"
#include "hgc.cdk"
#include "e_grdc.cdk"

      logical, external :: samegrid_file
      integer, external :: cascindx, ezgdef_fmem

      logical same_grid_L
      integer i,err,ni_target,nj_target,ni,nj
      real,   dimension(:), allocatable :: xpx,xpxu,ypx,ypxv
      real*8  dx_8, dy_8, debut_8, pos_8, deg2rad_8, offsetx, offsety
!
! ---------------------------------------------------------------------
!
      if (Pil_bmf_L .or. (.not.LAM)) return

      ni= F_nia ; nj= F_nja
      call e_3dfreso ( F_grda_S, dx_8,dy_8, ni,nj, &
                       F_ig1a,F_ig2a,F_ig3a,F_ig4a )

      same_grid_L= samegrid_file ( e_fu_anal, F_ig1a, F_ig2a, F_ig3a ,&
                           Hgc_ig1ro, Hgc_ig2ro, Hgc_ig3ro, Hgc_ig4ro,&
                                                      xfi, yfi, ni, nj)

      ni_target = nint((xfi(nifi) - xfi(1))/dx_8) + 1
      nj_target = nint((yfi(njfi) - yfi(1))/dy_8) + 1

      offsetx= 0.
      offsety= 0.
      if (.not.same_grid_L) then
         ni_target = ni_target + 12
         nj_target = nj_target + 12
         offsetx   = dx_8*5.0
         offsety   = dy_8*5.0
      endif

      deg2rad_8 = acos( -1.0d0 )/180.0d0

      allocate (xpx(ni_target),ypx(nj_target),xpxu(ni_target-1),ypxv(nj_target-1))
      if (associated( xg_8)) deallocate( xg_8)
      if (associated( yg_8)) deallocate( yg_8)
      if (associated(ygv_8)) deallocate(ygv_8)
      allocate (xg_8(ni_target), yg_8(nj_target), ygv_8(nj_target-1))

      debut_8= xfi(1) - offsetx
      do i=1,ni_target
         pos_8  = debut_8 + (i-1)*dx_8
         xpx (i)= pos_8
         xg_8(i)= pos_8 * deg2rad_8
      end do

      debut_8= yfi(1) - offsety
      do i=1,nj_target
         pos_8  = debut_8 + (i-1)*dy_8
         ypx (i)= pos_8
         yg_8(i)= pos_8 * deg2rad_8
      end do

      do i=1,ni_target-1
         xpxu(i)= 0.5 * ( xpx(i) + xpx(i+1) )
      enddo
      do i=1,nj_target-1
         ypxv(i)= 0.5 * ( ypx(i) + ypx(i+1) )
         ygv_8(i) = ypxv(i) * deg2rad_8
      enddo

      err = cascindx ( e_grdc_gid,e_grdc_gif,e_grdc_gjd,e_grdc_gjf, &
                       xfi,yfi,nifi,njfi,xpx,ypx,ni_target,nj_target )

      if ((err.lt.0).and.(same_grid_L)) then
         e_grdc_gid= 1
         e_grdc_gif= ni_target
         e_grdc_gjd= 1
         e_grdc_gjf= nj_target
      endif

      e_grdc_ni = e_grdc_gif-e_grdc_gid+1
      e_grdc_nj = e_grdc_gjf-e_grdc_gjd+1

      dstf_gid = ezgdef_fmem (e_grdc_ni  , e_grdc_nj  , 'Z', 'E', Hgc_ig1ro, &
           Hgc_ig2ro, Hgc_ig3ro, Hgc_ig4ro, xpx (e_grdc_gid), ypx (e_grdc_gjd))
      dstu_gid = ezgdef_fmem (e_grdc_ni-1, e_grdc_nj  , 'Z', 'E', Hgc_ig1ro, &
           Hgc_ig2ro, Hgc_ig3ro, Hgc_ig4ro, xpxu(e_grdc_gid), ypx (e_grdc_gjd))
      dstv_gid = ezgdef_fmem (e_grdc_ni  , e_grdc_nj-1, 'Z', 'E', Hgc_ig1ro, &
           Hgc_ig2ro, Hgc_ig3ro, Hgc_ig4ro,  xpx(e_grdc_gid), ypxv(e_grdc_gjd))

      deallocate(xpx,ypx,xpxu,ypxv,stat=err)

      F_nia= -1 ; F_nja= -1
!
! ---------------------------------------------------------------------
!
      return
      end subroutine e_target3df
