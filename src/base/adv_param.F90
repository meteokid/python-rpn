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
      subroutine adv_param ()
      implicit none
#include <arch_specific.hf>

   !@objective set 1-D interpolation of grid reflexion across the pole 
   !@author  alain patoine
   !@revisions
   !*@/

#include "glb_ld.cdk"
#include "adv_grid.cdk"
#include "adv_dims.cdk"
#include "adv_interp.cdk"
   
      real*8, parameter :: LARGE_8 = 1.D20
      real*8, parameter :: FRAC1OV6_8 = 1.D0/6.D0

      integer :: i, j, k, ij, pnerr, trj_i_off, nij, n, istat, &
                 istat2, ind, i0, j0, k0, pnx, pny, pnz, nkm, nkt

      real*8 :: ra,rb,rc,rd
      real*8 :: prhxmn, prhymn, prhzmn, dummy, pdfi

      real*8 :: whx(G_ni+2*adv_halox)
      real*8 :: why(G_nj+2*adv_haloy)

      real*8 :: qzz_m_8(3*l_nk), qzi_m_8(4*l_nk)
      real*8 :: qzz_t_8(3*l_nk), qzi_t_8(4*l_nk)

      real*8, dimension(:),allocatable :: whzt,whzm

#if !defined(TRIPROD)
#define TRIPROD(za,zb,zc,zd) ((za-zb)*(za-zc)*(za-zd))
#endif
!
!     ---------------------------------------------------------------
!   
      nkm=l_nk ; nkt=l_nk
      
      allocate( whzm(0:nkm+1), whzt(0:nkt+1))
      allocate( &
      adv_xbc_8(G_ni+2*adv_halox), & ! (xc-xb)     along x
      adv_xabcd_8(G_ni+2*adv_halox), & ! triproducts along x
      adv_xbacd_8(G_ni+2*adv_halox), &
      adv_xcabd_8(G_ni+2*adv_halox), &
      adv_xdabc_8(G_ni+2*adv_halox), &
      adv_ybc_8(G_nj+2*adv_haloy), & ! (yc-yb)     along y 
      adv_yabcd_8(G_nj+2*adv_haloy), & ! triproducts along y
      adv_ybacd_8(G_nj+2*adv_haloy), &
      adv_ycabd_8(G_nj+2*adv_haloy), &
      adv_ydabc_8(G_nj+2*adv_haloy) )

      do i = adv_gminx+1,adv_gmaxx-2
         ra = adv_xg_8(i-1)
         rb = adv_xg_8(i)
         rc = adv_xg_8(i+1)
         rd = adv_xg_8(i+2)
         adv_xabcd_8(adv_halox+i) = 1.D0/TRIPROD(ra,rb,rc,rd)
         adv_xbacd_8(adv_halox+i) = 1.D0/TRIPROD(rb,ra,rc,rd)
         adv_xcabd_8(adv_halox+i) = 1.D0/TRIPROD(rc,ra,rb,rd)
         adv_xdabc_8(adv_halox+i) = 1.D0/TRIPROD(rd,ra,rb,rc)
      enddo

      do i = adv_gminx,adv_gmaxx-1
         rb = adv_xg_8(i)
         rc = adv_xg_8(i+1)
         adv_xbc_8(adv_halox+i) = 1.D0/(rc-rb)
      enddo

      do j = adv_gminy+1,adv_gmaxy-2
         ra = adv_yg_8(j-1)
         rb = adv_yg_8(j)
         rc = adv_yg_8(j+1)
         rd = adv_yg_8(j+2)
         adv_yabcd_8(adv_haloy+j) = 1.D0/TRIPROD(ra,rb,rc,rd)
         adv_ybacd_8(adv_haloy+j) = 1.D0/TRIPROD(rb,ra,rc,rd)
         adv_ycabd_8(adv_haloy+j) = 1.D0/TRIPROD(rc,ra,rb,rd)
         adv_ydabc_8(adv_haloy+j) = 1.D0/TRIPROD(rd,ra,rb,rc)
      enddo

      do j = adv_gminy,adv_gmaxy-1
         rb = adv_yg_8(j)
         rc = adv_yg_8(j+1)
         adv_ybc_8(adv_haloy+j) = 1.D0/(rc-rb)
      enddo

      trj_i_off = 0

      adv_x00_8 = adv_xg_8(adv_gminx)
      adv_y00_8 = adv_yg_8(adv_gminy)

      prhxmn = LARGE_8
      prhymn = LARGE_8
      prhzmn = LARGE_8

      do i = adv_gminx,adv_gmaxx-1
         whx(adv_halox+i) = adv_xg_8(i+1) - adv_xg_8(i)
         prhxmn = min(whx(adv_halox+i), prhxmn)
      enddo

      do j = adv_gminy,adv_gmaxy-1
         why(adv_haloy+j) = adv_yg_8(j+1) - adv_yg_8(j)
         prhymn = min(why(adv_haloy+j), prhymn)
      enddo

! Prepare zeta on super vertical grid
      
      whzt(0    ) = 1.0
      whzt(nkt  ) = 1.0
      whzt(nkt+1) = 1.0
      do k = 1,nkt-1
         whzt(k) = adv_verZ_8%t(k+1) - adv_verZ_8%t(k)
         prhzmn = min(whzt(k), prhzmn)
      enddo

      whzm(0    ) = 1.0
      whzm(nkm  ) = 1.0
      whzm(nkm+1) = 1.0
      do k = 1,nkm-1
         whzm(k) = adv_verZ_8%m(k+1) - adv_verZ_8%m(k)
         prhzmn = min(whzm(k), prhzmn)
      enddo
      
      adv_ovdx_8 = 1.0d0/prhxmn
      adv_ovdy_8 = 1.0d0/prhymn
      adv_ovdz_8 = 1.0d0/prhzmn

      pnx = int(1.0+(adv_xg_8(adv_gmaxx)-adv_x00_8)   *adv_ovdx_8)
      pny = int(1.0+(adv_yg_8(adv_gmaxy)-adv_y00_8)   *adv_ovdy_8)
      pnz = nint(1.0+(adv_verZ_8%m(nkm+1)-adv_verZ_8%m(0))*adv_ovdz_8)

      allocate( &
      adv_lcx(pnx), &
      adv_lcy(pny), &
      adv_bsx_8(G_ni+2*adv_halox), &
      adv_dlx_8(G_ni+2*adv_halox), &     
      adv_bsy_8(G_nj+2*adv_haloy), &
      adv_dly_8(G_nj+2*adv_haloy), &      
      adv_lcz%t(pnz), &
      adv_lcz%m(pnz),  &
      adv_bsz_8%t(0:nkt-1)  , &
      adv_bsz_8%m(0:nkm-1), &
      adv_dlz_8%t(-1:nkt) , &
      adv_dlz_8%m(-1:nkm))

      i0 = 1
      do i=1,pnx
         pdfi = adv_xg_8(adv_gminx) + (i-1) * prhxmn
         if (pdfi > adv_xg_8(i0+1-adv_halox)) i0 = min(G_ni+2*adv_halox-1,i0+1)
         adv_lcx(i) = i0
      enddo
      do i = adv_gminx,adv_gmaxx-1
         adv_dlx_8(adv_halox+i) =       whx(adv_halox+i)
      enddo
      do i = adv_gminx,adv_gmaxx
         adv_bsx_8(adv_halox+i) = adv_xg_8(i)
      enddo

      j0 = 1
      do j = 1,pny
         pdfi = adv_yg_8(adv_gminy) + (j-1) * prhymn
         if (pdfi > adv_yg_8(j0+1-adv_haloy)) j0 = min(G_nj+2*adv_haloy-1,j0+1)
         adv_lcy(j) = j0
      enddo
      do j = adv_gminy,adv_gmaxy-1
         adv_dly_8(adv_haloy+j) =       why(adv_haloy+j)
         

      enddo
      do j = adv_gminy,adv_gmaxy
         adv_bsy_8(adv_haloy+j) = adv_yg_8(j)

      enddo

      k0 = 1

      do k = 1,pnz
         pdfi = adv_verZ_8%m(0) + (k-1) * prhzmn
         if (pdfi > adv_verZ_8%t(k0+1)) k0 = min(nkt-2, k0+1)
         adv_lcz%t(k) = k0
      enddo
      do k = 0,nkt+1            !! warning note the shift in k !!
         adv_dlz_8%t(k-1) =       whzt(k)

      enddo
      do k = 1,nkt
         adv_bsz_8%t(k-1) = adv_verZ_8%t(k)

      enddo

      k0 = 1
      do k = 1,pnz
         pdfi = adv_verZ_8%m(0) + (k-1) * prhzmn
         if (pdfi > adv_verZ_8%m(k0+1)) k0 = min(nkm-2, k0+1)
         adv_lcz%m(k) = k0
      enddo
      do k = 0,nkm+1            !! warning note the shift in k !!
         adv_dlz_8%m(k-1) =       whzm(k)
      enddo
      do k = 1,nkm
         adv_bsz_8%m(k-1) = adv_verZ_8%m(k)
      enddo
           
      allocate( &
      adv_zbc_8%t(nkt),   adv_zbc_8%m(nkm),    &
      adv_zabcd_8%t(nkt), adv_zabcd_8%m(nkm),  &
      adv_zbacd_8%t(nkt), adv_zbacd_8%m(nkm),  &
      adv_zcabd_8%t(nkt), adv_zcabd_8%m(nkm),  &
      adv_zdabc_8%t(nkt), adv_zdabc_8%m(nkm))

      do k = 2,nkm-2
         ra = adv_verZ_8%m(k-1)
         rb = adv_verZ_8%m(k)
         rc = adv_verZ_8%m(k+1)
         rd = adv_verZ_8%m(k+2)

         adv_zabcd_8%m(k) = 1.0/TRIPROD(ra,rb,rc,rd)
         adv_zbacd_8%m(k) = 1.0/TRIPROD(rb,ra,rc,rd)
         adv_zcabd_8%m(k) = 1.0/TRIPROD(rc,ra,rb,rd)
         adv_zdabc_8%m(k) = 1.0/TRIPROD(rd,ra,rb,rc)
      enddo

      do k = 2,nkt-2
         ra = adv_verZ_8%t(k-1)
         rb = adv_verZ_8%t(k)
         rc = adv_verZ_8%t(k+1)
         rd = adv_verZ_8%t(k+2)

         adv_zabcd_8%t(k) = 1.0/TRIPROD(ra,rb,rc,rd)
         adv_zbacd_8%t(k) = 1.0/TRIPROD(rb,ra,rc,rd)
         adv_zcabd_8%t(k) = 1.0/TRIPROD(rc,ra,rb,rd)
         adv_zdabc_8%t(k) = 1.0/TRIPROD(rd,ra,rb,rc)
      enddo
     
      do k = 1,nkm-1
         rb = adv_verZ_8%m(k)
         rc = adv_verZ_8%m(k+1)
         adv_zbc_8%m(k) = 1.0/(rc-rb)
      enddo

      do k = 1,nkt-1
         rb = adv_verZ_8%t(k)
         rc = adv_verZ_8%t(k+1)
         adv_zbc_8%t(k) = 1.0/(rc-rb)
      enddo
      
      deallocate(whzt, whzm)
!     
!     ---------------------------------------------------------------
!       
      return
      end subroutine adv_param
