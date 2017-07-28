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
      use glb_ld
      use ver
      use adv_grid
      use adv_interp
      use outgrid
      implicit none
#include <arch_specific.hf>

   !@objective set 1-D interpolation of grid reflexion across the pole
   !@author  alain patoine
   !@revisions
   !*@/


      real*8, parameter :: LARGE_8 = 1.D20 , TWO_8 = 2.D0  , SIX_8 = 6.D0
      integer :: i, j, k, i0, j0, k0, pnx, pny
      real*8 :: ra,rb,rc,rd
      real*8 :: prhxmn, prhymn, prhzmn, pdfi
      real*8 :: whx(G_ni+2*adv_halox)
      real*8 :: why(G_nj+2*adv_haloy)
      real*8, dimension(0:l_nk+1) :: whzt,whzm,whzx

      real*8 :: TRIPROD,za,zb,zc,zd
      TRIPROD(za,zb,zc,zd) = ((za-zb)*(za-zc)*(za-zd))
!
!     ---------------------------------------------------------------
!

      adv_xbc_8 = 1.D0/(adv_xg_8(adv_gminx+1)-adv_xg_8(adv_gminx))
      adv_xabcd_8=-adv_xbc_8**3/SIX_8
      adv_xdabc_8=-adv_xabcd_8
      adv_xbacd_8= adv_xbc_8**3/TWO_8
      adv_xcabd_8=-adv_xbacd_8


      adv_ybc_8 = 1.D0/(adv_yg_8(adv_gminx+1)-adv_yg_8(adv_gminx))
      adv_yabcd_8=-adv_ybc_8**3/SIX_8
      adv_ydabc_8=-adv_yabcd_8
      adv_ybacd_8=adv_ybc_8**3/TWO_8
      adv_ycabd_8=-adv_ybacd_8

      adv_x00_8 = adv_xg_8(adv_gminx)
      adv_y00_8 = adv_yg_8(adv_gminy)

      prhxmn = LARGE_8 ; prhymn = LARGE_8 ; prhzmn = LARGE_8

      do i = adv_gminx,adv_gmaxx-1
         whx(adv_halox+i) = adv_xg_8(i+1) - adv_xg_8(i)
         prhxmn = min(whx(adv_halox+i), prhxmn)
      enddo

      do j = adv_gminy,adv_gmaxy-1
         why(adv_haloy+j) = adv_yg_8(j+1) - adv_yg_8(j)
         prhymn = min(why(adv_haloy+j), prhymn)
      enddo

! Prepare zeta on super vertical grid

      whzt(0    ) = 1.0 ; whzt(l_nk  ) = 1.0 ; whzt(l_nk+1) = 1.0
      do k = 1,l_nk-1
         whzt(k) = Ver_z_8%t(k+1) - Ver_z_8%t(k)
         prhzmn = min(whzt(k), prhzmn)
      enddo

      whzm(0    ) = 1.0  ; whzm(l_nk  ) = 1.0 ; whzm(l_nk+1) = 1.0
      do k = 1,l_nk-1
         whzm(k) = Ver_z_8%m(k+1) - Ver_z_8%m(k)
         prhzmn = min(whzm(k), prhzmn)
      enddo

      whzx(0    ) = 1.0 ; whzx(l_nk  ) = 1.0 ; whzx(l_nk+1) = 1.0
      do k = 1,l_nk-1
         whzx(k) = Ver_z_8%x(k+1) - Ver_z_8%x(k)
         prhzmn = min(whzx(k), prhzmn)
      enddo

      adv_ovdx_8 = 1.0d0/prhxmn
      adv_ovdy_8 = 1.0d0/prhymn
      adv_ovdz_8 = 1.0d0/prhzmn

      pnx = int(1.0+(adv_xg_8(adv_gmaxx)-adv_x00_8)   *adv_ovdx_8)
      pny = int(1.0+(adv_yg_8(adv_gmaxy)-adv_y00_8)   *adv_ovdy_8)
      pnz = nint(1.0+(Ver_z_8%m(l_nk+1)-Ver_z_8%m(0))*adv_ovdz_8)

      allocate( adv_lcx(pnx), adv_lcy(pny),  adv_bsx_8(G_ni+2*adv_halox), &
              adv_dlx_8(G_ni+2*adv_halox), adv_bsy_8(G_nj+2*adv_haloy), &
              adv_dly_8(G_nj+2*adv_haloy), adv_lcz%t(pnz), adv_lcz%m(pnz), adv_lcz%x(pnz), &
              adv_bsz_8%t(0:l_nk-1), adv_bsz_8%m(0:l_nk-1), adv_bsz_8%x(0:l_nk-1), &
              adv_dlz_8%t(-1:l_nk) , adv_dlz_8%m(-1:l_nk) , adv_dlz_8%x(-1:l_nk),adv_diz_8(-1:l_nk))

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
         pdfi = Ver_z_8%m(0) + (k-1) * prhzmn
         if (pdfi > Ver_z_8%t(k0+1)) k0 = min(l_nk-2, k0+1)
         adv_lcz%t(k) = k0
      enddo

      do k = 0,l_nk+1            !! warning note the shift in k !!
         adv_dlz_8%t(k-1) =       whzt(k)
      enddo

      do k = 1,l_nk
         adv_bsz_8%t(k-1) = Ver_z_8%t(k)
      enddo

      k0 = 1
      do k = 1,pnz
         pdfi = Ver_z_8%m(0) + (k-1) * prhzmn
         if (pdfi > Ver_z_8%m(k0+1)) k0 = min(l_nk-2, k0+1)
         adv_lcz%m(k) = k0
      enddo

      do k = 0,l_nk+1            !! warning note the shift in k !!
         adv_dlz_8%m(k-1) =       whzm(k)
      enddo

      do k = 1,l_nk
         adv_bsz_8%m(k-1) = Ver_z_8%m(k)
      enddo

      k0 = 1
      do k = 1,pnz
         pdfi = Ver_z_8%m(0) + (k-1) * prhzmn
         if (pdfi > Ver_z_8%x(k0+1)) k0 = min(l_nk-2, k0+1)
         adv_lcz%x(k) = k0
      enddo

      do k = 0,l_nk+1            !! warning note the shift in k !!
         adv_dlz_8%x(k-1) =       whzx(k)
      enddo

      do k = 1,l_nk
         adv_bsz_8%x(k-1) = Ver_z_8%x(k)
      enddo

      do k = 0,l_nk+1                  !! warning note the shift in k !
         adv_diz_8(k-1) = 1.0d0/whzm(k)
      enddo

      allocate( &
      adv_zbc_8%t(l_nk),   adv_zbc_8%m(l_nk),    adv_zbc_8%x(l_nk), &
      adv_zabcd_8%t(l_nk), adv_zabcd_8%m(l_nk),  adv_zabcd_8%x(l_nk), &
      adv_zbacd_8%t(l_nk), adv_zbacd_8%m(l_nk),  adv_zbacd_8%x(l_nk), &
      adv_zcabd_8%t(l_nk), adv_zcabd_8%m(l_nk),  adv_zcabd_8%x(l_nk), &
      adv_zdabc_8%t(l_nk), adv_zdabc_8%m(l_nk),  adv_zdabc_8%x(l_nk) )

      do k = 2,l_nk-2
         ra = Ver_z_8%m(k-1)
         rb = Ver_z_8%m(k)
         rc = Ver_z_8%m(k+1)
         rd = Ver_z_8%m(k+2)
         adv_zabcd_8%m(k) = 1.0/TRIPROD(ra,rb,rc,rd)
         adv_zbacd_8%m(k) = 1.0/TRIPROD(rb,ra,rc,rd)
         adv_zcabd_8%m(k) = 1.0/TRIPROD(rc,ra,rb,rd)
         adv_zdabc_8%m(k) = 1.0/TRIPROD(rd,ra,rb,rc)

         ra = Ver_z_8%t(k-1)
         rb = Ver_z_8%t(k)
         rc = Ver_z_8%t(k+1)
         rd = Ver_z_8%t(k+2)
         adv_zabcd_8%t(k) = 1.0/TRIPROD(ra,rb,rc,rd)
         adv_zbacd_8%t(k) = 1.0/TRIPROD(rb,ra,rc,rd)
         adv_zcabd_8%t(k) = 1.0/TRIPROD(rc,ra,rb,rd)
         adv_zdabc_8%t(k) = 1.0/TRIPROD(rd,ra,rb,rc)

         ra = Ver_z_8%x(k-1)
         rb = Ver_z_8%x(k)
         rc = Ver_z_8%x(k+1)
         rd = Ver_z_8%x(k+2)
         adv_zabcd_8%x(k) = 1.0/TRIPROD(ra,rb,rc,rd)
         adv_zbacd_8%x(k) = 1.0/TRIPROD(rb,ra,rc,rd)
         adv_zcabd_8%x(k) = 1.0/TRIPROD(rc,ra,rb,rd)
         adv_zdabc_8%x(k) = 1.0/TRIPROD(rd,ra,rb,rc)
      enddo

      do k = 1,l_nk-1
         rb = Ver_z_8%m(k)
         rc = Ver_z_8%m(k+1)
         adv_zbc_8%m(k) = 1.0/(rc-rb)

         rb = Ver_z_8%t(k)
         rc = Ver_z_8%t(k+1)
         adv_zbc_8%t(k) = 1.0/(rc-rb)

         rb = Ver_z_8%x(k)
         rc = Ver_z_8%x(k+1)
         adv_zbc_8%x(k) = 1.0/(rc-rb)
      enddo

!
!     ---------------------------------------------------------------
!
      return
      end subroutine adv_param
