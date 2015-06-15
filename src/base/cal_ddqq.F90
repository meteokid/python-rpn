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

!** s/r cal_ddqq - Computes horizontal divergence, vorticity and relative vorticity


      subroutine cal_ddqq ( F_DD,F_QR,F_QQ, F_uu,F_vv           , &
                            F_filtdd,F_coefdd, F_filtqq,F_coefqq, &
                            F_div_L, F_relvor_L, F_absvor_L     , &
                            Minx,Maxx,Miny,Maxy,Nk )
      implicit none
#include <arch_specific.hf>

      integer  F_filtdd,F_filtqq, Minx,Maxx,Miny,Maxy,Nk
      logical  F_div_L,F_relvor_L,F_absvor_L
      real     F_DD(Minx:Maxx,Miny:Maxy,Nk), &
               F_QR(Minx:Maxx,Miny:Maxy,Nk), &
               F_QQ (Minx:Maxx,Miny:Maxy,Nk), &
               F_uu (Minx:Maxx,Miny:Maxy,Nk), &
               F_vv (Minx:Maxx,Miny:Maxy,Nk), F_coefdd,F_coefqq
!author
!    Michel Desgagne   - spring 2014
!
!revision
! v4_70 - Desgagne M.       - initial version
!

#include "glb_ld.cdk"
#include "geomn.cdk"
#include "geomg.cdk"
#include "dcst.cdk"

      integer i, j, k, i0, in, j0, jn
      real deg2rad
!     __________________________________________________________________
!
      i0 = 1
      in = l_niu
      j0 = 1
      jn = l_njv
      if ((G_lam).and.(l_west)) i0 = 2
      if (l_south)              j0 = 2

      if(F_div_L)then
         do k = 1 , Nk
            do j = j0, jn
            do i = i0, in
               F_DD(i,j,k) = ((F_uu(i,j,k) - F_uu(i-1,j,k)) * geomg_invDX_8(j)) &
                           + ((F_vv(i,j,k)*geomg_cyv_8(j) - F_vv(i,j-1,k)*geomg_cyv_8(j-1)) &
                             * Geomg_invDY_8* geomg_invcy_8(j))
            end do
            end do
            F_DD(1:i0-1,:,k) = 0. ; F_DD(in+1:l_ni,:,k)= 0.
            F_DD(:,1:j0-1,k) = 0. ; F_DD(:,jn+1:l_nj,k)= 0.
         end do
      endif

      if (F_filtdd.gt.0) &
           call filter2( F_DD, F_filtdd,F_coefdd, &
                         l_minx,l_maxx,l_miny,l_maxy,Nk )

      if (F_relvor_L) then
         do k = 1 , Nk
            do j = j0, jn
            do i = i0, in
               F_QR(i,j,k) = ((F_vv(i+1,j,k) - F_vv(i,j,k)) * geomg_invDXz_8(j)) &
                           - ((F_uu(i,j+1,k)*geomg_cy_8(j+1) - F_uu(i,j,k)*geomg_cy_8(j)) * &
                              geomg_invDY_8 * geomg_invcyv_8(j))
            end do
            end do
            F_QR(1:i0-1,:,k) = 0. ; F_QR(in+1:l_ni,:,k)= 0.
            F_QR(:,1:j0-1,k) = 0. ; F_QR(:,jn+1:l_nj,k)= 0.
         end do
         if (F_filtqq.gt.0) &
           call filter2( F_QR, F_filtqq,F_coefqq, &
                         l_minx,l_maxx,l_miny,l_maxy,Nk )      
      endif

      if (F_absvor_L)then
         deg2rad= acos( -1.0)/180.
         if (F_relvor_L) then
            do k =  1, Nk
               do j = j0, jn
               do i = i0, in
                  F_QQ(i,j,k)= F_QR(i,j,k) + 2.0*Dcst_omega_8 &
                              * sin(Geomn_latrx(i,j)*deg2rad)
               end do
               end do
               F_QQ(1:i0-1,:,k) = 0. ; F_QQ(in+1:l_ni,:,k)= 0.
               F_QQ(:,1:j0-1,k) = 0. ; F_QQ(:,jn+1:l_nj,k)= 0.
            end do
         else           
            do k = 1 , Nk
               do j = j0, jn
               do i = i0, in
                  F_QQ(i,j,k) = ((F_vv(i+1,j,k) - F_vv(i,j,k)) * geomg_invDXz_8(j)) &
                              - ((F_uu(i,j+1,k)*geomg_cy_8(j+1) - F_uu(i,j,k)*geomg_cy_8(j)) * &
                                 geomg_invDY_8 * geomg_invcyv_8(j))
               end do
               end do
               F_QQ(1:i0-1,:,k) = 0. ; F_QQ(in+1:l_ni,:,k)= 0.
               F_QQ(:,1:j0-1,k) = 0. ; F_QQ(:,jn+1:l_nj,k)= 0.
            end do
            if (F_filtqq.gt.0) &
                 call filter2( F_QQ, F_filtqq,F_coefqq, &
                               l_minx,l_maxx,l_miny,l_maxy,Nk )      
            do k =  1, Nk
               do j = j0, jn
               do i = i0, in
                  F_QQ(i,j,k)= F_QQ(i,j,k) + 2.0*Dcst_omega_8 &
                              * sin(Geomn_latrx(i,j)*deg2rad)
               end do
               end do
               F_QQ(1:i0-1,:,k) = 0. ; F_QQ(in+1:l_ni,:,k)= 0.
               F_QQ(:,1:j0-1,k) = 0. ; F_QQ(:,jn+1:l_nj,k)= 0.
            end do
         endif
      endif
!     __________________________________________________________________
!
      return
      end
