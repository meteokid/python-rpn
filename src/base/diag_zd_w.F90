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

!** s/r diag_zd_w - Computes model vertical velocities zd and w diagnostically.
!

!
      subroutine diag_zd_w2( F_zd, F_w, F_xd, F_u, F_v, F_t, F_s, &
                             Minx,Maxx,Miny,Maxy, Nk, F_zd_L, F_w_L )
      implicit none
#include <arch_specific.hf>
!
      integer Minx,Maxx,Miny,Maxy, Nk
      real    F_zd(Minx:Maxx,Miny:Maxy,Nk), F_w(Minx:Maxx,Miny:Maxy,Nk), &
              F_u (Minx:Maxx,Miny:Maxy,Nk), F_v(Minx:Maxx,Miny:Maxy,Nk), &
              F_t (Minx:Maxx,Miny:Maxy,Nk), F_s(Minx:Maxx,Miny:Maxy)   , &
              F_xd(Minx:Maxx,Miny:Maxy,Nk)
      logical ::  F_zd_L,F_w_L
!
!authors
!      C.Girard & A.Plante, August 2011, based on routine uv2psd
!
!revision
!
! v4.7  - Gaudreault S.     - Reformulation in terms of real winds (removing wind images)
!
!arguments
!______________________________________________________________________
!        |                                             |           |   |
! NAME   |             DESCRIPTION                     | DIMENSION |I/O|
!--------|---------------------------------------------|-----------|---|
! F_zd   | coordinat vertical motion ( 1/s )           | 3D (Nk)   | o |
! F_w    | true vertical motion      ( m/s )           | 3D (Nk)   | o |
!--------|---------------------------------------------|-----------|---|
! F_u    | x component of velocity                     | 3D (Nk)   | i |
! F_v    | y component of velocity                     | 3D (Nk)   | i |
! F_t    | temperature                                 | 3D (Nk)   | i |
! F_s    | s log of surface pressure over constant     | 2D (1)    | i |
! F_zd_L | true to compute zdot                        | scal      | i |
! F_w_L  | true to compute w                           | scal      | i |
!________|_____________________________________________|___________|___|
!
#include "glb_ld.cdk"
#include "geomg.cdk"
#include "ver.cdk"
#include "dcst.cdk"
#include "zdot.cdk"
#include "cstv.cdk"
#include "intuv.cdk"
#include "lun.cdk"

      integer i, j, k, kp, i0, in, j0, jn, j00, jnn
      real*8 c1,c2
      real roverg
      real div (Minx:Maxx,Miny:Maxy,Nk),adv,div_i(Minx:Maxx,Miny:Maxy,0:Nk)
      real pi_t(Minx:Maxx,Miny:Maxy,Nk),lnpi_t(Minx:Maxx,Miny:Maxy,Nk),pidot
      real lnpi,pbX,pbY
      real sbX (Minx:Maxx,Miny:Maxy),sbY(Minx:Maxx,Miny:Maxy),pi_s(Minx:Maxx,Miny:Maxy)
      real UdpX(Minx:Maxx,Miny:Maxy),    VdpY(Minx:Maxx,Miny:Maxy)
      real lapse(Minx:Maxx,Miny:Maxy,Nk)
      real*8, parameter :: half=0.5d0
!     ________________________________________________________________
!
      if(.not.(F_zd_L.or.F_w_L)) return

      if (Lun_debug_L.and.F_zd_L) write (Lun_out,1000)
      if (Lun_debug_L.and.F_w_L ) write (Lun_out,1001)

      ! enlever calcul si F_zd_L=.false.
!
! Halo exchange needed because scope below goes one point in halo
!
      call rpn_comm_xch_halo( F_s,  l_minx,l_maxx,l_miny,l_maxy, l_ni, l_nj , 1,   &
                  G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
      call rpn_comm_xch_halo (F_u,  l_minx,l_maxx,l_miny,l_maxy, l_niu,l_nj, Nk,   &
                  G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
      call rpn_comm_xch_halo (F_v,  l_minx,l_maxx,l_miny,l_maxy, l_ni,l_njv, Nk,   &
                  G_halox,G_haloy,G_periodx,G_periody,l_ni,0)

!     Initializations

!     local grid setup for final results
      i0 = 1
      in = l_ni
      j0 = 1
      jn = l_nj
      if (G_lam) then ! peripheral points will be zeroed
         if(l_west)  i0 = 2
         if(l_east)  in = l_niu
         if(l_south) j0 = 2
         if(l_north) jn = l_njv
      endif

      j00=j0-1
      jnn=jn
      if (.not.G_lam) then
         if(l_south) then
             sbY(i0:in,j0-1)=0.
             F_v(i0:in,j0-1,1:Nk)=0.
            j00=j0
         endif
         if(l_north) then
             sbY(i0:in,jn)=0.
             F_v(i0:in,jn,1:Nk)=0.
            jnn=jn-1
         endif
      endif

!     CALCULATION of sbX, sbY, i.e. F_s on U and V points

      do j=j0,jn
      do i=i0-1,in
         sbX(i,j)=F_s(i,j)+(F_s(i+1,j)-F_s(i,j))*half
      end do
      end do

      do j=j00,jnn
      do i=i0,in
         sbY(i,j)=F_s(i,j)+(F_s(i,j+1)-F_s(i,j))*half
      end do
      end do

!     CALCULATION of pi_s

      do j=j0,jn
      do i=i0,in
         pi_s(i,j) = exp(Ver_z_8%m(Nk+1)+F_s(i,j))
      end do
      end do

      do k = 1, Nk

!        CALCULATION of U*dpi/dz and V*dpi/dz

         do j=j0,jn
         do i=i0-1,in
            lnpi=Ver_z_8%m(k)+Ver_b_8%m(k)*sbX(i,j)
             pbX=exp(lnpi)
            UdpX(i,j)=F_u(i,j,k)*pbX*(1.d0+Ver_dbdz_8%m(k)*sbX(i,j))
         end do
         end do
         do j=j0-1,jn
         do i=i0,in
            lnpi=Ver_z_8%m(k)+Ver_b_8%m(k)*sbY(i,j)
             pbY=exp(lnpi)
            VdpY(i,j)=F_v(i,j,k)*geomg_cyv_8(j)*pbY*(1.d0+Ver_dbdz_8%m(k)*sbY(i,j))
         end do
         end do

!        CALCULATION of DIV(V*dpi/dz)

         do j=j0,jn
         do i=i0,in
            div(i,j,k) = (UdpX(i,j)-UdpX(i-1,j))*geomg_invDX_8(j) &
                       + (VdpY(i,j)-VdpY(i,j-1))*geomg_invcy_8(j)*geomg_invDY_8
         enddo
         enddo

!        CALCULATION of lnpi_t and pi_t

         do j=j0,jn
         do i=i0,in
             lnpi_t(i,j,k) = Ver_a_8%t(k) + Ver_b_8%t(k) * F_s(i,j)
               pi_t(i,j,k) = exp(lnpi_t(i,j,k))
         end do
         end do

      enddo

      if (Zdot_divHLM_L) then
!     Compute lapse rate
         call verder (lapse,F_t,lnpi_t,2.0,2.0,l_minx,l_maxx,l_miny,l_maxy, &
                                          Nk,i0,in,j0,jn)

!     Scale divergence by the lapse rate
         do k=1,Nk
            do j=j0,jn
            do i=i0,in
               div(i,j,k) = div(i,j,k) * (tanh(2.*lapse(i,j,k)/Dcst_pi_8)+1.)/2.
            enddo
            enddo
         enddo
      endif

!     INTEGRATION OF THE DIVERGENCE IN THE VERTICAL

      div_i(i0:in,j0:jn,0)=0.
      do k=1,Nk
         do j=j0,jn
         do i=i0,in
            div_i(i,j,k)=div_i(i,j,k-1)+div(i,j,k)*Ver_dz_8%m(k)
         enddo
         enddo
      enddo

!     CALCULATION of ZDOT

      if (F_zd_L) then

         do k=1,Nk-1
            do j=j0,jn
            do i=i0,in
               F_zd(i,j,k) = ( Ver_b_8%t(k)*div_i(i,j,Nk)/pi_s(i,j) &
                             - div_i(i,j,k)/pi_t(i,j,k) ) /           &
                              (1.+Ver_dbdz_8%t(k)*F_s(i,j))
            end do
            end do
         end do

      endif

!     CALCULATION of W

      if (F_w_L) then

!        Compute W=-omega/(g*ro)      ro=p/RT

         RoverG=Dcst_rgasd_8/Dcst_grav_8
         do k=1,Nk
            kp=min(k+1,Nk)
            do j=j0,jn
            c1=geomg_cyv_8(j  ) ! pgi optimizer is bugged without those 2 lines
            c2=geomg_cyv_8(j-1)
            do i=i0,in
              !ADV = V*grad(s) = DIV(s*Vbarz)-s*DIV(Vbarz)
                adv= 0.5 * ( geomg_invDX_8(j) *  &
                    ( (F_u(i  ,j,kp)+F_u(i  ,j,k))*(sbX(i  ,j)-F_s(i,j))   &
                     -(F_u(i-1,j,kp)+F_u(i-1,j,k))*(sbX(i-1,j)-F_s(i,j)) ) &
                  + geomg_invcy_8(j) * geomg_invDY_8 *  &
                    ( (F_v(i,j  ,kp)+F_v(i,j  ,k))*c1*(sbY(i,j  )-F_s(i,j))   &
                     -(F_v(i,j-1,kp)+F_v(i,j-1,k))*c2*(sbY(i,j-1)-F_s(i,j)) ) )
              !pidot=omega
               pidot=pi_t(i,j,k)*Ver_b_8%t(k)*adv - div_i(i,j,k)
               F_w(i,j,k)= -RoverG*F_t(i,j,k)/pi_t(i,j,k)*pidot
              !ksidot=omega/pi
               F_xd(i,j,k)=pidot/pi_t(i,j,k)
            end do
            end do
         end do

      endif

1000  format(3X,'COMPUTE DIAGNOSTIC ZDT1: (S/R DIAG_ZD_W)')
1001  format(3X,'COMPUTE DIAGNOSTIC WT1:  (S/R DIAG_ZD_W)')
!
!     ________________________________________________________________
!
      return
      end
