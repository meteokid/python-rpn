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

!**s/r hzd_in_rhs

      subroutine hzd_in_rhs ( F_du,F_dv, F_dw, F_dlnT, F_u, F_v , F_w, F_t, F_s, &
                              i0u,inu,j0u,jnu,i0v,inv,j0v,jnv, &
                              i0,in,j0,jn,Minx,Maxx,Miny,Maxy,Nk )
      use hzd_mod
      use grid_options
      use gem_options
      use tdpack
      use glb_ld
      use cstv
      use ver
      implicit none
#include <arch_specific.hf>

      real, dimension(Minx:Maxx,Miny:Maxy,NK),   intent (INOUT) :: F_du, F_dv, F_dw, F_dlnT
      real, dimension(Minx:Maxx,Miny:Maxy,NK),   intent (IN)    :: F_u, F_v, F_w, F_t
      real, dimension(Minx:Maxx,Miny:Maxy),      intent (IN)    :: F_s
      integer, intent(IN) :: i0u,inu,j0u,jnu,i0v,inv,j0v,jnv
      integer, intent(IN) :: i0,in,j0,jn,Minx,Maxx,Miny,Maxy,Nk
!author
!   Claude Girard
!

      integer mm,dpwr,niter,i,j,k
      real u0(Minx:Maxx,Miny:Maxy,Nk),v0(Minx:Maxx,Miny:Maxy,Nk)
      real u1(Minx:Maxx,Miny:Maxy,Nk),v1(Minx:Maxx,Miny:Maxy,Nk)
      real ppinv(Minx:Maxx,Miny:Maxy,Nk)
      real*8 coef_8(Nk)
!
!     ---------------------------------------------------------------
!
!     DIFFUSION des VENTS
      dpwr = Hzd_pwr/2
      niter=Hzd_niter

      if(niter > 0) then
         coef_8(1:NK) = Hzd_coef_8(1:Nk)

         u0=F_u
         v0=F_v
         do mm=1,dpwr
            call hzd_exp5p ( u0, u1, l_minx,l_maxx,l_miny,l_maxy,&
                             Nk, coef_8, 'U' , mm,dpwr )
            call hzd_exp5p ( v0, v1, l_minx,l_maxx,l_miny,l_maxy,&
                             Nk, coef_8, 'V' , mm,dpwr )
         enddo

         do k =1, Nk
            do j=j0u,jnu
            do i=i0u,inu
               F_du(i,j,k)=F_du(i,j,k)+Cstv_invT_m_8*(u0(i,j,k)-F_u(i,j,k))
            enddo
            enddo
            do j=j0v,jnv
            do i=i0v,inv
               F_dv(i,j,k)=F_dv(i,j,k)+Cstv_invT_m_8*(v0(i,j,k)-F_v(i,j,k))
            enddo
            enddo
         enddo

         if(.not.Schm_hydro_L) then

            u0=F_w
            do mm=1,dpwr
               call hzd_exp5p ( u0, u1, l_minx,l_maxx,l_miny,l_maxy,&
                                Nk, coef_8, 'M' , mm,dpwr )
            enddo

            do k =Vspng_nk+1, Nk
               do j=j0,jn
               do i=i0,in
                  F_dw(i,j,k)=F_dw(i,j,k)+Cstv_invT_nh_8*(u0(i,j,k)-F_w(i,j,k))
               enddo
               enddo
           enddo

         endif

      endif
!
!     DIFFUSION de la TEMPERATURE POTENTIELLE
      dpwr = Hzd_pwr_theta/2
      niter= Hzd_niter_theta

      if(niter > 0) then
         coef_8(1:NK) = Hzd_coef_8_theta(1:Nk)

         !theta=t/pi; pi=(p/p0)**cappa; p=exp(a+b*s); p0=1.
         do k=1,Nk
            ppinv(:,:,k)=exp(-cappa_8*(Ver_a_8%t(k)+Ver_b_8%t(k)*F_s(:,:)))
         enddo
         u0=F_t*ppinv

         do mm=1,dpwr
            call hzd_exp5p ( u0, u1, l_minx,l_maxx,l_miny,l_maxy,&
                             Nk, coef_8, 'M' , mm,dpwr )
         enddo

         u1=F_t*ppinv
         do k =1, Nk
            do j=j0,jn
            do i=i0,in
               F_dlnT(i,j,k)=F_dlnT(i,j,k)+Cstv_invT_8*(u0(i,j,k)-u1(i,j,k))/u1(i,j,k)
            enddo
            enddo
         enddo

      endif

!     ---------------------------------------------------------------
!
      return
      end subroutine hzd_in_rhs
