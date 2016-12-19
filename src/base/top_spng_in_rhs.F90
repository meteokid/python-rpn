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

!**s/r top_spng_in_rhs

      subroutine top_spng_in_rhs ( F_du,F_dv, F_dw, F_dlnT, F_u, F_v , F_w, F_t, F_s, &
                                   i0u,inu,j0u,jnu,i0v,inv,j0v,jnv, &
                                   i0,in,j0,jn,Minx,Maxx,Miny,Maxy,Nk )
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
#include "glb_ld.cdk"
#include "grd.cdk"
#include "hzd.cdk"
#include "vspng.cdk"
#include "dcst.cdk"
#include "ver.cdk"
#include "schm.cdk"
#include "cstv.cdk"

      integer iter,niter,i,j,k, itercnt
      real u0(Minx:Maxx,Miny:Maxy,Nk),v0(Minx:Maxx,Miny:Maxy,Nk)
      real wk(Minx:Maxx,Miny:Maxy,Nk)
      real*8 coef_8(Nk)
!
!     ---------------------------------------------------------------
!
!     EPONGE AU TOIT
      niter= Vspng_niter

      if (niter.gt.0) then
         coef_8(1:Vspng_Nk) = Vspng_coef_8(1:Vspng_Nk)

!        POUR LES VENTS
         itercnt=0
         u0=F_u
         v0=F_v
         do iter=1,niter
            call hzd_exp5p ( u0, wk, l_minx,l_maxx,l_miny,l_maxy,&
                             Vspng_Nk, coef_8, 'U' , 1,1 )
            call hzd_exp5p ( v0, wk, l_minx,l_maxx,l_miny,l_maxy,&
                             Vspng_Nk, coef_8, 'V' , 1,1 )
            itercnt=itercnt+1
            if(iter.lt.niter) then
               if (itercnt.eq.G_halox) then
                  if (Grd_yinyang_L) then
                     call yyg_nestuv(u0, v0, l_minx,l_maxx,l_miny,l_maxy,Vspng_Nk)
                  endif
                  call rpn_comm_xch_halo( u0 , l_minx,l_maxx,l_miny,l_maxy,l_niu,l_nj ,Vspng_Nk, &
                                 G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
                  call rpn_comm_xch_halo( v0 , l_minx,l_maxx,l_miny,l_maxy,l_ni ,l_njv,Vspng_Nk, &
                                 G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
                  itercnt=0
               endif
            endif
         enddo
         do k =1, Vspng_Nk
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
            itercnt=0
            u0=F_w
            do iter=1,niter
               call hzd_exp5p ( u0, wk, l_minx,l_maxx,l_miny,l_maxy,&
                                Vspng_Nk, coef_8, 'M' , 1,1 )
               itercnt=itercnt+1
               if(iter.lt.niter) then
                  if (itercnt.eq.G_halox) then
                     if (Grd_yinyang_L) then
                        call yyg_xchng (u0, l_minx,l_maxx,l_miny,l_maxy, Vspng_Nk,&
                                                 .false., 'CUBIC')
                     endif
                     call rpn_comm_xch_halo( u0 , l_minx,l_maxx,l_miny,l_maxy,l_niu,l_nj ,Vspng_Nk, &
                                    G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
                     itercnt=0
                  endif
               endif
            enddo
            do k =1, Vspng_Nk
               do j=j0,jn
               do i=i0,in
                  F_dw(i,j,k)=F_dw(i,j,k)+Cstv_invT_nh_8*(u0(i,j,k)-F_w(i,j,k))
               enddo
               enddo
            enddo
         endif

!        POUR LA TEMPERATURE
         itercnt=0
         u0=F_T
         do iter=1,niter
            call hzd_exp5p ( u0, wk, l_minx,l_maxx,l_miny,l_maxy,&
                             Vspng_Nk, coef_8, 'M' , 1,1 )
            itercnt=itercnt+1
            if(iter.lt.niter) then
               if (itercnt.eq.G_halox) then
                  if (Grd_yinyang_L) then
                     call yyg_xchng (u0, l_minx,l_maxx,l_miny,l_maxy, Vspng_Nk,&
                                                 .false., 'CUBIC')
                  endif
                  call rpn_comm_xch_halo( u0 , l_minx,l_maxx,l_miny,l_maxy,l_niu,l_nj ,Vspng_Nk, &
                                 G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
                  itercnt=0
               endif
            endif
         enddo
         do k =1, Vspng_Nk
            do j=j0,jn
            do i=i0,in
               F_dlnT(i,j,k)=F_dlnT(i,j,k)+Cstv_invT_8*(u0(i,j,k)-F_T(i,j,k))/u0(i,j,k)
            enddo
            enddo
         enddo

      endif
!     ---------------------------------------------------------------

!
      return
      end subroutine top_spng_in_rhs
