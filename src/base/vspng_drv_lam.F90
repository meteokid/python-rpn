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

!**s/r vspng_drv_lam - Top sponge layer driver for LAMs

      subroutine vspng_drv_lam ( F_u, F_v, F_zd, F_w, F_t, &
                                   Minx,Maxx,Miny,Maxy, Nk )
      implicit none
#include <arch_specific.hf>

      integer Minx,Maxx,Miny,Maxy, Nk
      real   F_u (Minx:Maxx,Miny:Maxy,Nk),F_v(Minx:Maxx,Miny:Maxy,Nk),&
             F_zd(Minx:Maxx,Miny:Maxy,Nk),F_w(Minx:Maxx,Miny:Maxy,Nk),&
             F_t (Minx:Maxx,Miny:Maxy,Nk)
!author    
!    Abdessamad Qaddouri - summer 2015
!
!revision
! v4_80 - Qaddouri A.      - initial version
! v4_80 - Desgagne & Lee   - optimization
!

#include "glb_ld.cdk"
#include "grd.cdk"
#include "vspng.cdk"

      integer iter1,iter2,BL_iter,SL_iter,m_eps 
!
!     ---------------------------------------------------------------
!
      BL_iter= int(Vspng_niter/G_halox)
       m_eps = mod(Vspng_niter,G_halox)
      if (m_eps.gt.0) BL_iter= BL_iter+1
      SL_iter= G_halox

      do iter1= 1, BL_iter

         if (iter1 .eq. BL_iter) then
            if (m_eps.gt.0) SL_iter= m_eps
         endif

!     Momentum
!     ~~~~~~~~
         if (Grd_yinyang_L) &
         call  yyg_nestuv (F_u, F_v, l_minx,l_maxx,l_miny,l_maxy, Vspng_nk )
         call rpn_comm_xch_halo (F_u,l_minx,l_maxx,l_miny,l_maxy,l_niu,l_nj, &
                          Vspng_nk,G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
         call rpn_comm_xch_halo (F_v,l_minx,l_maxx,l_miny,l_maxy,l_ni ,l_njv, &
                          Vspng_nk,G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
!     Vertical motion
!     ~~~~~~~~~~~~~~~
         if (Grd_yinyang_L) &
         call yyg_xchng (F_zd , l_minx,l_maxx,l_miny,l_maxy, Vspng_nk,&
                                                      .false., 'CUBIC')
         call rpn_comm_xch_halo (F_zd,l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj, &
                          Vspng_nk,G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
         
!     Vertical wind
!     ~~~~~~~~~~~~~
         if (Grd_yinyang_L) &
         call yyg_xchng (F_w , l_minx,l_maxx,l_miny,l_maxy, Vspng_nk,&
                                                     .false., 'CUBIC')
         call rpn_comm_xch_halo (F_w,l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj, &
                       Vspng_nk,G_halox,G_haloy,G_periodx,G_periody,l_ni,0)

!     Temperature
!     ~~~~~~~~~~~~~
         if (Grd_yinyang_L) &
         call yyg_xchng (F_t , l_minx,l_maxx,l_miny,l_maxy, Vspng_nk,&
                                                     .false., 'CUBIC')
         call rpn_comm_xch_halo (F_t,l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj, &
                       Vspng_nk,G_halox,G_haloy,G_periodx,G_periody,l_ni,0)

         do iter2= 1, SL_iter
            call hzd_exp5p ( F_u , l_minx,l_maxx,l_miny,l_maxy,Vspng_nk, Vspng_coef_8, 'U')
            call hzd_exp5p ( F_v , l_minx,l_maxx,l_miny,l_maxy,Vspng_nk, Vspng_coef_8, 'V')
            call hzd_exp5p ( F_t , l_minx,l_maxx,l_miny,l_maxy,Vspng_nk, Vspng_coef_8, 'M')
            call hzd_exp5p ( F_zd, l_minx,l_maxx,l_miny,l_maxy,Vspng_nk, Vspng_coef_8, 'M')
            call hzd_exp5p ( F_w , l_minx,l_maxx,l_miny,l_maxy,Vspng_nk, Vspng_coef_8, 'M')
         enddo
     
      enddo
!
!     ---------------------------------------------------------------
!
      return
      end
