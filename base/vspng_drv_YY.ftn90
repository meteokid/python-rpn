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

!**s/r vspng_drv_lam - Top sponge layer driver for Yin-Yang 

      subroutine vspng_drv_YY ( F_u, F_v, F_zd, F_w, F_t, Minx,Maxx,Miny,Maxy, Nk)
      implicit none
#include <arch_specific.hf>

      integer Minx,Maxx,Miny,Maxy, Nk
      real     F_u (Minx:Maxx,Miny:Maxy,Nk),F_v(Minx:Maxx,Miny:Maxy,Nk), &
               F_zd(Minx:Maxx,Miny:Maxy,Nk),F_w(Minx:Maxx,Miny:Maxy,Nk), &
               F_t (Minx:Maxx,Miny:Maxy,Nk)

!author
!     Abdessamad Qaddouri  -  2012
!
!revision
! v4_50   Qaddouri A.    - initial version for Yin-Yang from vspng_drv
!
!object
!     vertical sponge is applied:
!              on Vspng_nk   levels    

#include "glb_ld.cdk"
#include "vspng.cdk"

      integer iter
      real T_champ(Minx:Maxx,Miny:Maxy,Nk)
!
!     ---------------------------------------------------------------
!
      do iter = 1, Vspng_niter

!     NOTE : This code need to be optimized for exchanges. In its current
!            state it represent a 20% increse of a 15 km YY integration 
!            with physics!
!            1) Vertical motion, Vertical wind and Temperature could be
!               diffuse with the scalair code, this would reduce by 3/5*20%=12%
!               the increase in time.
!            2) Using the halows would cut the exchange by a factor equal to
!               the hallow size.

!     Momentum
!     ~~~~~~~~
         call yyg_rhsuv  (F_u,F_v, l_minx,l_maxx,l_miny,l_maxy, Nk )
         T_champ(:,:,:)= F_u(:,:,:)   
         call  vspngu_YY (F_u,F_v    , l_minx,l_maxx,l_miny,l_maxy,l_niu,l_nj )
         call  vspngv_YY (F_v,T_champ, l_minx,l_maxx,l_miny,l_maxy,l_ni ,l_njv)

!     Vertical motion
!     ~~~~~~~~~~~~~~~
!********not physical, done nevertheless*******

         call yyg_xchng (F_zd, l_minx,l_maxx,l_miny,l_maxy, Nk,&
                         .false., 'CUBIC')
         call vspng_YY  (F_zd, l_minx,l_maxx,l_miny,l_maxy, l_ni , l_nj)

!     Vertical wind
!     ~~~~~~~~~~~~~
         call yyg_xchng (F_w , l_minx,l_maxx,l_miny,l_maxy, Nk,&
                         .false., 'CUBIC')
         call vspng_YY  (F_w , l_minx,l_maxx,l_miny,l_maxy, l_ni , l_nj)
      
!     Temperature
         call yyg_xchng (F_t , l_minx,l_maxx,l_miny,l_maxy, Nk,&
                         .false., 'CUBIC')
         call vspng_YY  (F_t , l_minx,l_maxx,l_miny,l_maxy, l_ni , l_nj)

      enddo
!
!     ---------------------------------------------------------------
!
      return
      end
