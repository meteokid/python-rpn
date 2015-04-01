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

!**s/r vspng_drv_lam - Top sponge layer driver for LAM
!

!
      subroutine vspng_drv_lam ( F_u, F_v, F_zd, F_w, F_t, Minx,Maxx,Miny,Maxy, Nk)
!
      implicit none
#include <arch_specific.hf>
!
      integer Minx,Maxx,Miny,Maxy, Nk
      real    F_u (Minx:Maxx,Miny:Maxy,Nk), F_v(Minx:Maxx,Miny:Maxy,Nk), &
              F_zd(Minx:Maxx,Miny:Maxy,Nk), F_w(Minx:Maxx,Miny:Maxy,Nk), &
              F_t (Minx:Maxx,Miny:Maxy,Nk)
!
!author
!     Michel Desgagne  - October 2000
!
!revision
! v3_02   Lee V.    - initial version for LAM from vspng_drv
! v4    - Gravel-Girard-PLante - staggered version
! v4_04 - Girard-PLante     - Diffuse only winds, zdot and first Temp. level.
! v4_05 - Girard-PLante     - Diffuse w.
!
!object
!     vertical sponge is applied:
!              on Vspng_nk   levels      for momentum only
! 
#include "glb_ld.cdk"
#include "schm.cdk"
#include "vspng.cdk"
#include "lun.cdk"
!
      integer i, j, k, nkspng
!
!     ---------------------------------------------------------------
!
!     Momentum
!     ~~~~~~~~
      call vspng_lam (F_u, l_minx,l_maxx,l_miny,l_maxy, l_niu, l_nj )
      call vspng_lam (F_v, l_minx,l_maxx,l_miny,l_maxy, l_ni , l_njv)

!     Vertical motion
!     ~~~~~~~~~~~~~~~
!********not physical, done nevertheless*******
      call vspng_lam (F_zd,l_minx,l_maxx,l_miny,l_maxy, l_ni , l_nj )

!     Vertical wind
!     ~~~~~~~~~~~~~
      call vspng_lam (F_w, l_minx,l_maxx,l_miny,l_maxy, l_ni , l_nj )
      
!     Temperature
      call vspng_lam (F_t, l_minx,l_maxx,l_miny,l_maxy, l_ni , l_nj )
!
!     ---------------------------------------------------------------
!
      return
      end
