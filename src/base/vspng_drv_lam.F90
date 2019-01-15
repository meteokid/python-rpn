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
      use hzd_exp
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
#include "vspng.cdk"
!
!     ---------------------------------------------------------------
!
!     Momentum
!     ~~~~~~~~
      call hzd_exp_deln ( F_u,  'U', l_minx,l_maxx,l_miny,l_maxy,&
                          Vspng_nk, F_VV=F_v, F_type_S='VSPNG' )

!     Vertical motion
!     ~~~~~~~~~~~~~~~
      call hzd_exp_deln ( F_zd, 'M', l_minx,l_maxx,l_miny,l_maxy,&
                          Vspng_nk, F_type_S='VSPNG' )
         
!     Vertical wind
!     ~~~~~~~~~~~~~
      call hzd_exp_deln ( F_w,  'M', l_minx,l_maxx,l_miny,l_maxy,&
                          Vspng_nk, F_type_S='VSPNG' )

!     Temperature
!     ~~~~~~~~~~~~~
      call hzd_exp_deln ( F_t,  'M', l_minx,l_maxx,l_miny,l_maxy,&
                          Vspng_nk, F_type_S='VSPNG' )
!
!     ---------------------------------------------------------------
!
      return
      end
