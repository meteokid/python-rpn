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
!**s/r yyg_blenuv -for interpolating and nesting Yin-Yang UV boundary conditions
!

!
      subroutine yyg_blenuv (F_u, F_v, Minx,Maxx,Miny,Maxy, Nk )
      use gem_options
      use glb_ld
      use lun
      use gmm_itf_mod
      implicit none
#include <arch_specific.hf>
      integer, intent(in) :: Minx,Maxx,Miny,Maxy, Nk
!
      real, dimension(Minx:Maxx, Miny:Maxy, Nk) :: F_u, F_v
!
!author
!     Michel Desgagne   - Spring 2006
!
!revision
! v4_40 - Lee/Qaddouri  - adapt (nest_bcs) for Yin-yang (called every timestep)
! v4_60 - Lee V.        - routine to interpolate U,V, in pilot area
!
!object
!
!arguments
!       none
!

!
!     temp variables to manipulate winds for Yin-Yang
      real, dimension(Minx:Maxx, Miny:Maxy, NK) :: tempu, tempv
      integer :: i,j,k

!----------------------------------------------------------------------
!
      if (Lun_debug_L) write (Lun_out,1001)
!
      tempu = 0.
      tempv = 0.
!
!$omp parallel private(i,j,k) shared(tempu,tempv)
!$omp do
       do k= 1, NK
        do j= 1, l_nj
          do i= 1, l_niu
         tempu (i,j,k)=F_u (i,j,k)

        enddo
       enddo
        do j= 1, l_njv
          do i= 1, l_ni
          tempv(i,j,k)=F_v (i,j,k)
          enddo
        enddo
       enddo
!$omp enddo
!$omp end parallel

      call rpn_comm_xch_halo(tempu,l_minx,l_maxx,l_miny,l_maxy,l_niu,l_nj,Nk, &
                  G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      call rpn_comm_xch_halo(tempv,l_minx,l_maxx,l_miny,l_maxy,l_ni,l_njv,Nk, &
                  G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      call  yyg_blenu(F_u,tempu,tempv,l_minx,l_maxx,l_miny,l_maxy,Nk)
      call  yyg_blenv(F_v,tempv,tempu,l_minx,l_maxx,l_miny,l_maxy,Nk)
!
!
!----------------------------------------------------------------------
 1001 format(3X,'NEST YY Boundary ConditionS: (S/R yyg_blenuv)')
!
      return
      end
