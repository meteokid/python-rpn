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

module hzd_exp
      use hzd_mod
      use gem_options
      use grid_options
  implicit none
#include <arch_specific.hf>
  private
  public :: hzd_exp_deln

contains

!**s/r hzd_exp_deln - 5 points explicit del 'n' horizontal diffusion 
!                     for LAM configuration

      subroutine hzd_exp_deln ( F_c1, F_hgrid_S, Minx,Maxx,Miny,Maxy,Nk,&
                                F_vv, F_type_S )
      implicit none
#include <arch_specific.hf>

      character*(*)          , intent(IN) :: F_hgrid_S
      character*(*), optional, intent(IN) :: F_type_S
      integer, intent(IN) :: Minx,Maxx,Miny,Maxy,Nk
      real, dimension(Minx:Maxx,Miny:Maxy,NK),           intent (INOUT) :: F_c1
      real, dimension(Minx:Maxx,Miny:Maxy,NK), optional, intent (INOUT) :: F_vv
!author    
!    Abdessamad Qaddouri - summer 2015
!
!revision
! v4_80 - Qaddouri A.      - initial version
! v4_80 - Lee   - optimization

#include "glb_ld.cdk"

      integer iter1,iter2,mm,dpwr,itercnt,Niter,Pwr
      real c1(Minx:Maxx,Miny:Maxy,Nk), c2(Minx:Maxx,Miny:Maxy,Nk)
      real*8 coef_8(Nk)
!
!     ---------------------------------------------------------------
!
      dpwr = Hzd_pwr
      niter= Hzd_niter
      if (niter.gt.0) coef_8(1:NK) = Hzd_coef_8(1:Nk)

      if (present(F_type_S)) then
         if (F_type_S.eq.'S_THETA') then
            dpwr = Hzd_pwr_theta
            niter= Hzd_niter_theta
            if (niter.gt.0) coef_8(1:NK) = Hzd_coef_8_theta(1:Nk)
         endif
         if (F_type_S.eq.'S_TR') then
            dpwr = Hzd_pwr_tr
            niter= Hzd_niter_tr
            if (niter.gt.0) coef_8(1:NK) = Hzd_coef_8_tr(1:Nk)
         endif
         if (F_type_S.eq.'VSPNG') then
            dpwr = 2
            niter= Vspng_niter
            if (niter.gt.0)coef_8(1:NK) = Vspng_coef_8(1:Nk)
         endif
      endif

      if (niter.le.0) return
      dpwr=dpwr/2

!     Fill all halo regions

      if (Grd_yinyang_L) then
         if (present(F_vv)) then
            call yyg_nestuv(F_c1, F_vv, l_minx,l_maxx,l_miny,l_maxy, Nk)
         else
            call yyg_xchng (F_c1, l_minx,l_maxx,l_miny,l_maxy, Nk,&
                                                 .false., 'CUBIC')
         endif
      endif
      call rpn_comm_xch_halo(F_c1,l_minx,l_maxx,l_miny,l_maxy,&
               l_ni,l_nj,Nk,G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      if (present(F_vv)) then
         call rpn_comm_xch_halo(F_vv,l_minx,l_maxx,l_miny,l_maxy,&
               l_ni,l_nj,Nk,G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      endif

      itercnt=0

      do iter1= 1, niter

         do mm=1,dpwr

            call hzd_exp5p ( F_c1, c1, l_minx,l_maxx,l_miny,l_maxy,&
                             Nk, coef_8, F_hgrid_S, mm,dpwr )
            if (present(F_vv)) &
            call hzd_exp5p ( F_vv, c2, l_minx,l_maxx,l_miny,l_maxy,&
                             Nk, coef_8, 'V'      , mm,dpwr )

            itercnt = itercnt + 1

            if (itercnt.eq.G_halox) then
               if (Grd_yinyang_L) then
                  if (present(F_vv)) then
                     call yyg_nestuv(F_c1, F_vv, l_minx,l_maxx,l_miny,l_maxy,Nk)
                  else
                     call yyg_xchng (F_c1,l_minx,l_maxx,l_miny,l_maxy, Nk,&
                                                     .false., 'CUBIC')
                  endif
               endif
               call rpn_comm_xch_halo(F_c1,l_minx,l_maxx,l_miny,l_maxy,&
                    l_ni,l_nj,Nk,G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
               if (present(F_vv)) &
               call rpn_comm_xch_halo(F_vv,l_minx,l_maxx,l_miny,l_maxy,&
                    l_ni,l_nj,Nk,G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
               itercnt=0
            endif

         enddo

      enddo
!
!     ---------------------------------------------------------------
!
      return
      end subroutine hzd_exp_deln

end module hzd_exp
