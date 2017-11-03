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
!
      subroutine adv_prepareWinds ( F_ud, F_vd, F_wd, F_ua, F_va, F_wa, F_wat , &
                                     ut0, vt0, zdt0, ut1, vt1 , zdt1          , &
                                     F_minx, F_maxx, F_miny, F_maxy , F_ni ,F_nj, F_nk )
      use dcst
      use gem_options
      use glb_ld
      use gmm_itf_mod
      use gmm_pw
      use gmm_vt2
      use tdpack
      implicit none
#include <arch_specific.hf>

      integer, intent(in) :: F_minx,F_maxx,F_miny,F_maxy    ! min, max values for indices
      integer, intent(in) :: F_ni,F_nj                      ! horizontal dims of position fields
      integer, intent(in) :: F_nk                           ! nb of winds vertical levels
      real, dimension(F_minx:F_maxx,F_miny:F_maxy,F_nk),intent(in) :: ut0, vt0 , zdt0
      real, dimension(F_minx:F_maxx,F_miny:F_maxy,F_nk),intent(in) :: ut1, vt1 , zdt1
      real, dimension(F_minx:F_maxx,F_miny:F_maxy,F_nk),intent(out) :: F_ud, F_vd, F_wd    ! model un-staggered departure winds
      real, dimension(F_ni,F_nj,F_nk), intent(out):: F_ua,F_va,F_wa,F_wat                  ! arrival unstaggered  winds

 !@Objectives: Process winds in preparation for advection: de-stagger and interpolate from thermo to momentum levels


      real, dimension(F_minx:F_maxx,F_miny:F_maxy,F_nk) :: uh,vh,wm,wh
      real :: err
      integer :: i,j,k
!
!     ---------------------------------------------------------------
!

      if (Schm_trapeze_L) then

!$omp parallel private(i,j,k) shared(uh,vh,wm)
!$omp do
         do k = 1,F_nk
            do j = F_miny,F_maxy
               do i = F_minx,F_maxx
                  uh(i,j,k) = ut0(i,j,k)
                  vh(i,j,k) = vt0(i,j,k)
                  wm(i,j,k) = 0.0
               end do
            end do
         end do

!$omp enddo
!$omp end parallel

         call adv_destagWinds (uh,vh,F_minx,F_maxx,F_miny,F_maxy,F_nk)

         call adv_thermo2mom (wm,zdt0,F_ni,F_nj,F_nk,F_minx,F_maxx,F_miny,F_maxy)

!     Unstaggered arrival winds

!$omp parallel private(i,j,k)  shared(uh,vh,wm)
!$omp do
         do k = 1,F_nk
            do j = 1,F_nj
               do i = 1,F_ni
                  F_ua(i,j,k)  = uh(i,j,k)
                  F_va(i,j,k)  = vh(i,j,k)
                  F_wa(i,j,k)  = wm(i,j,k)
                  F_wat(i,j,k) = zdt0(i,j,k)
               enddo
            enddo
         enddo
!$omp enddo
!$omp end parallel

! DEPARTURE WINDS: NO DESTAGRING
         err = gmm_get (gmmk_pw_uu_moins_s, pw_uu_moins)
         err = gmm_get (gmmk_pw_vv_moins_s, pw_vv_moins)

!$omp parallel private(i,j,k) shared(uh,vh,wh)
!$omp do
         do k = 1,l_nk
            do j = 1,l_nj
               do i = 1,l_ni
                  uh(i,j,k) = Dcst_inv_rayt_8 * pw_uu_moins(i,j,k)
                  vh(i,j,k) = Dcst_inv_rayt_8 * pw_vv_moins(i,j,k)
                  wh(i,j,k) = zdt1(i,j,k)
               enddo
            enddo
         enddo
!$omp enddo
!$omp end parallel

      else

!$omp parallel private(i,j,k) shared(uh,vh,wh)
!$omp do
         do k = 1,l_nk
            do j = 1,l_nj
               do i = 1,l_ni
                  uh(i,j,k) = 0.5d0 * ( ut1(i,j,k) + ut0(i,j,k) )
                  vh(i,j,k) = 0.5d0 * ( vt1(i,j,k) + vt0(i,j,k) )
                  wh(i,j,k) = 0.5d0 * (zdt1(i,j,k) + zdt0(i,j,k))
               enddo
            enddo
         enddo
!$omp enddo
!$omp end parallel

         call adv_destagWinds (uh,vh,F_minx,F_maxx,F_miny,F_maxy,F_nk)

      endif

      call adv_thermo2mom (wm,wh,F_ni,F_nj,F_nk,F_minx,F_maxx,F_miny,F_maxy)

!     Destag departure winds

!$omp parallel private(i,j,k) shared(uh,vh,wm)
!$omp do
      do k = 1,l_nk
         do j = 1,l_nj
            do i = 1,l_ni
               F_ud(i,j,k) = uh(i,j,k)
               F_vd(i,j,k) = vh(i,j,k)
               F_wd(i,j,k) = wm(i,j,k)
            enddo
         enddo
      enddo
!$omp enddo
!$omp end parallel

!
!     ---------------------------------------------------------------
!
      return
      end subroutine adv_prepareWinds
