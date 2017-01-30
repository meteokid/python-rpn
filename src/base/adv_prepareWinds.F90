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

#include "gmm.hf"
#include "glb_ld.cdk"
#include "schm.cdk"
#include "vt2.cdk"
#include "pw.cdk"
#include "dcst.cdk"

      real, dimension(:,:,:), allocatable :: uh,vh,wm,wh
      real ::  beta, err
      integer :: i,j,k
      real*8  :: inv_rayt_8
!
!     ---------------------------------------------------------------
!      
    inv_rayt_8 = 1.D0 / Dcst_rayt_8

      allocate ( uh(F_minx:F_maxx,F_miny:F_maxy,F_nk), &
                 vh(F_minx:F_maxx,F_miny:F_maxy,F_nk), &
                 wm(F_minx:F_maxx,F_miny:F_maxy,F_nk), &
                 wh(F_minx:F_maxx,F_miny:F_maxy,F_nk) )

      if(Schm_trapeze_L) then
        
         uh = ut0
         vh = vt0
  
         call adv_destagWinds (uh,vh,F_minx,F_maxx,F_miny,F_maxy,F_nk)

         
         wm = 0.
         

         call adv_thermo2mom  (wm,zdt0,F_ni,F_nj,F_nk,F_minx,F_maxx,F_miny,F_maxy)
       
!     Unstaggered arrival winds
         F_ua = uh(1:F_ni,1:F_nj,1:F_nk)
         F_va = vh(1:F_ni,1:F_nj,1:F_nk)
         F_wa = wm(1:F_ni,1:F_nj,1:F_nk)
         F_wat= zdt0(1:F_ni,1:F_nj,1:F_nk)

! DEPARTURE WINDS: NO DESTAGRING         
         err = gmm_get (gmmk_pw_uu_moins_s, pw_uu_moins)
         err = gmm_get (gmmk_pw_vv_moins_s, pw_vv_moins)

         do k = 1,l_nk
            do j = 1,l_nj
            do i = 1,l_ni
               uh(i,j,k) = inv_rayt_8 * pw_uu_moins(i,j,k)
               vh(i,j,k) = inv_rayt_8 * pw_vv_moins(i,j,k)
            enddo
            enddo
         enddo

         wh = zdt1

      else

         uh (:,:,:) = .5*( ut1(:,:,:) + ut0(:,:,:) )
         vh (:,:,:) = .5*( vt1(:,:,:) + vt0(:,:,:) )
         wh (:,:,:) = .5*(zdt1(:,:,:) +zdt0(:,:,:) )

         call adv_destagWinds (uh,vh,F_minx,F_maxx,F_miny,F_maxy,F_nk)

      endif
  
      call adv_thermo2mom  (wm,wh,F_ni,F_nj,F_nk,F_minx,F_maxx,F_miny,F_maxy)

!     Destag departure winds
      F_ud=uh
      F_vd=vh
      F_wd=wm
      
      deallocate(uh,vh,wm,wh)
!     
!     ---------------------------------------------------------------
!     
      return
      end subroutine adv_prepareWinds
