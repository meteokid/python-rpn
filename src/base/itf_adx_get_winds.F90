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

subroutine itf_adx_get_winds2 ( F_ud, F_vd, F_wd, F_ua, F_va, F_wa, F_wat, &	
                                F_minx,F_maxx,F_miny,F_maxy,F_nk         , &
                                F_lni,F_lnj,F_nk_winds )
   implicit none
#include <arch_specific.hf>
   !@objective process winds in preparation for advection
   !@arguments

   integer :: F_minx,F_maxx,F_miny,F_maxy,F_nk,F_lni,F_lnj,F_nk_winds
   real, dimension(F_minx:F_maxx,F_miny:F_maxy,F_nk_winds) :: &
                                        F_ud, F_vd, F_wd    !O, model de-stag winds
   real, dimension(F_lni,F_lnj,F_nk) :: F_ua,F_va,F_wa,F_wat

   !@author alain patoine
   !@revisions
   ! v2_31 - Desgagne M.       - removed stkmemw
   ! v3_00 - Desgagne & Lee    - Lam configuration
   ! v3_10 - Corbeil & Desgagne & Lee - AIXport+Opti+OpenMP
   ! v3_21 - Desgagne M.       - Revision OpenMP
   ! v4_   - Gravel S.         - Staggered version
   ! v4_10 - Plante A.         - Add interpolation of wind on their non-native
   ! Jan 2010, S. Chamberland: split out of adw and into smaller/logical units
   ! v4_40 - Qaddouri/Lee - Yin-Yang, to exchange de-stag winds,use global range
   !@description
   !

#include "glb_ld.cdk"
#include "gmm.hf"
#include "vth.cdk"
#include "vt0.cdk"
#include "vt1.cdk"
#include "vt2.cdk"
#include "schm.cdk"

   integer :: i,j,k,k2,k2m1,err
   real :: poid, beta
   real, dimension(:,:,:), allocatable :: ut,uh,vt,vh,wm,wh

   !---------------------------------------------------------------------

   err = gmm_get(gmmk_ut0_s ,  ut0)
   err = gmm_get(gmmk_vt0_s ,  vt0)
   err = gmm_get(gmmk_zdt0_s, zdt0)
   err = gmm_get(gmmk_ut1_s ,  ut1)
   err = gmm_get(gmmk_vt1_s ,  vt1)
   err = gmm_get(gmmk_zdt1_s, zdt1)

   if(Schm_step_settls_L) then
      err = gmm_get(gmmk_ut2_s ,  ut2)
      err = gmm_get(gmmk_vt2_s ,  vt2)
      err = gmm_get(gmmk_zdt2_s, zdt2)
   endif

   allocate ( ut(l_minx:l_maxx,l_miny:l_maxy,l_nk+1), &
              uh(l_minx:l_maxx,l_miny:l_maxy,l_nk  ), &
              vt(l_minx:l_maxx,l_miny:l_maxy,l_nk+1), &
              vh(l_minx:l_maxx,l_miny:l_maxy,l_nk  ), &
              wm(l_minx:l_maxx,l_miny:l_maxy,l_nk  ), &
              wh(l_minx:l_maxx,l_miny:l_maxy,l_nk) )

   if(Schm_trapeze_L.and..NOT.Schm_step_settls_L) then
      uh = ut0 ; vh = vt0
      call itf_adx_destag_winds2 (uh,vh,l_minx,l_maxx,l_miny,l_maxy,l_nk)
      call itf_adx_interp_thermo2mom2 (wm,zdt0,l_minx,l_maxx,l_miny,l_maxy,l_nk)
      F_ua =  uh(1:l_ni,1:l_nj,1:l_nk)
      F_va =  vh(1:l_ni,1:l_nj,1:l_nk)
      F_wa =  wm(1:l_ni,1:l_nj,1:l_nk)
      F_wat=zdt0(1:l_ni,1:l_nj,1:l_nk)
      uh = ut1 ; vh = vt1 ; wh = zdt1
   elseif(Schm_step_settls_L) then

      !Set V_a = V(r,t1)
      !-----------------
      uh = ut1 ; vh = vt1
      call itf_adx_destag_winds2 (uh,vh,l_minx,l_maxx,l_miny,l_maxy,l_nk)
      call itf_adx_interp_thermo2mom2 (wm,zdt1,l_minx,l_maxx,l_miny,l_maxy,l_nk)
      F_ua =  uh(1:l_ni,1:l_nj,1:l_nk)
      F_va =  vh(1:l_ni,1:l_nj,1:l_nk)
      F_wa =  wm(1:l_ni,1:l_nj,1:l_nk)
      F_wat=zdt1(1:l_ni,1:l_nj,1:l_nk)
       
      !Set V_d = 2*V(r,t1)-V(r,t2)
      !---------------------------
      uh = 2.*ut1-ut2 ; vh = 2.*vt1-vt2 ; wh = 2.*zdt1-zdt2

      !SETTLS limiter according to Diamantakis
      if(Schm_settls_lim_L) then
         beta=1.9
         !do k=12,20 ! was sufficient on test case
         do k=1,l_nk
            do j=1,l_nj
            do i=1,l_ni
               if(abs(zdt1(i,j,k)-zdt2(i,j,k)).gt.beta*0.5*(abs(zdt1(i,j,k))+abs(zdt2(i,j,k)))) then
                  wh(i,j,k)=zdt1(i,j,k)
               endif
            enddo
            enddo
         enddo
      endif

   else
      uh (:,:,:) = .5*( ut1(:,:,:) + ut0(:,:,:) )
      vh (:,:,:) = .5*( vt1(:,:,:) + vt0(:,:,:) )
      wh (:,:,:) = .5*(zdt1(:,:,:) +zdt0(:,:,:) )
   endif

   call itf_adx_destag_winds2 (uh,vh,l_minx,l_maxx,l_miny,l_maxy,l_nk)

   if (Schm_superwinds_L) then

      call itf_adx_interp_mom2thermo2 (ut,uh,l_minx,l_maxx,l_miny,l_maxy,l_nk)
      call itf_adx_interp_mom2thermo2 (vt,vh,l_minx,l_maxx,l_miny,l_maxy,l_nk)
      call itf_adx_interp_thermo2mom2 (wm,wh(l_minx,l_miny,1), &
                                       l_minx,l_maxx,l_miny,l_maxy,l_nk)
      k2= 0
      do k = 1, F_nk_winds, 2
         k2 = k2+1
         k2m1=max(k2-1,1)
         poid=1.
         if(k.eq.1) poid=0.
         do j = l_miny, l_maxy
         do i = l_minx, l_maxx
            F_ud(i,j,k) =   ut(i,j,k2)
            F_vd(i,j,k) =   vt(i,j,k2)
            F_wd(i,j,k) =   wh(i,j,k2m1)*poid
         enddo
         enddo
      enddo
      k2= 0
      do k = 2, F_nk_winds, 2
         k2 = k2+1
         do j = l_miny, l_maxy
         do i = l_minx, l_maxx
            F_ud(i,j,k) = uh(i,j,k2)
            F_vd(i,j,k) = vh(i,j,k2)
            F_wd(i,j,k) = wm(i,j,k2)
         enddo
         enddo
      enddo

   else

      call itf_adx_interp_thermo2mom2 (wm,wh,l_minx,l_maxx,l_miny,l_maxy,l_nk)
      F_ud=uh; F_vd=vh; F_wd=wm

   endif

   deallocate(ut,uh,vt,vh,wm,wh)

   !---------------------------------------------------------------------
   return

end subroutine itf_adx_get_winds2




