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


!**s/r  pressure_sponge -  Performs vertical blending
!
      subroutine height_sponge ()
      use gmm_vt1
      use gmm_geof
      use gem_options
      use theo_options
      implicit none
#include <arch_specific.hf>
!author 
!     Plante A.           - May 2004 
!
!revision
!
#include "gmm.hf"
#include "glb_ld.cdk"
#include "mtn.cdk"

      type(gmm_metadata) :: mymeta
      integer err,i,j,k
      integer n,istat
      real betav_m(l_minx:l_maxx,l_miny:l_maxy,l_nk),betav_t(l_minx:l_maxx,l_miny:l_maxy,l_nk),ubar

!----------------------------------------------------------------------
 
      istat = gmm_get(gmmk_ut1_s,ut1,mymeta)
      if (GMM_IS_ERROR(istat)) print *,'height_sponge ERROR at gmm_get(ut1)'
      istat = gmm_get(gmmk_vt1_s,vt1,mymeta)
      if (GMM_IS_ERROR(istat)) print *,'height_sponge ERROR at gmm_get(vt1)'
      istat = gmm_get(gmmk_wt1_s,wt1,mymeta)
      if (GMM_IS_ERROR(istat)) print *,'height_sponge ERROR at gmm_get(wt1)'
      istat = gmm_get(gmmk_tt1_s,tt1,mymeta)
      if (GMM_IS_ERROR(istat)) print *,'height_sponge ERROR at gmm_get(tt1)'
      istat = gmm_get(gmmk_st1_s,st1,mymeta)
      if (GMM_IS_ERROR(istat)) print *,'height_sponge ERROR at gmm_get(st1)'
      istat = gmm_get(gmmk_qt1_s,qt1,mymeta)
      if (GMM_IS_ERROR(istat)) print *,'height_sponge ERROR at gmm_get(qt1)'
      istat = gmm_get(gmmk_sls_s,sls,mymeta)
      if (GMM_IS_ERROR(istat)) print *,'height_sponge ERROR at gmm_get(sls)'
 
      call set_betav_2(betav_m,betav_t,st1,sls,l_minx,l_maxx,l_miny,l_maxy,l_nk)

      ubar= mtn_flo

      if(Theo_case_S .ne. 'MTN_SCHAR' ) then
         call apply (ut1, ubar, betav_m, l_minx,l_maxx,l_miny,l_maxy, l_nk)
         call apply (vt1, 0.  , betav_m, l_minx,l_maxx,l_miny,l_maxy, l_nk)
         call apply (qt1, 0.  , betav_m, l_minx,l_maxx,l_miny,l_maxy, l_nk)
         if(Zblen_spngtt_L)then
            call apply_tt(tt1,betav_t,st1,sls,l_minx,l_maxx,l_miny,l_maxy,l_nk)
         endif
      endif

      call apply (wt1, 0., betav_t, l_minx,l_maxx,l_miny,l_maxy, l_nk)
 
!----------------------------------------------------------------------
      return
      end

!=======================================================================


      subroutine apply(ff,value,betav, Minx,Maxx,Miny,Maxy, Nk)
      implicit none
#include <arch_specific.hf>

      integer  Minx,Maxx,Miny,Maxy, Nk 

#include "glb_ld.cdk"

      real ff(Minx:Maxx,Miny:Maxy,Nk),value,betav(Minx:Maxx,Miny:Maxy,Nk)

      integer i,j,k,i0,in,j0,jn 

      i0 = 1
      in = l_ni
      j0 = 1
      jn = l_nj     
      if (l_west ) i0 = 1+pil_w
      if (l_east ) in = l_ni-pil_e
      if (l_south) j0 = 1+pil_s
      if (l_north) jn = l_nj-pil_n

      do k=1,Nk
         do j=j0,jn
            do i=i0,in
               ff(i,j,k)=(1.-betav(i,j,k))*ff(i,j,k)+betav(i,j,k)*value
            enddo
         enddo
      enddo
      
      return

      end
!=======================================================================


      subroutine apply_tt(tt,betav_t, F_s, F_sl,Minx,Maxx,Miny,Maxy, Nk)
      use mtn_options
      use tdpack
      implicit none
#include <arch_specific.hf>

      integer  Minx,Maxx,Miny,Maxy, Nk 

#include "glb_ld.cdk"
#include "mtn.cdk"
#include "type.cdk"
#include "cstv.cdk"
#include "ver.cdk"

      real tt(Minx:Maxx,Miny:Maxy,Nk),F_s(Minx:Maxx,Miny:Maxy)
      real betav_t(Minx:Maxx,Miny:Maxy,Nk)
      real F_sl(Minx:Maxx,Miny:Maxy)

      real capc1,my_tt,a00,a02,tempo,hauteur

      integer i,j,k,i0,in,j0,jn 

      a00 = mtn_nstar * mtn_nstar/grav_8
      capc1 = grav_8*grav_8/(mtn_nstar*mtn_nstar*cpd_8*mtn_tzero)

      i0 = 1
      in = l_ni
      j0 = 1
      jn = l_nj     
      if (l_west ) i0 = 1+pil_w
      if (l_east ) in = l_ni-pil_e
      if (l_south) j0 = 1+pil_s
      if (l_north) jn = l_nj-pil_n

      do k=1,Nk
         do j=j0,jn
            do i=i0,in
               tempo = exp(Ver_a_8%t(k)+Ver_b_8%t(k)*F_s(i,j)+Ver_c_8%t(k)*F_sl(i,j))
               a02 = (tempo/Cstv_pref_8)**cappa_8
               hauteur=-log((capc1-1.+a02)/capc1)/a00
               my_tt=mtn_tzero*((1.-capc1)*exp(a00*hauteur)+capc1)
               tt(i,j,k)=(1.-betav_t(i,j,k))*tt(i,j,k)+ &
                    betav_t(i,j,k)*my_tt
            enddo
         enddo
      enddo
      
      return

      end
!=======================================================================
