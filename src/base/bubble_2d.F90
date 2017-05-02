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

!**s/r bubble_2d - generates initial condition for Robert's bubble
!                 experiment (Robert 1993 JAS)
!
      subroutine bubble_2d(F_u, F_v, F_w, F_t, F_zd, F_s, F_topo, &
                           F_q, pref_tr, suff_tr, Mminx,Mmaxx,Mminy,Mmaxy,nk)
      use gmm_vt1
      implicit none
#include <arch_specific.hf>

      character* (*) pref_tr,suff_tr
      integer Mminx,Mmaxx,Mminy,Mmaxy,nk
      real F_u    (Mminx:Mmaxx,Mminy:Mmaxy,nk), F_v(Mminx:Mmaxx,Mminy:Mmaxy,nk), &
           F_w    (Mminx:Mmaxx,Mminy:Mmaxy,nk), F_t(Mminx:Mmaxx,Mminy:Mmaxy,nk), &
           F_zd   (Mminx:Mmaxx,Mminy:Mmaxy,nk), F_s(Mminx:Mmaxx,Mminy:Mmaxy  ), &
           F_topo (Mminx:Mmaxx,Mminy:Mmaxy)  , F_q(Mminx:Mmaxx,Mminy:Mmaxy,nk)

#include "gmm.hf"
#include "glb_pil.cdk"
#include "glb_ld.cdk"
#include "dcst.cdk"
#include "lun.cdk"
#include "ptopo.cdk"
#include "cstv.cdk" 
#include "geomg.cdk"
#include "grd.cdk"
#include "out3.cdk"
#include "tr3d.cdk"
!#include "vt1.cdk"
#include "theo.cdk"
#include "vtopo.cdk"
#include "zblen.cdk"
#include "type.cdk"
#include "ver.cdk"
#include "schm.cdk"
#include "p_geof.cdk"
#include "bubble.cdk"

      type(gmm_metadata) :: mymeta
      character(len=GMM_MAXNAMELENGTH) :: tr_name
      integer i,j,k,err,istat,ii
      real*8 pp, pi,theta
      real, pointer    , dimension(:,:,:) :: tr
!
!     ---------------------------------------------------------------

      istat = gmm_get (gmmk_sls_s     ,   sls )
!
      sls   (:,:) = 0.0
      F_topo(:,:) = 0.0
      F_s   (:,:) = 0.0
      F_u (:,:,:) = 0.0
      F_v (:,:,:) = 0.0
      F_w (:,:,:) = 0.0
      F_zd(:,:,:) = 0.0
      F_q (:,:,:) = 0.0
!
!---------------------------------------------------------------------
!     Initialize temperature and vertical motion fields
!---------------------------------------------------------------------
!
!---------------------------------------------------------------------
!
       do k=1,g_nk
         do j=1,l_nj
            do i=1,l_ni
               ii=i+l_i0-1
               pp = exp(Ver_a_8%t(k)+Ver_b_8%t(k)*F_s(i,j))
               pi = (pp/Cstv_pref_8)**Dcst_cappa_8
               theta=bubble_theta
               if( (((ii)-bubble_ictr)**2 +((k)-bubble_kctr)**2) .lt. bubble_rad**2 ) &
                        theta=theta+0.5d0
               F_t(i,j,k)=theta*pi
            enddo
         enddo
      enddo        
!
!-----------------------------------------------------------------------
!     create tracers (humidity and MTN)
!-----------------------------------------------------------------------
      do k=1,Tr3d_ntr
         if (Tr3d_name_S(k)(1:2).eq.'HU') then
            nullify(tr)
            tr_name = trim(pref_tr)//trim(Tr3d_name_S(k))//trim(suff_tr)
            istat = gmm_get(tr_name,tr,mymeta)
            tr = 0.
         endif
      end do
!
 9000 format(/,'CREATING INPUT DATA FOR MOUNTAIN WAVE THEORETICAL CASE' &
            /,'======================================================')
!
!     -----------------------------------------------------------------
!
      return
      end
