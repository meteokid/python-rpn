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
module mtn_options
   use grid_options
   use gem_options
   use tdpack
   implicit none
   public
   save

   !#
   integer :: mtn_ni = 401
   namelist /mtn_cfgs/ mtn_ni
   !#
   integer :: mtn_nj = 1
   namelist /mtn_cfgs/ mtn_nj
   !#
   integer :: mtn_gnk = 65
   namelist /mtn_cfgs/ mtn_gnk
   !#
   real :: mtn_dx = 500.
   namelist /mtn_cfgs/ mtn_dx
   !#
   real :: mtn_tzero = 303.16
   namelist /mtn_cfgs/ mtn_tzero
   !#
   real :: mtn_flo = 10.
   namelist /mtn_cfgs/ mtn_flo
   !#
   real :: mtn_hght_top = 19500.
   namelist /mtn_cfgs/ mtn_hght_top
   !#
   real :: mtn_hwx = 10.
   namelist /mtn_cfgs/ mtn_hwx
   !#
   real :: mtn_hwx1 = 8.
   namelist /mtn_cfgs/ mtn_hwx1
   !#
   real :: mtn_hght = 250.
   namelist /mtn_cfgs/ mtn_hght
   !#
   real :: mtn_nstar = 0.01
   namelist /mtn_cfgs/ mtn_nstar
   !#
   real*8 :: mtn_pref_8 = 100000.d0
   namelist /mtn_cfgs/ mtn_pref_8

contains

      integer function mtn_nml (F_namelistf_S)
      implicit none

      character* (*) F_namelistf_S

#include "lun.cdk"
#include "dcst.cdk"

      integer, external :: fnom
      integer unf
!
!-------------------------------------------------------------------
!
      mtn_nml = -1

      if ((F_namelistf_S.eq.'print').or.(F_namelistf_S.eq.'PRINT')) then
         mtn_nml = 0
         if ( Lun_out.ge.0) write (Lun_out,nml=mtn_cfgs)
         return
      endif

      if (F_namelistf_S .ne. '') then

         unf = 0
         if (fnom (unf,F_namelistf_S, 'SEQ+OLD', 0) .ne. 0) goto 9110
         rewind(unf)
         read (unf, nml=mtn_cfgs, end= 1000, err=9130)
 1000    call fclos (unf)

      endif

      mtn_nml = 0
      ! establish horizontal grid configuration
      Grd_typ_S='LU'
      Grd_ni = mtn_ni ; Grd_nj = mtn_nj
      Grd_dx = (mtn_dx/Dcst_rayt_8)*(180./pi_8)
      Grd_dy = Grd_dx
      Grd_latr = 0.
      Grd_lonr = (mtn_ni/2 + 20) * Grd_dx
      Grd_maxcfl = 3

      goto 9999

 9110 if (Lun_out.gt.0) then
         write (Lun_out, 9050) trim( F_namelistf_S )
         write (Lun_out, 8000)
      endif
      goto 9999

 9130 call fclos (unf)
      if (Lun_out.ge.0) then
         write (Lun_out, 9150) 'mtn_cfgs',trim( F_namelistf_S )
         write (Lun_out, 8000)
      endif

 8000 format (/,'========= ABORT IN S/R mtn_nml.f ============='/)
 9050 format (/,' FILE: ',A,' NOT AVAILABLE'/)
 9150 format (/,' NAMELIST ',A,' INVALID IN FILE: ',A/)

 9999 return
      end function mtn_nml
!
!-------------------------------------------------------------------
!
      integer function mtn_cfg()
      implicit none
#include <arch_specific.hf>

#include "glb_ld.cdk"
#include "cstv.cdk"

      integer k
      real*8 c1_8,Exner_8,height_8,pres_8
!
!     ---------------------------------------------------------------
!
      mtn_cfg = -1

      G_nk = mtn_gnk

      Zblen_hmin = mtn_hght_top - Zblen_spngthick

      if (hyb(1) .lt. 0.) then
         if(mtn_nstar.lt.0.)            &!  isothermal case
            mtn_nstar=grav_8/sqrt(cpd_8*mtn_tzero)
         if(mtn_nstar.eq.0.0) then     !  isentropic case
            c1_8=grav_8/(cpd_8*mtn_tzero)
            Exner_8=1.d0-c1_8*mtn_hght_top
         else
            c1_8=grav_8**2/(cpd_8*mtn_tzero*mtn_nstar**2)
            Exner_8=1.d0-c1_8+c1_8 &
                       *exp(-mtn_nstar**2/grav_8*mtn_hght_top)
         endif
         Cstv_ptop_8 = Exner_8**(1.d0/cappa_8)*mtn_pref_8
!        Uniform distribution of levels in terms of height
         do k=1,G_nk
            height_8=mtn_hght_top*(1.d0-(dble(k)-.5d0)/G_nk)
            Exner_8=1.d0-c1_8+c1_8*exp(-mtn_nstar**2/grav_8*height_8)
            pres_8=Exner_8**(1.d0/cappa_8)*mtn_pref_8
            hyb(k)=(pres_8-Cstv_ptop_8)/(mtn_pref_8-Cstv_ptop_8)
         enddo

!        denormalize
         do k=1,G_nk
            hyb(k) = hyb(k) + (1.-hyb(k))*Cstv_ptop_8/mtn_pref_8
         enddo
      else
         do k=1024,1,-1
         if(hyb(k).lt.0) G_nk=k-1
         enddo
      endif

      mtn_cfg = 1

      return
      end function mtn_cfg
!
!     ---------------------------------------------------------------
!
      subroutine mtn_data ( F_u, F_v, F_w, F_t, F_zd, F_s, F_topo,&
                            F_q, pref_tr, suff_tr                ,&
                            Mminx,Mmaxx,Mminy,Mmaxy,F_theocase_S )
      use gmm_vt1
      use gmm_geof
      use geomh
      implicit none
#include <arch_specific.hf>

      character* (*) pref_tr,suff_tr,F_theocase_S
      integer Mminx,Mmaxx,Mminy,Mmaxy
      real F_u    (Mminx:Mmaxx,Mminy:Mmaxy,*), &
           F_v    (Mminx:Mmaxx,Mminy:Mmaxy,*), &
           F_w    (Mminx:Mmaxx,Mminy:Mmaxy,*), &
           F_t    (Mminx:Mmaxx,Mminy:Mmaxy,*), &
           F_zd   (Mminx:Mmaxx,Mminy:Mmaxy,*), &
           F_s    (Mminx:Mmaxx,Mminy:Mmaxy  ), &
           F_topo (Mminx:Mmaxx,Mminy:Mmaxy  ), &
           F_q    (Mminx:Mmaxx,Mminy:Mmaxy,*)

#include "gmm.hf"
#include "glb_pil.cdk"
#include "glb_ld.cdk"
#include "lun.cdk"
#include "ptopo.cdk"
#include "out3.cdk"
#include "tr3d.cdk"
#include "type.cdk"
#include "ver.cdk"
#include "cstv.cdk"
#include "dcst.cdk"

      type(gmm_metadata) :: mymeta
      character(len=GMM_MAXNAMELENGTH) :: tr_name
      integer i,j,k,i00,err,istat
      real a00, a01, a02, xcntr, zdi, zfac, zfac1, capc1
      real, allocatable, dimension(:,:) :: psurf, topo_ls
      real hauteur, tempo, dx, slp, slpmax
      real*8 temp1, temp2
      real, pointer, dimension(:,:,:) :: tr
!
!     ---------------------------------------------------------------
!
      allocate ( psurf  (l_minx:l_maxx,l_miny:l_maxy),&
                 topo_ls(l_minx:l_maxx,l_miny:l_maxy) )

!---------------------------------------------------------------------
!     Initialize orography
!---------------------------------------------------------------------

      xcntr = int(float(Grd_ni-1)*0.5)+1
      do j=1,l_nj
      do i=1,l_ni
         i00 = i + l_i0 - 1
         zdi  = float(i00)-xcntr
         zfac = (zdi/mtn_hwx)**2
         if ( F_theocase_S .eq. 'MTN_SCHAR' &
             .or.  F_theocase_S .eq. 'MTN_SCHAR2' ) then
            zfac1= pi_8 * zdi / mtn_hwx1
            F_topo(i,j) = mtn_hght* exp(-zfac) * cos(zfac1)**2
            ! Note : get_s_large_scale takes topo_ls in m2/s2
            topo_ls(i,j)= mtn_hght/2.* exp(-zfac)*grav_8
         else
            F_topo(i,j) = mtn_hght/(zfac + 1.)
            topo_ls(i,j)= mtn_hght/2./(zfac + 1.)*grav_8
         endif
      enddo
      enddo

      call get_s_large_scale ( topo_ls, Mminx,Mmaxx,Mminy,Mmaxy )
      istat = gmm_get (gmmk_sls_s, sls)

!---------------------------------------------------------------------
!     If time-dependant topography
!---------------------------------------------------------------------
      if(Vtopo_L) then
         istat = gmm_get(gmmk_topo_low_s , topo_low , mymeta)
         istat = gmm_get(gmmk_topo_high_s, topo_high, mymeta)
         topo_low (1:l_ni,1:l_nj) = 0.
         topo_high(1:l_ni,1:l_nj) = F_topo(1:l_ni,1:l_nj) * grav_8
         F_topo   (1:l_ni,1:l_nj) = 0.
       endif

      if (      F_theocase_S .eq. 'MTN_SCHAR'  &
           .or. F_theocase_S .eq. 'MTN_SCHAR2' &
           .or. F_theocase_S .eq. 'MTN_PINTYNL') then

!---------------------------------------------------------------------
!     Generate surface pressure field and its logarithm (s)
!     Set wind imags (u,v)
!     Transform orography from geometric to geopotential height
!     Set non-hydrostatic perturbation pressure (q)
!---------------------------------------------------------------------
!
      a00 = mtn_nstar**2/grav_8
      a01 = (cpd_8*mtn_tzero*a00)/grav_8
      capc1 = grav_8**2/(mtn_nstar**2*cpd_8*mtn_tzero)
!
      do j=1,l_nj
      do i=1,l_ni
         psurf(i,j)=Cstv_pref_8*(1.-capc1 &
               +capc1*exp(-a00*F_topo(i,j)))**(1./cappa_8)
         F_s   (i,j)   = log(psurf(i,j)/Cstv_pref_8)
         F_u   (i,j,1) = mtn_flo
      enddo
      enddo
!
      do k=1,g_nk
      do j=1,l_nj
      do i=1,l_ni
         F_u (i,j,k) = F_u (i,j,1)
         F_v (i,j,k) = 0.0
      enddo
      enddo
      enddo
!
!---------------------------------------------------------------------
!     Initialize temperature and vertical motion fields
!---------------------------------------------------------------------
!
       do k=1,g_nk
         do j=1,l_nj
            do i=1,l_ni
               tempo = exp(Ver_a_8%m(k)+Ver_b_8%m(k)*F_s(i,j)+Ver_c_8%m(k)*sls(i,j))
               a02 = (tempo/Cstv_pref_8)**cappa_8
               hauteur=-log((capc1-1.+a02)/capc1)/a00
               temp1=mtn_tzero*((1.-capc1)*exp(a00*hauteur)+capc1)
               tempo = exp(Ver_a_8%m(k+1)+Ver_b_8%m(k+1)*F_s(i,j)+Ver_c_8%m(k+1)*sls(i,j))
               a02 = (tempo/Cstv_pref_8)**cappa_8
               hauteur=-log((capc1-1.+a02)/capc1)/a00
               temp2=mtn_tzero*((1.-capc1)*exp(a00*hauteur)+capc1)
               F_t(i,j,k)=Ver_wp_8%t(k)*temp2+Ver_wm_8%t(k)*temp1
            enddo
         enddo
      enddo
!
      else   ! MTN_PINTY or MTN_PINTY2 or NOFLOW
!-----------------------------------------------------------------------
!     Generate pressure from Cstv_ptop_8, Cstv_pref_8, and coordinate
!     Generate corresponding geopotential for isothermal atmosphere
!     Set wind and temperature
!-----------------------------------------------------------------------
      do k=1,g_nk
      do j=1,l_nj
      do i=1,l_ni
         F_t (i,j,k) = mtn_tzero
      enddo
      enddo
      enddo

      do k=1,g_nk
      do j=1,l_nj
      do i=1,l_ni
         F_u (i,j,k) = mtn_flo
         F_v (i,j,k) = 0.
      enddo
      enddo
      enddo

      do j=1,l_nj
      do i=1,l_ni
         psurf(i,j) = Cstv_pref_8 *  &
                      exp( -grav_8 * F_topo(i,j)/ &
                           (Rgasd_8 * mtn_tzero ) )
         F_s(i,j) = log(psurf(i,j)/Cstv_pref_8)
      enddo
      enddo

!     calculate maximum mountain slope

      slpmax=0
      dx=Dcst_rayt_8*Grd_dx*pi_8/180.
      do j=1,l_nj
      do i=1,l_ni
      slp=abs(F_topo(i,j)-F_topo(i-1,j))/dx
      slpmax=max(slp,slpmax)
      enddo
      enddo
      slpmax=(180.d0/pi_8)*atan(slpmax)

      print*,"SLPMAX=",slpmax," DEGREES"

      endif
!
!-----------------------------------------------------------------------
!     Transform orography from geometric to geopotential height
!-----------------------------------------------------------------------
      do j=1,l_nj
      do i=1,l_ni
         F_topo(i,j) = grav_8 * F_topo(i,j)
      end do
      end do
!
      call rpn_comm_xch_halo ( F_topo, l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj,1, &
                    G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      if(schm_sleve_L)&
           call rpn_comm_xch_halo ( sls, l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj,1, &
           G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )

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
      end subroutine mtn_data

end module mtn_options
