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
module bubble_options
   use dynkernel_options
   use grid_options
   use gem_options
   use tdpack
   implicit none
   public
   save

   !#
   integer :: bubble_ni = 101
   namelist /bubble_cfgs/ bubble_ni
   !#
   integer :: bubble_nj = 1
   namelist /bubble_cfgs/ bubble_nj
   !#
   integer :: bubble_nk = 100
   namelist /bubble_cfgs/ bubble_nk
   !#
   real :: bubble_dx = 10.
   namelist /bubble_cfgs/ bubble_dx
   !#
   real :: bubble_dz = 10.
   namelist /bubble_cfgs/ bubble_dz
   !#
   real :: bubble_theta = 303.16
   namelist /bubble_cfgs/  bubble_theta
   !#
   integer :: bubble_rad = 25
   namelist /bubble_cfgs/ bubble_rad
   !#
   integer :: bubble_ictr = -1
   namelist /bubble_cfgs/ bubble_ictr
   !#
   integer :: bubble_kctr = -1
   namelist /bubble_cfgs/ bubble_kctr

contains

      integer function bubble_nml (F_namelistf_S)
      use dcst
      use lun
      implicit none

      character(len=*) F_namelistf_S



      integer, external :: fnom
      integer unf
!
!-------------------------------------------------------------------
!
      bubble_nml = -1

      if ((F_namelistf_S == 'print').or.(F_namelistf_S == 'PRINT')) then
         bubble_nml = 0
         if ( Lun_out >= 0) write (Lun_out,nml=bubble_cfgs)
         return
      endif

      if (F_namelistf_S /= '') then

         unf = 0
         if (fnom (unf,F_namelistf_S, 'SEQ+OLD', 0) /= 0) goto 9110
         rewind(unf)
         read (unf, nml=bubble_cfgs, end= 1000, err=9130)

      endif

 1000 bubble_nml = 0

      ! establish horizontal grid configuration
      Dcst_rayt_8= Dcst_rayt_8*0.1d0 ! an accuracy problem
      Dcst_inv_rayt_8 = Dcst_inv_rayt_8 * 10.d0 ! an accuracy problem
      Grd_typ_S='LU'
      Grd_ni = bubble_ni ; Grd_nj = bubble_nj
      Grd_dx = (bubble_dx/Dcst_rayt_8)*(180./pi_8)
      Grd_dy = Grd_dx
      Grd_latr = 0.
      Grd_lonr = (bubble_ni/2 + 20) * Grd_dx
      Grd_maxcfl = 3

      goto 9999

 9110 if (Lun_out > 0) then
         write (Lun_out, 9050) trim( F_namelistf_S )
         write (Lun_out, 8000)
      endif
      goto 9999

 9130 if (Lun_out >= 0) then
         write (Lun_out, 9150) 'bubble_cfgs',trim( F_namelistf_S )
         write (Lun_out, 8000)
      endif

 8000 format (/,'========= ABORT IN S/R bubble_nml.f ============='/)
 9050 format (/,' FILE: ',A,' NOT AVAILABLE'/)
 9150 format (/,' NAMELIST ',A,' INVALID IN FILE: ',A/)

 9999 call fclos (unf)
      return
      end function bubble_nml
!
!-------------------------------------------------------------------
!
      integer function bubble_cfg()
      use glb_ld
      use cstv
      implicit none
#include <arch_specific.hf>

      integer k
      real*8 c1_8,Exner_8,height_8,pres_8,pref_8,ptop_8,htop_8
!
!     ---------------------------------------------------------------
!
      bubble_cfg = -1

      pref_8 = 1.d5

      G_nk   = bubble_nk
      htop_8 = G_nk*bubble_dz

      if ( hyb(1) < 0 ) then

        !isentropic case
         c1_8=grav_8/(cpd_8*bubble_theta)
         Exner_8=1.d0-c1_8*htop_8
         ptop_8 = Exner_8**(1.d0/cappa_8)*pref_8
!        Uniform distribution of levels in terms of height
         do k=1,G_nk
            height_8=htop_8*(1.d0-(dble(k)-.5d0)/G_nk)
            Exner_8=1.d0-c1_8*height_8
            pres_8=Exner_8**(1.d0/cappa_8)*pref_8
            hyb(k)=(pres_8-ptop_8)/(pref_8-ptop_8)
            hyb(k) = hyb(k) + (1.-hyb(k))*ptop_8/pref_8
            if (trim(Dynamics_Kernel_S) == 'DYNAMICS_FISL_H') hyb(k)=height_8
         enddo

      else

         do k=1024,1,-1
            if(hyb(k) < 0) G_nk=k-1
         enddo

      endif

      if (bubble_ictr < 0) bubble_ictr = int(float(Grd_ni-1)*0.5)+1
      if (bubble_kctr < 0) bubble_kctr = G_nk - bubble_rad - 1

      bubble_cfg = 1

      return
      end function bubble_cfg
!
!-------------------------------------------------------------------
!

!**s/r bubble_data - generates initial condition for Robert's bubble
!                  experiment (Robert 1993 JAS)
!
      subroutine bubble_data ( F_u, F_v, F_t, F_s, F_q, F_topo,&
                               Mminx,Mmaxx,Mminy,Mmaxy,nk )
      use gmm_geof
      use geomh
      use gmm_itf_mod
      use glb_pil
      use glb_ld
      use lun
      use ptopo
      use type_mod
      use ver
      use cstv
      use metric
      implicit none
#include <arch_specific.hf>

      integer Mminx,Mmaxx,Mminy,Mmaxy,nk
      real F_u    (Mminx:Mmaxx,Mminy:Mmaxy,nk), &
           F_v    (Mminx:Mmaxx,Mminy:Mmaxy,nk), &
           F_t    (Mminx:Mmaxx,Mminy:Mmaxy,nk), &
           F_s    (Mminx:Mmaxx,Mminy:Mmaxy   ), &
           F_topo (Mminx:Mmaxx,Mminy:Mmaxy   ), &
           F_q    (Mminx:Mmaxx,Mminy:Mmaxy,nk+1)

      integer i,j,k,istat,ii
      real*8 pp,ex,theta
!
!     ---------------------------------------------------------------
!
      istat = gmm_get (gmmk_sls_s     ,   sls )

      sls   (:,:) = 0.0
      F_topo(:,:) = 0.0
      F_s   (:,:) = 0.0
      F_u (:,:,:) = 0.0
      F_v (:,:,:) = 0.0
      F_q (:,:,:) = 0.0
!
!---------------------------------------------------------------------
!     Initialize temperature
!---------------------------------------------------------------------
!
      if (trim(Dynamics_Kernel_S) == 'DYNAMICS_FISL_P') then


         do k=1,g_nk
            do j=1,l_nj
            do i=1,l_ni
               ii=i+l_i0-1
               theta=bubble_theta
               if ( (((ii)-bubble_ictr)**2 +((k)-bubble_kctr)**2) < bubble_rad**2 ) then
                  theta=theta+0.5d0
               end if
               pp = exp(Ver_a_8%t(k)+Ver_b_8%t(k)*F_s(i,j))
               ex = (pp/Cstv_pref_8)**cappa_8
               F_t(i,j,k)=theta*ex
            enddo
            enddo
         enddo

      else if (trim(Dynamics_Kernel_S) == 'DYNAMICS_FISL_H') then

         call set_metric

         do k=1,g_nk
            do j=1,l_nj
            do i=1,l_ni
               ii=i+l_i0-1
               theta=bubble_theta
               if ( (((ii)-bubble_ictr)**2 +((k)-bubble_kctr)**2) < bubble_rad**2 ) then
                  theta=theta+0.5d0
               end if
               ex=1.d0-grav_8/(cpd_8*bubble_theta)*Ver_z_8%t(k)
               F_t(i,j,k)=theta*ex
            enddo
            enddo
         enddo
!
!---------------------------------------------------------------------
!     Initialize (horizontally uniform) pressure
!---------------------------------------------------------------------
!
         do k=1,g_nk+1
            do j=1,l_nj
            do i=1,l_ni
               ii=i+l_i0-1
               ex=1.d0-grav_8/(cpd_8*bubble_theta)*Ver_z_8%m(k)
               pp=1.d5*ex**(1.d0/cappa_8)
               F_q(i,j,k)=rgasd_8*Cstv_Tstr_8*log(pp/1.d5)+grav_8*Ver_z_8%m(k)
            enddo
            enddo
         enddo

      endif
!
 9000 format(/,'CREATING INPUT DATA FOR MOUNTAIN WAVE THEORETICAL CASE' &
            /,'======================================================')
!
!     -----------------------------------------------------------------
!
      return
      end subroutine bubble_data

end module bubble_options
