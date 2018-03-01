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

!**s/r ac_posi - find the positional points to extract cascade grid
!                from the current model grid configuration

      subroutine ac_posi (xp,yp,dimgx,dimgy,prout)
      use grid_options
      use grdc_options
      use gem_options
      use tdpack
      use glb_ld
      use lun
      use tr3d
      use rstr
      use glb_pil
      implicit none
#include <arch_specific.hf>

      logical, intent(in) :: prout
      integer, intent(in) :: dimgx,dimgy
      real*8, intent(in) :: xp(dimgx), yp(dimgy)

!author
!        Michel Desgagne - 2001 (from MC2)
!revision
! v3_30 - Lee V.       - initial version for GEMDM
! v3_30 - McTaggart-Cowan R.- Allow for user-defined domain tag extensions
! v4_03 - Lee/Desgagne - ISST
!


      integer, external :: stretch_axis2
      logical flag_hu
      integer i,k,cnt,ierx,dum1,dum2
      real x0, xl, y0, yl, dum
      real*8, dimension(:), allocatable:: ac_xp,ac_yp
      real*8 xpx(dimgx), ypx(dimgy), rad2deg_8
!
!---------------------------------------------------------------------
!
      rad2deg_8 = 180.0d0/pi_8
      xpx = xp * rad2deg_8
      ypx = yp * rad2deg_8

      Grdc_gid = 0
      Grdc_gjd = 0
      Grdc_gif = 0
      Grdc_gjf = 0

      flag_hu = .false.

      if ( (Grdc_ndt < 0) .or. (Grd_yinyang_S == 'YAN') .or. &
           (Grdc_ni == 0) .or. (Grdc_nj == 0) .or. (Grdc_dx < 0.) ) then
         Grdc_ndt = -1
         return
      endif

      if (Grdc_dy < 0) Grdc_dy = Grdc_dx

!     Calculate the rest of Grdc parameters in here like grid_nml

      Grdc_pil = Grdc_maxcfl + Grd_bsc_base + Grd_bsc_ext1
      if ((Grdc_iref==-1) .and. (Grdc_jref==-1)) then
         Grdc_iref = Grdc_ni / 2 + Grdc_pil
         if (mod(Grdc_ni,2)==0) then
            Grdc_lonr = dble(Grdc_lonr) - dble(Grdc_dx)/2.d0
         else
            Grdc_iref = Grdc_iref + 1
         endif
         Grdc_jref = Grdc_nj / 2 + Grdc_pil
         if (mod(Grdc_nj,2)==0) then
            Grdc_latr = dble(Grdc_latr) - dble(Grdc_dy)/2.d0
         else
            Grdc_jref = Grdc_nj / 2 + Grdc_pil + 1
         endif
      else
         Grdc_iref = Grdc_iref + Grdc_pil
         Grdc_jref = Grdc_jref + Grdc_pil
         if (Grdc_iref < 1 .or. Grdc_iref > Grdc_ni .or. &
            Grdc_jref < 1 .or. Grdc_jref > Grdc_nj) then
            if (prout) then
               write (6,1002)Grdc_ni,Grdc_nj,Grdc_iref,Grdc_jref
            end if
            Grdc_ndt = -1
            return
         endif
      endif
      Grdc_ni = Grdc_ni + 2*Grdc_pil
      Grdc_nj = Grdc_nj + 2*Grdc_pil
      allocate(ac_xp(Grdc_ni),ac_yp(Grdc_nj))
!
!     *** Positional parameters for f and q points
!
      x0   = Grdc_lonr - (Grdc_iref-1) * Grdc_dx
      y0   = Grdc_latr - (Grdc_jref-1) * Grdc_dy
      xl   = x0 + (Grdc_ni  -1) * Grdc_dx
      yl   = y0 + (Grdc_nj  -1) * Grdc_dy

      if (x0 < 0.0)x0=x0+360.0
      if (xl < 0.0)xl=xl+360.0

      ierx = stretch_axis2 ( ac_xp, Grdc_dx, x0, xl, dum1, Grdc_ni, &
                 Grdc_ni, dum, .false.,Lun_debug_L,360., dum2, .false.)
      ierx = stretch_axis2 ( ac_yp, Grdc_dy, y0, yl, dum1, Grdc_nj, &
                 Grdc_nj, dum, .false.,Lun_debug_L,180., dum2, .false.)

      Grdc_xp1 = ac_xp(1)
      Grdc_yp1 = ac_yp(1)

      do i=1,dimgx
         if (xpx(i) <= ac_xp(1)      ) Grdc_gid=i
         if (xpx(i) <= ac_xp(Grdc_ni)) Grdc_gif=i
      enddo

      do i=1,dimgy
         if (ypx(i) <= ac_yp(1)      ) Grdc_gjd=i
         if (ypx(i) <= ac_yp(Grdc_nj)) Grdc_gjf=i
      enddo
      deallocate(ac_xp,ac_yp)

      ! Tests if same grid
      if( Grdc_iref == Grd_iref.and.&
          Grdc_jref == Grd_jref.and.&
          abs(Grdc_latr-Grd_latr)<1.e-6.and.&
          abs(Grdc_lonr-Grd_lonr)<1.e-6.and.&
          abs(Grdc_dx-Grd_dx)/Grd_dx<1.e-6) then
         write (6,1007)
         Grdc_gid=1
         Grdc_gif=dimgx
         Grdc_gjd=1
         Grdc_gjf=dimgy
      else
         Grdc_gid = Grdc_gid - 2
         Grdc_gjd = Grdc_gjd - 2
         Grdc_gif = Grdc_gif + 3
         Grdc_gjf = Grdc_gjf + 3
      endif

      if ( (Grdc_gid <  1       ).or.(Grdc_gjd <  1       ) .or. &
           (Grdc_gid >= Grdc_gif).or.(Grdc_gjd >= Grdc_gjf) .or. &
           (Grdc_gif >  dimgx   ).or.(Grdc_gjf > dimgy    ) ) Grdc_ndt = -1

      if (Grdc_ndt > 0) then
         if ((prout).and.(.not.Rstri_rstn_L)) then
            write (6,1005) Grdc_gid,Grdc_gif,Grdc_gjd,Grdc_gjf,Grdc_ndt,Grdc_start,Grdc_end
            write (6,1006) Grdc_gif-Grdc_gid+1,xpx(Grdc_gid),xpx(Grdc_gif),&
                           Grdc_gjf-Grdc_gjd+1,ypx(Grdc_gjd),ypx(Grdc_gjf)

            write(6,1100) 'LU', &
                          Grdc_ni, x0, xl, &
                          Grdc_nj, y0, yl,'LU', &
                          Grdc_dx ,Grdc_dy , &
                          Grdc_dx*40000./360.,Grdc_dy*40000./360.
         end if
      else
         if (prout) write (6,1004)
         return
      endif

      if (Grdc_trnm_S(1) == '@#$%') then
         do i=1,Tr3d_ntr
            Grdc_trnm_S(i) = Tr3d_name_S(i)
         end do
         Grdc_ntr = Tr3d_ntr
      else
         cnt = 0
         do 10 k=1,max_trnm
            if (Grdc_trnm_S(k) == '@#$%') goto 89
            flag_hu= (trim(Grdc_trnm_S(k)) == 'HU')
            do i=1,Tr3d_ntr
               if (trim(Grdc_trnm_S(k)) == trim(Tr3d_name_S(i))) then
                  cnt=cnt+1
                  Grdc_trnm_S(cnt) = Tr3d_name_S(i)
                  goto 10
               endif
            end do
 10      continue

 89      if (.not.flag_hu) then
            cnt=cnt+1
            Grdc_trnm_S(cnt) = 'HU'
         endif
         Grdc_ntr = cnt
      endif

      if (prout) then
          write (6,1001)
          write (6,'(5(x,a))') Grdc_trnm_S(1:Grdc_ntr)
          write (6,1010)
      endif

 1001 format ( ' Cascade grid: Tracers to be written for cascade run are: ')
 1002 format(/,' Cascade grid: Wrong configuration: ', &
               ' Grd_ni,Grd_nj,Grd_iref,Grd_jref:'/4I8/)
 1004 format (/' Cascade grid: insufficient, NO SELF CASCADE DATA will be produced')
 1005 format (/'################ SELF CASCADE DATA WILL BE PRODUCED ################'/&
               ' Cascade grid: Grdc_gid,Grdc_gif=',2I5, ';    Grdc_gjd,Grdc_gjf=',2I5/&
               ' Cascade grid: Grdc_ndt,Grdc_start,Grdc_end=',3i5)
 1006 format (/'CASCADE OUTPUT DIMENSIONS: ',&
        /1X,' NI=',I5,' FROM x0=',F11.5,' TO xl=',F11.5,' DEGREES' &
        /1X,' NJ=',I5,' FROM y0=',F11.5,' TO yl=',F11.5,' DEGREES'/)

 1007 format ( ' This is an acid test' )
 1010 format ( '####################################################################')
 1100 FORMAT (1X,'TARGET CASCADE GRID CONFIGURATION: UNIFORM RESOLUTION: ',a, &
        /1X,' NI=',I5,' FROM x0=',F11.5,' TO xl=',F11.5,' DEGREES' &
        /1X,' NJ=',I5,' FROM y0=',F11.5,' TO yl=',F11.5,' DEGREES' &
        /1X,' GRIDTYPE= ',a,'     DX= ',F11.5,'   DY= ',F11.5,' degrees' &
        /14X,               '     DX= ',F11.5,'   DY= ',F11.5 ' km'/)
!--------------------------------------------------------------------
      return
      end

