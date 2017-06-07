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
      implicit none
#include <arch_specific.hf>

      logical prout
      integer dimgx,dimgy
      real*8 xp(dimgx), yp(dimgy)

!author
!        Michel Desgagne - 2001 (from MC2)
!revision
! v3_30 - Lee V.       - initial version for GEMDM
! v3_30 - McTaggart-Cowan R.- Allow for user-defined domain tag extensions
! v4_03 - Lee/Desgagne - ISST
!

#include "glb_ld.cdk"
#include "glb_pil.cdk"
#include "rstr.cdk"
#include "tr3d.cdk"
#include "lun.cdk"

      integer, external :: stretch_axis2
      logical flag_hu
      integer i,k,cnt,ierx,dum1,dum2
      integer is,nis,js,njs,iw,ie,niw,jw,jn,njw
      real x0, xl, y0, yl, dum, n1, n2, b1, b2
      real*8 ac_xp(max(1,Grdc_ni)), ac_yp(max(1,Grdc_nj)), &
             xpx(dimgx), ypx(dimgy), rad2deg_8,xgi_8(G_ni),ygi_8(G_ni)
!
!---------------------------------------------------------------------
!
      rad2deg_8 = 180.0d0/pi_8
      xpx = xp * rad2deg_8
      ypx = yp * rad2deg_8
!
      Grdc_gid = 0
      Grdc_gjd = 0
      Grdc_gif = 0
      Grdc_gjf = 0
!
      if ( (Grdc_ndt.lt.0) .or. (Grd_yinyang_S .eq. 'YAN') .or. &
           (Grdc_ni .eq.0) .or. (Grdc_nj.eq.0) .or. (Grdc_dx.lt.0.) ) then
         Grdc_ndt = -1
         return
      endif

      if (Grdc_dy.lt.0) Grdc_dy = Grdc_dx
!
!     *** Positional parameters for f and q points
!
      x0   = Grdc_lonr - (Grdc_iref-1) * Grdc_dx
      y0   = Grdc_latr - (Grdc_jref-1) * Grdc_dy
      xl   = x0 + (Grdc_ni  -1) * Grdc_dx
      yl   = y0 + (Grdc_nj  -1) * Grdc_dy
!
      if (x0.lt.0.0)x0=x0+360.0
      if (xl.lt.0.0)xl=xl+360.0
!
      ierx = stretch_axis2 ( ac_xp, Grdc_dx, x0, xl, dum1, Grdc_ni, &
                 Grdc_ni, dum, .false.,Lun_debug_L,360., dum2, .false.)
      ierx = stretch_axis2 ( ac_yp, Grdc_dy, y0, yl, dum1, Grdc_nj, &
                 Grdc_nj, dum, .false.,Lun_debug_L,180., dum2, .false.)
!
      Grdc_xp1 = ac_xp(1)
      Grdc_yp1 = ac_yp(1)
!
      do i=1,dimgx
         if (xpx(i).le.ac_xp(1)      ) Grdc_gid=i
         if (xpx(i).le.ac_xp(Grdc_ni)) Grdc_gif=i
      enddo

      do i=1,dimgy
         if (ypx(i).le.ac_yp(1)      ) Grdc_gjd=i
         if (ypx(i).le.ac_yp(Grdc_nj)) Grdc_gjf=i
      enddo

      ! Tests if same grid
      if(&
           Grdc_iref.eq.Grd_iref.and.&
           Grdc_jref.eq.Grd_jref.and.&
           abs(Grdc_latr-Grd_latr)<1.e-6.and.&
           abs(Grdc_lonr-Grd_lonr)<1.e-6.and.&
           abs(Grdc_dx-Grd_dx)/Grd_dx<1.e-6)then
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

      if ( (Grdc_gid.lt.1    ).or.(Grdc_gjd.lt.1    ).or. &
           (Grdc_gif.gt.dimgx).or.(Grdc_gjf.gt.dimgy) ) Grdc_ndt = -1

      if (Grdc_ndt.gt.0) then
         if ((prout).and.(.not.Rstri_rstn_L)) &
         write (6,1006) Grdc_gid,Grdc_gif,Grdc_gjd,Grdc_gjf,Grdc_ndt,Grdc_start,Grdc_end
         !#TODO: update since out_sgrid is no longuer avail.
!!$         call out_sgrid2 (Grdc_gid, Grdc_gif, Grdc_gjd, Grdc_gjf, &
!!$                                            0, 0, .false., 1, '')
      else
         if (prout) write (6,1004)
         return
      endif

      if (Grdc_trnm_S(1).eq.'@#$%') then
         do i=1,Tr3d_ntr
            Grdc_trnm_S(i) = Tr3d_name_S(i)
         end do
         Grdc_ntr = Tr3d_ntr
      else
         cnt = 0
         do 10 k=1,max_trnm
            if (Grdc_trnm_S(k).eq.'@#$%') goto 89
            flag_hu= (trim(Grdc_trnm_S(k)) == 'HU')
            do i=1,Tr3d_ntr
               if (trim(Grdc_trnm_S(k)).eq.trim(Tr3d_name_S(i))) then
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
 1004 format (/' Cascade grid: Is too large, NO SELF CASCADE DATA will be produced')
 1006 format (/'################ SELF CASCADE DATA WILL BE PRODUCED ################'/&
               ' Cascade grid: Grdc_gid,Grdc_gif=',2I5, ';    Grdc_gjd,Grdc_gjf=',2I5/&
               ' Cascade grid: Grdc_ndt,Grdc_start,Grdc_end=',3i5)
 1007 format ( ' This is an acid test' )
 1010 format ( '####################################################################')
!--------------------------------------------------------------------
      return
      end

