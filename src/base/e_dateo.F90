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

!**s/r e_dateo - obtain initial date from either the given analysis (GLB configs)
!                or from the namelist key Step_runstrt_S
!
      subroutine e_dateo
      implicit none
#include <arch_specific.hf>
!
!
!author Desgagne M. - MC2
!
!revision
! v4_03 - Lee/Desgagne - ISST

#include "e_grids.cdk"
#include "step.cdk"
#include "lun.cdk"
#include "pilot1.cdk"
#include "filename.cdk"
#include "grd.cdk"
#include "path.cdk"
!
      logical  get_date
      external get_date

      integer nsec1, nsec2, nsec3
      real*8  one,sid,rsid,dayfrac
      parameter(one=1.0d0, sid=86400.0d0, rsid=one/sid )
      character*16 ladate_S
! ---------------------------------------------------------------------
!
      if (Grd_typ_S(1:1).eq.'L') then
!
         call e_infiles
!
         if (Pil_jobstrt_S == 'NIL') Pil_jobstrt_S=Step_runstrt_S
         if (Pil_jobstrt_S .ne. 'NIL') then
            if (Pil_jobend_S == 'NIL') then
               if ((Step_total.ge.0).and.(Step_dt.gt.0.)) then
                  nsec1 = Step_total * Step_dt
                  nsec2 = mod(nsec1,Step_nesdt)
                  nsec1 = nsec1 + (Step_nesdt-nsec2)*min(1,nsec2)
                  dayfrac = dble(nsec1) * rsid
                  call incdatsd (Pil_jobend_S, Pil_jobstrt_S, dayfrac)
               else
                  Pil_jobend_S = Pil_jobstrt_S
               endif
            endif
         endif
         if ( Pil_jobstrt_S.eq.'NIL' .or.  &
              Pil_jobend_S .eq.'NIL' ) then
            if (Lun_out.gt.0) write (6,1005) 
            stop
         endif
         if ( (Pil_jobstrt_S.ne.Pil_jobend_S).and. &
              (Step_nesdt.le.0) ) then
            if (Lun_out.gt.0) write (6,1006) 
            stop
         endif  
!
      else
!
         if (.not. &
         get_date (Step_runstrt_S ,trim(Path_input_S)//'/ANALYSIS')) stop
         Pil_jobstrt_S = Step_runstrt_S
         Pil_jobend_S  = Step_runstrt_S
!
      endif
!
      call gemtim4 ( Lun_out, 'END OF e_date0', .false. )
!
 1001 format (/' Incorrect VALIDITY time in ',a,/' VALIDITY=',a,' INTENDED=',&
                 a,' -----ABORT-----'/)
 1005 format(/' In LAM configuration: Pil_jobstrt_S and Pil_jobend_S must both', &
             /' be specified when not specifying Step_runstrt_S - ABORT -'/)
 1006 format(/' Step_nesdt must ', &
             /' be specified in LAM configuration - ABORT -'/)
!
! ---------------------------------------------------------------------
!
      return
      end
!
      logical function get_date (datpdf,filename)
      implicit none
#include <arch_specific.hf>
!
#include "lun.cdk"
      character* (*) datpdf,filename
!
      integer  fnom, fstouv, fstinf, fstprm, fstfrm, fclos, wkoffit
      external fnom, fstouv, fstinf, fstprm, fstfrm, fclos, wkoffit
!
      character*1   typ, grd
      character*4   var
      character*12  lab
      character*16  datev_S
      integer key,ni1,nj1,nk1,datestp,unf, &
              det, ipas, p1, p2, p3, g1, g2, g3, g4, bit, &
              dty, swa, lng, dlf, ubc, ex1, ex2, ex3, kind, err
      real*8  one,sid,rsid,dayfrac
      parameter (one=1.0d0, sid=86400.0d0, rsid=one/sid)
!
! ---------------------------------------------------------------------
!
      get_date = .false.

      if (datpdf == 'NIL') then
         if (Lun_out.gt.0) write (6,1001) trim(filename)
         datestp = -1
      else
         if (Lun_out.gt.0) write (6,1002) datpdf,trim(filename)
         call datp2f ( datestp, datpdf )
      endif
!
      unf = 0
      if (wkoffit(filename) .gt. -1 ) then
         if (fnom  (unf,filename,'RND+OLD+R/O',0).lt.0) then
            if (Lun_out.gt.0) write (6,2001) trim(filename)
            stop
         endif
         if (fstouv(unf ,'RND').lt.0) then
            if (Lun_out.gt.0) write (6,2002) trim(filename)
            stop
         endif
         if (Lun_out.gt.0) write (6,2003) trim(filename),unf
!
         key = fstinf (unf, ni1,nj1,nk1,datestp,' ',-1,-1,-1,' ','UU' )
         if ( key .lt. 0 ) then
            key = fstinf (unf, ni1,nj1,nk1,datestp,' ',-1,-1,-1,' ','UT1')
            if ( key .lt. 0 ) then
               if (Lun_out.gt.0) write(6,1004) 
               stop
            endif
         endif
!
         if (datpdf == 'NIL') then
            err = fstprm ( key, datestp, det, ipas, ni1, nj1, nk1,bit,dty, &
                           p1,p2,p3, typ, var, lab, grd, g1,g2,g3,g4,    &
                           swa,lng, dlf, ubc, ex1, ex2, ex3 )
            call datf2p ( datpdf, datestp )
            dayfrac = dble(det*ipas)*rsid
            call incdatsd ( datev_S, datpdf, dayfrac )
            datpdf = datev_S
         endif
!
         err = fstfrm (unf)
         err = fclos  (unf)
!
         if (Lun_out.gt.0) write (6,1003) trim(datpdf)
         get_date = .true.
!
      endif
!
 1001 format (/' ESTABLISHING VALIDITY TIME OF U-component of the wind' &
              /' in file: ',a)
 1002 format (/' TRYING TO FIND U-component of the wind valid: ',a, /&
               ' in file: ',a)
 1003 format (/' INITIAL CONDITION VALIDITY TIME= ',a)
 1004 format (/' UNABLE TO DETERMINE INITIAL CONDITION VALIDITY TIME', &
              /' -----ABORT ----- in e_dateo'/)
 2001 format (/' ABORT in e_dateo:  Trying to open file ',a)
 2002 format (/' ABORT in e_dateo:  File ',a,' not RND')
 2003 format (/' E_DATEO:  file ',a,' open on unit: ',i7)
!
! ---------------------------------------------------------------------
!
      return
      end
