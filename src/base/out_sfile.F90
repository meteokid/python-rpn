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

!**s/r Out_sfile - to open new output file

      subroutine out_sfile3 (F_stepno)
      use timestr_mod, only: timestr_prognum
      implicit none
#include <arch_specific.hf>

      integer,intent(in) ::  F_stepno

!AUTHOR   Michel Desgagne     September   2003 (MC2)
!
!REVISION
! v3_20 - Lee V.            - Adapted for GEMDM
! v3_30 - Dugas B.          - 1) Climate mode dm,dp and pm files are sent to directory
!                             ../../output/current_last_step/Out_myblocx-Out_myblocy
!                             2) Do not use Out_endstepno in climate mode
! v3_30 - McTaggart-Cowan R.- Use existing Out_etik_S string instead of namelist value
! v3_31 - Winger K.         - correction to size in ypq(Out_nisg) to Out_njsg
! v4_03 - Lee V.            - ISST + modification of Out_etik_S in out_sgrid only

#include <rmnlib_basics.hf>
#include "cstv.cdk"
#include "out.cdk"
#include "out3.cdk"
#include "path.cdk"
#include "step.cdk"
#include "ptopo.cdk"

      real,   parameter :: eps=1.e-12
      real*8, parameter :: OV_day = 1.0d0/86400.0d0

      character*1024 out_filename_s,myformat_S,my_hour
      character*16   datev,fdate
      character*15   startindx
      character*6    my_block
      character*4    unit_ext
      character*2    my_prefix
      logical flag
      integer prognum,err,i,indx,len0,len1
      real*8 dayfrac
!
!----------------------------------------------------------------------
!
!      err = fstopc('MSGLVL','INFORM',.false.)
      err = fstopc('MSGLVL','SYSTEM',.false.)

      Out_npas = F_stepno

      flag= (Out_blocme.eq.0) .and. (Ptopo_couleur.eq.0)

      if (flag) then

         Out_ip2 = int (dble(F_stepno) * Out_deet / 3600. + eps)

         Out_dateo = Out3_date
         call datf2p(fdate,Out_dateo)
         if (F_stepno.lt.0) then
            dayfrac = dble(F_stepno-Step_delay) * Cstv_dt_8 * OV_day
            call incdatsd (datev,Step_runstrt_S,dayfrac)
            call datp2f   (Out_dateo,datev)
         endif

         err = timestr_prognum(prognum,Out3_unit_S,Out3_close_interval,Out_dateo,float(Out_deet),F_stepno,Out_endstepno)
         unit_ext = ' '
         if (Out3_unit_S(1:3) == 'SEC') unit_ext = 's'
         if (Out3_unit_S(1:3) == 'MIN') unit_ext = 'm'
         if (Out3_unit_S(1:3) == 'DAY') unit_ext = 'd'
         if (Out3_unit_S(1:3) == 'STE') unit_ext = 'p'
         if (Out3_unit_S(1:3) == 'MON') unit_ext = 'n'
         
         call up2low ( Out_prefix_S,my_prefix)
         write(my_block,'(a,i2.2,a,i2.2)') '-',Out_myblocx,'-',Out_myblocy

         len1 = max(3,Out3_ndigits)
         if (any(Out3_unit_s(1:3) == (/'SEC','STE'/))) len1 = max(6,len1)
         len0 = len1
         if (prognum < 0) len0 = len0+1
         write(myformat_S,'(a,i1.1,a,i1.1,a)') '(a,i',len0,'.',len1,')'
         my_hour = ' '
         write(my_hour,trim(myformat_S)) '_',prognum
         
!        Out_filename_S= ppYYYYMMDDhh[-XX-YY]_ddd[U]
         Out_filename_S= trim(my_prefix)//fdate(1:8)//fdate(10:11)// &
                         my_block//trim(my_hour)//trim(unit_ext)

         Out_filename_S = trim(Out_dirname_S)//'/'//trim(Out_filename_S)

         if (Out_unf.eq.0) then
            err = fnom  (Out_unf, trim(Out_filename_S),'STD+RND',0)
            err = fstouv(Out_unf ,'RND')
         endif

      endif

 101  format (' FST FILE UNIT=',i3,' FILE = ',a,' IS OPENED')
!----------------------------------------------------------------------
      return
      end
!
      subroutine out_cfile2
      implicit none
#include <arch_specific.hf>
!
#include "out.cdk"
!
      integer, external :: fstfrm
      integer err
      real dummy
!
!----------------------------------------------------------------------
!
      call out_fstecr ( dummy,dummy,dummy,dummy,dummy,dummy,&
                        dummy,dummy,dummy,dummy,dummy,dummy,&
                        dummy,dummy, .true. )

      if ((Out_blocme.eq.0).and.(Out_unf.gt.0)) then
         err = fstfrm(Out_unf)
         call fclos(Out_unf)
         Out_unf = 0
      endif

 102  format (' FST FILE UNIT=',i3,' FILE = ',a,' IS CLOSED')
!----------------------------------------------------------------------
      return
      end
