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

!**s/r e_infiles - to determine list of inputfiles for LAM
!
      subroutine e_infiles ()
      implicit none
#include <arch_specific.hf>
!
!author   Michel Desgagne (MC2) 2006
!
!revision
! v3_30 - Desgagne M.      - initial version
! v4_03 - Lee/Desgagne - ISST
!
#include "filename.cdk"
#include "path.cdk"
#include "lun.cdk"
!
      integer  fnom,longueur,wkoffit
      external fnom,longueur,wkoffit
      integer cnt,i,err,unf
      character*500 fn
!
!----------------------------------------------------------------------
!
      npilf = 0
      cnt   = 0
!
      unf = 0
      if (fnom  (unf ,trim(Path_work_S)//'/liste_inputfiles_for_LAM', &
                                                  'SEQ',0).lt.0) stop
!
 77   cnt=cnt+1
      read (unf, '(a)', end = 9120) pilot_f(cnt)
      goto 77
 9120 npilf = cnt - 1
      close(unf)
!
      pilot_dir = trim(Path_input_S)//'/INREP'
      do cnt = 1, npilf
         err = -1
         fn = pilot_dir(1:longueur(pilot_dir))//'/'// &
              pilot_f(cnt)(1:longueur(pilot_f(cnt)))
         err = wkoffit(fn)
         if ((err.ne.1).and.(err.ne.33)) then
            if (Lun_out.gt.0) &
                 write(6,905) pilot_f(cnt)(1:longueur(pilot_f(cnt))), &
                              pilot_dir(1:longueur(pilot_dir))
            pilot_f (cnt) = '@#$%^&'
         endif
      end do
!
      i=0
      do cnt = 1, npilf
         if (pilot_f(cnt).ne.'@#$%^&') then
            i = i+1
            pilot_f(i) = pilot_f(cnt)
         endif
      end do
      npilf = i
!
      ipilf  =  1
!
      if (Lun_out.gt.0) write (6,900) pilot_dir(1:longueur(pilot_dir))
      do cnt=1,npilf
         if (Lun_out.gt.0) write(6,901) pilot_f(cnt)(1:longueur(pilot_f(cnt)))
      end do
!
 900  format (/' Available files in directory: ',a)
 901  format (4x,a)
 905  format (' FILE ',a,' FROM DIRECTORY ',a, &
              ' UNAVAILABLE OR NOT RPN-STD FORMAT - WILL BE IGNORED')
!---------------------------------------------------------------------
      return
      end
