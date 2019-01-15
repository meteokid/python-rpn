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
!
      subroutine timing_terminate2 ( myproc, msg )
      implicit none
#include <arch_specific.hf>
!
      character* (*) msg
      integer myproc
!
!author
!     M. Desgagne   -- Winter 2012 --
!
!revision
! v4_40 - Desgagne - initial version
!
      include "timing.cdk"

      character*16 name
      character*64 fmt,nspace
      logical flag(MAX_instrumented)
      integer i,j,k,elem,lvl,lvlel(0:100)

      if (Timing_S=='YES') call tmg_terminate ( myproc, msg )

      if (myproc.ne.0) return

      print *,'___________________________________________________________' 
      print *,'__________________TIMINGS ON PE #0_________________________'

      flag=.false.

      do i = 1,MAX_instrumented
         lvl= 0 ; elem= i
 55      if ( (trim(nam_subr_S(elem)).ne.'') .and. (.not.flag(elem)) ) then

            write (nspace,'(i3)') 5*lvl+1
            fmt='('//trim(nspace)//'x,a,1x,a,i3,a,3x,a,1pe13.6,2x,a,i8)'
            do k=1,len(name)
               name(k:k) = '.'
            end do
            name (len(name)-len(trim(nam_subr_S(elem)))+1:len(name))= &
            trim(nam_subr_S(elem))
            write (6,trim(fmt)) name,'(',elem,')','Wall clock= ',&
                                sum_tb(elem),'count= ',timer_cnt(elem)
            flag(elem) = .true. ; lvlel(lvl) = elem
 65         do j = 1,MAX_instrumented
               if ((timer_level(j) .eq. elem) .and. (.not.flag(j)) )then
                  lvl= lvl+1
                  elem= j
                  goto 55
               endif
            end do
            lvl= lvl - 1
            if (lvl .ge. 0) then
               elem= lvlel(lvl)
               goto 65
            endif
         endif
      enddo

      print *,'___________________________________________________________' 

      return
      end
