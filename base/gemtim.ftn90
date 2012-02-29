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

!**s/r gemtim4 - Timing routine for AIX architecture
!
      subroutine gemtim4 ( unf, from, last )
      implicit none
#include <arch_specific.hf>
!
      character*(*) from
      logical last
      integer unf
!author
!     B. Dugas
!
!revision
! v4_40 - Desgagne M.   - Add wall clock and accumulators
!
      logical, save :: timini_L
      data timini_L / .false. /
!
#if defined (AIX)
!#include "cstv.cdk"
      integer get_max_rss
      external get_max_rss
      real    spJour,Jour
      integer Used0,Used,SoftLim,HardLim,ppjour
      integer maxJour,Hold_Rsti,ierr
      save    Used0,ppJour,Jour
!
      character date_S*8 ,time_S*10
      character jour_S*11,heure_S*10
      real          users,systs
      real, save :: user0,syst0
      DOUBLE PRECISION, save ::  START, END, ACCUM_w,ACCUM_u,ACCUM_s
      DOUBLE PRECISION omp_get_wtime
      data user0,syst0,START / 0.0, 0.0, -1.d0 /
      data ACCUM_w,ACCUM_u,ACCUM_s / 0.d0, 0.d0, 0.d0 /
!
!----------------------------------------------------------------
!
      if (unf.le.0) return

      if (START.lt.0.d0) START = omp_get_wtime()

      call date_and_time( date_S,time_S )
      jour_S  = date_S(1:4)//'/'//date_S(5:6)//'/'//date_S(7:8)//' '
      heure_S = time_S(1:2)//'h'//time_S(3:4)//'m'//time_S(5:6)//'s,'

      call setrteopts('cpu_time_type=total_usertime')
      call cpu_time( users )

      call setrteopts('cpu_time_type=total_systime')
      call cpu_time( systs )

      END = omp_get_wtime()

      if (timini_L) then
         write(unf,1000) 'TIME: '//jour_S//heure_S, END-START, users-user0, &
                          systs-syst0,get_max_rss(),from
         ACCUM_w= ACCUM_w + END-START
         ACCUM_u= ACCUM_u + users-user0
         ACCUM_s= ACCUM_s + systs-syst0
         if (last) write(unf,1001) ACCUM_w,ACCUM_u,ACCUM_s
      endif

      user0 = users
      syst0 = systs
      START = END
#endif
      timini_L = .true.
!
!----------------------------------------------------------------
!
 1000    format(/A,' W: ',1pe12.6, &
                   ' U: ',1pe12.6, &
                   ' S: ',1pe12.6, &
                  ', Mem: ',i7,' (Kbytes/PE) ',a)
 1001    format(/'ACCUMULATED TIME: W: ',1pe12.6, &
                   ' U: ',1pe12.6, &
                   ' S: ',1pe12.6)

      return
      end
