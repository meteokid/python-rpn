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

!**s/r wil_nml - Read williamson namelist

      integer function wil_nml (F_namelistf_S, F_unout, F_williamson_L)
      use wil_options
      implicit none
#include <arch_specific.hf>

      character(len=*) F_namelistf_S
      logical F_williamson_L
      integer F_unout

#include <rmnlib_basics.hf>
      integer err, unf, nrec
!
!-------------------------------------------------------------------
!
      wil_nml = -1 ; F_williamson_L = .false.

      if ((F_namelistf_S == 'print').or.(F_namelistf_S == 'PRINT')) then
         wil_nml = 0
         if (F_unout > 0) then
            write (F_unout, nml=williamson)
         endif
         return
      endif

      if (F_namelistf_S /= '') then
         unf = 0
         if (fnom (unf,F_namelistf_S, 'SEQ+OLD', nrec) /= 0) then
            if (F_unout >= 0) write (F_unout, 7050) trim( F_namelistf_S )
            goto 9999
         endif
         rewind(unf)
         read (unf, nml=williamson, end = 9000, err=9130)
         goto 9000
      endif

 9130 if (F_unout >= 0) write (F_unout, 7070) trim( F_namelistf_S )
      goto 9999

 9000 F_williamson_L = Williamson_case > 0

      wil_nml = 1

 7050 format (/,' FILE: ',A,' NOT AVAILABLE'/)
 7070 format (/,' NAMELIST &williamson IS INVALID IN FILE: ',a/)

 9999 err = fclos (unf)
!
!-------------------------------------------------------------------
!
      return
      end function wil_nml
