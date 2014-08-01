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

!**s/r bcs_ftype - to determine whether pilot file is BMF,BCS or 3DF

      subroutine bcs_ftype2 ( ft, datev )
      implicit none
#include <arch_specific.hf>

      character* (*) ft, datev

!author
!        Michel Desgagne - 2001 (from MC2 bcs_ftype)
!revision
! v3_30 - Desgagne M.  - initial version for GEMDM
! v4_03 - Lee/Desgagne - ISST
!
!ARGUMENTS
!    NAMES     I/O  TYPE  A/S        DESCRIPTION
!     ft        O   C     S       Input file type

#include "ifd.cdk"
#include "path.cdk"
#include "ptopo.cdk"

      integer  nav_3df2
      external nav_3df2

      character*1024 fn
      logical done
      integer unf,err
      save done
      data done /.false./
!
!-----------------------------------------------------------------------
!
      if (.not.done) call blk_coverage

      err= 0

      if (Ptopo_blocme.eq.0) then

         unf = 76
         fn= trim(Path_ind_S)//'/3df_'//trim(datev)//'_filemap.txt'
         open (unf,file=fn,access='SEQUENTIAL',status='OLD', &
                                iostat=err,form='FORMATTED')
         if ( err.eq.0 ) then
            ft = '3DF'
         else
            ft = 'BMF'
            err= 0
         endif

         if (ft.eq.'3DF') then
            err= nav_3df2 (unf)
            close (unf)
         endif

         fn(1:len(ft))= ft

      endif

      call handle_error (err,'bcs_ftype','Problem with nav_3df')

      call RPN_COMM_bcastc (fn, len(ft), "MPI_CHARACTER",0,"BLOC",err)

      ft= fn(1:len(ft))
         
      done= .true.
!
!-----------------------------------------------------------------------
!
      return
      end

