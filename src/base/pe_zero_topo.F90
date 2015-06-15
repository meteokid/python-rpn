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

!**s/r pe_zero_topo - Initialize processor topology
!

!
      subroutine pe_zero_topo (F_npx, F_npy )
      implicit none
#include <arch_specific.hf>
!
      integer F_npx, F_npy
!
!author
!     Michel Desgagne - Summer 2006
!
!revision
! v4_03 - Desgagne M.       - initial version, ISST
!

#include "component.cdk"
#include "ptopo.cdk"
#include "version.cdk"
#include "path.cdk"
#include <clib_interface_mu.hf>
      include 'gemdyn_version.inc'
!
      integer,external :: exdb,ptopo_nml

      character(len=50) :: DSTP,name_S,arch_S,compil_S
      logical :: is_official_L
      integer :: err
!
!-------------------------------------------------------------------
!
      err= clib_mkdir (Path_output_S)

      call  open_status_file3 (trim(Path_output_S)//'/status_'//trim(COMPONENT)//'.dot')
      call write_status_file3 ('_status=ABORT' )

      call atm_model_getversion(name_S,Version_number_S,DSTP,arch_S,compil_S,is_official_L)
      if (is_official_L) then
         Version_title_S = trim(name_S)//' --- Release of: '//trim(DSTP)
      else
         Version_title_S = trim(name_S)//' --- User Build: '//trim(DSTP)
      endif

      err = exdb(trim(Version_title_S),trim(Version_number_S),'NON')
!
! Read namelist ptopo from file model_settings
!
      if (ptopo_nml (trim(Path_work_S)//'/model_settings.nml') .eq. 1 ) then
         F_npx = Ptopo_npex
         F_npy = Ptopo_npey
         err = ptopo_nml ('print')
      else
         write (6, 8000)
         F_npx = 0
         F_npy = 0
      endif

      write (6,1001) trim(GEMDYN_NAME_S),trim(GEMDYN_VERSION_S), &
                     trim(GEMDYN_DSTP_S),trim(GEMDYN_EC_ARCH_S)

 1001 format (/3x,60('*')/3x,'Package: ',a,5x,'version: ',a/ &
               3x,'Release of: ',a,5x,'COMPILER: 'a/3x,60('*')/)
 8000 format (/,'========= ABORT IN S/R PE_TOPO ============='/)
!
!-------------------------------------------------------------------
!
      return
      end
