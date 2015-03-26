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

!**s/r out_launchpost - Signals monitor for output processing
!

!
      subroutine out_launchpost2 (F_forceit_L,F_last_L)
      implicit none
#include <arch_specific.hf>
!
      logical F_forceit_L,F_last_L
!
!AUTHOR   Michel Desgagne  - Summer 2010
!
!REVISION
! v4_14 - Desgagne M.      - Initial version
!
#include "out.cdk"
#include "path.cdk"
#include "ptopo.cdk"
#include "lctl.cdk"
#include <clib_interface_mu.hf>
      include "rpn_comm.inc"
!
      logical slice_last_L
      common /tempory_common_slice/ slice_last_L

      character*1024 filen,filen_link,append
      logical pe0_master_L, release_dir_L
      integer i, rank, err, unf
!
!----------------------------------------------------------------------
!
      pe0_master_L = (Ptopo_myproc.eq.0) .and. (Ptopo_couleur.eq.0)
      release_dir_L= Out_post_L .or. F_forceit_L

      if (pe0_master_L) then
         
         append=''
         if (F_last_L .and. slice_last_L) append='^last'
         
         if (release_dir_L) then
            
            filen      = trim(Path_output_S)//'/output_ready_MASTER'
            filen_link = trim(Path_output_S)//'/output_ready'
            unf   = 474
            open  ( unf,file=filen,access='SEQUENTIAL',&
                    form='FORMATTED',position='APPEND' )
            write (unf,'(2(a))') trim(Out_laststep_S),trim(append)
            close (unf)
            
         endif
         
      endif
      
      if (release_dir_L.and.(.not.F_forceit_L)) then
         call rpn_comm_barrier (RPN_COMM_ALLGRIDS, err)
         call rpn_comm_rank (RPN_COMM_ALLGRIDS, rank , err)
         if (rank.eq.0) then
            err = clib_symlink ( trim(filen), trim(filen_link) )
            write (6,1001) trim(Out_laststep_S),lctl_step
         endif
      endif
      
1001  format (' OUT_LAUNCHPOST: DIRECTORY output/',a, &
              ' was released for postprocessing at timestep: ',i9)
!
!----------------------------------------------------------------------
!
      return
      end
