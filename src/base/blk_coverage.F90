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

      subroutine blk_coverage
      implicit none
#include <arch_specific.hf>
!
#include "glb_ld.cdk"
#include "ptopo.cdk"
#include "blk_input.cdk"
!
      integer iproc, tag, err, status
      real*8  pos(4)
      data tag /210/
!
!----------------------------------------------------------------------
!
      if (Ptopo_blocme.eq.0) then

         if (.not.associated(blk_xg_8)) &
         allocate (blk_xg_8(0:Ptopo_numpe_perb-1,2))
         if (.not.associated(blk_yg_8)) &
         allocate (blk_yg_8(0:Ptopo_numpe_perb-1,2))
         if (.not.associated(blk_indx)) &
         allocate (blk_indx(0:Ptopo_numpe_perb-1,5))

         blk_xg_8(0,1) = G_xg_8(l_i0)
         blk_xg_8(0,2) = G_xg_8(l_i0 + l_ni - 1)
         blk_yg_8(0,1) = G_yg_8(l_j0)
         blk_yg_8(0,2) = G_yg_8(l_j0 + l_nj - 1)
!
! Receive local data (LD) segments from other processors of bloc
!
         do iproc = 1, Ptopo_numpe_perb-1
!
            call RPN_COMM_recv ( pos, 4, 'MPI_DOUBLE_PRECISION', iproc, &
                                               tag, 'BLOC', status, err )
            blk_xg_8(iproc,1) = pos(1)
            blk_xg_8(iproc,2) = pos(2)
            blk_yg_8(iproc,1) = pos(3)
            blk_yg_8(iproc,2) = pos(4)

         end do
!
      else
!
! Send local data (LD) segment to processor 0 of mybloc
!
         pos(1) = G_xg_8(l_i0)
         pos(2) = G_xg_8(l_i0 + l_ni - 1)
         pos(3) = G_yg_8(l_j0)
         pos(4) = G_yg_8(l_j0 + l_nj - 1)
         call RPN_COMM_send ( pos, 4, 'MPI_DOUBLE_PRECISION', 0, &
                                                tag, 'BLOC', err )
!
      endif
!
!----------------------------------------------------------------------
!
      return
      end
!
