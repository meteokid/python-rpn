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

!**s/r out_mergeyy
!
      subroutine out_mergeyy (F_vector, k, nis, njs, nks, n)
      use iso_c_binding
      use ptopo
      implicit none
#include <arch_specific.hf>

      integer, intent(in) :: nis, njs, nks, n, k
      real, dimension(nis, njs, nks), intent(inout) :: F_vector
!
!author
!    Michel Desgagne - Fall 2012
!
!revision
! v4_50 - Desgagne M. - Initial version

      include "rpn_comm.inc"

      real, dimension(n, 2) :: F_vec
      integer :: tag, stat, err
!
!----------------------------------------------------------------------
!
      tag= 401

      F_vec = reshape(F_vector(:,:,k), [n ,2])

      if (Ptopo_couleur == 0) then
         call RPN_COMM_recv ( F_vec(1,2), n, 'MPI_REAL', 1, &
                              tag, 'GRIDPEERS', stat, err )
      else
         call RPN_COMM_send ( F_vec     , n, 'MPI_REAL', 0, &
                              tag, 'GRIDPEERS',         err )
      endif

      F_vector(:,:,k) = reshape(F_vec, [nis, njs])
!
!----------------------------------------------------------------------
!
      return
      end


