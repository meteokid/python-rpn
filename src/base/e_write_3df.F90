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
!**s/r e_write_3df
!
      subroutine e_write_3df ( tr1,nis,njs,nks, nomvar,unf )
      implicit none
#include <arch_specific.hf>
!
      character* (*) nomvar
      integer nis,njs,nks, unf
      real tr1(nis,njs,nks)

!author  M. Desgagne 2001 (MC2)
!
!revision
! v3_30 - Lee. V - modified to write data extracted from analysis input
!                  into 3DF files
!
!*
!
      integer i,j,k,n,nbits,nb
      real, dimension (:), allocatable :: wkc
      logical prout_L
!
!----------------------------------------------------------------------
!
      nb = 0
      nbits = 32

      write (unf) nomvar(1:4),nis,njs,nks,nbits
      if (nbits.ge.32) then
         do k=1,nks
            write (unf) tr1(:,:,k)
         end do
      else
          n = (nis*njs*nbits+120+32-1)/32
          allocate (wkc(n))
          do k=1,nks
             call xxpak (tr1(1,1,k), wkc, nis, njs, -nbits, nb, 1)
             write (unf) wkc
          end do
          deallocate (wkc)
      endif
!
!----------------------------------------------------------------------
!
      return
      end
