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

!**s/r e_fill_3df - to fill a K plane of a 3d variable with a 2d variable
!                   as well as doing a conversion on the field.
!
      subroutine e_fill_3df ( fa,tr1,nis,njs,nks,k,con,add)
      implicit none
#include <arch_specific.hf>
!
      integer nis,njs,nks,k
      real tr1(nis,njs,nks),con,add
      real fa(nis,njs)
!
!author V.Lee 2006
!
!revision
! v3_30 - Lee.V - initial version
!
!*
!
      integer i,j,n
!
!----------------------------------------------------------------------
!
!     fill 2D field from fa to 3D field tr1 binary
!
      do j=1,njs
      do i=1,nis
         tr1(i,j,k)=(fa(i,j)+add)*con
      enddo
      enddo
!
!----------------------------------------------------------------------
      return
      end
!
