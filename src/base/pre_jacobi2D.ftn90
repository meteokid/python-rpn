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

!**s/r  pre_jacobi2D -  Jacobi additive-Schwarz preconditioning
!
      subroutine pre_jacobi2D ( Sol,Rhs,evec_local,Ni,Nj,Nk,ai,bi,ci )
      implicit none
#include <arch_specific.hf>
!
      integer Ni,Nj,Nk
      real*8 Rhs(Ni,Nj,Nk),Sol(Ni,Nj,Nk)
      real*8  ai(Ni,Nj,Nk), bi(Ni,Nj,Nk), ci(Ni,Nj,Nk)
      real*8 evec_local(Ni,Ni) 
!
!author
!       Abdessamad Qaddouri - December  2006
!
!revision
! v3_30 - Qaddouri A.       - initial version
!
      integer i,j,k,jr
      real*8 fdg(Ni,Nj,Nk)
!
!     ---------------------------------------------------------------
!
!$omp parallel private(j,jr,i)
!$omp do
      do k=1,Nk
         call dgemm('T','N',Ni,Nj,Ni,1.0d0,evec_local,Ni, &
                       Rhs(1,1,k),Ni,0.0d0,Fdg(1,1,k),Ni)

         Do j =2, Nj
            jr =  j - 1
            Do i=1,Ni
               Fdg(i,j,k) = Fdg(i,j,k) - ai(i,j,k)*Fdg(i,jr,k)
            Enddo
         Enddo
         j = Nj
         Do i=1,Ni
            Fdg(i,j,k) = Fdg(i,j,k)/bi(i,j,k)
         Enddo
         Do j = Nj-1, 1, -1
            jr =  j + 1
            Do i=1 , Ni
            Fdg(i,j,k)=(Fdg(i,j,k)-ci(i,j,k)*Fdg(i,jr,k))/bi(i,j,k)
            Enddo
         Enddo

         call dgemm('N','N',Ni,Nj,Ni,1.0d0,evec_local,Ni, &
                        Fdg(1,1,k),Ni,0.d0,Sol(1,1,k),Ni)
      Enddo
!$omp enddo
!$omp end parallel
!
!     ---------------------------------------------------------------
!
      return
      end

