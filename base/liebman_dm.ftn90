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

!**s/r liebman_dm - modifiy a portion of input field by overrelaxation with del**2=0.
!

      subroutine liebman_dm (F_field,  F_mask, F_conv, F_maxite, &
                                     Minx, Maxx, Miny, Maxy, Nk)
      implicit none
#include <arch_specific.hf>
!
      integer F_maxite, Minx, Maxx, Miny, Maxy, NK
      real    F_conv
      real    F_field (Minx:Maxx,Miny:Maxy,NK), &
              F_mask  (Minx:Maxx,Miny:Maxy,NK)
!
!author
!     Michel Desgagne - after version v1_03 of liebman.ftn
!
!revision
! v4_40 - Desgagne M.       - initial MPI version
!
#include "glb_ld.cdk"
#include "ptopo.cdk"

      integer i,j,k,ite,i0,in,ic,j0,jn,jc,ier,count
      real    prfact,prmax(Nk),prmaxall(Nk),prmod,mask(2,Nk),maskall(2,Nk)
      real*8  sum0(2,Nk),sumall(2,Nk)
!     __________________________________________________________________
!
      call rpn_comm_xch_halo(F_field, Minx,Maxx,Miny,Maxy,l_ni,l_nj,Nk, &
                             G_halox,G_haloy,G_periodx,G_periody,l_ni,0)

      prfact   = 1.75 * 0.25
      prmaxall = 1000.

!$omp parallel shared (i0,in,ic,j0,jn,jc,prmax,prmaxall,prfact, &
!$omp                  count,sum0,sumall,mask,maskall) private (i,j,k,ite,prmod)

      if (.not.G_lam) then
!$omp do
         do k=1,Nk
            sum0 (:,k) = 0.0
            mask(:,k) = 0.0
         end do
!$omp enddo
         if (Ptopo_myrow.eq.0) then
!$omp do
            do k=1,Nk
            do i=1,L_ni
               sum0 (1,k) = sum0(1,k) + F_field(i,1,k)
               mask(1,k) = max( mask(1,k), F_mask(i,1,k) )
            end do
            end do
!$omp enddo
         endif
         if (Ptopo_myrow.eq.(Ptopo_npey-1)) then
!$omp do
            do k=1,Nk
            do i=1,L_ni
               sum0 (2,k) = sum0(2,k) + F_field(i,l_nj,k)
               mask(2,k) = max( mask(2,k), F_mask(i,l_nj,k) )
            end do
            end do
!$omp enddo
         endif
!$omp single
         call rpn_comm_ALLREDUCE (sum0,sumall  ,2*Nk,"MPI_double_precision","MPI_SUM","grid",ier)
         call rpn_comm_ALLREDUCE (mask,maskall,2*nk,"MPI_REAL"            ,"MPI_MAX","grid",ier)
!$omp end single

         if (Ptopo_myrow.eq.0             ) then
!$omp do
            do k=1,Nk
            do i=1,L_ni
               F_field(i,0,k) = sumall(1,k) / G_ni
            end do
            end do
!$omp enddo
         endif
         if (Ptopo_myrow.eq.(Ptopo_npey-1)) then
!$omp do
            do k=1,Nk
            do i=1,L_ni
               F_field(i,l_nj+1,k) = sumall(2,k) / G_ni
            end do
            end do
!$omp enddo
         endif
      endif

!$omp single         
      i0 = 1 ; in = l_ni ; ic = 1
      j0 = 1 ; jn = l_nj ; jc = 1
!$omp end single

      do ite=1,F_maxite
!$omp do
         do k=1,Nk
            prmax(k) = 0.0
         end do
!$omp enddo

         if (.not.G_lam) then
            if (Ptopo_myrow.eq.0) then
!$omp do
               do k=1,Nk
                  if ( prmaxall(k) .gt. F_conv ) then
                     prmod = prfact * ( sumall(1,k) - G_ni * F_field(1,     0,k) ) * maskall(1,k)
                     prmod = prmod * 4.0 / G_ni    
                     prmax(k) = max ( prmax(k), abs(prmod) )
                     F_field (:,     0,k) = F_field (:,     0,k) + prmod
                  endif
               end do
!$omp enddo
            endif
            if (Ptopo_myrow.eq.(Ptopo_npey-1)) then
!$omp do
               do k=1,Nk
                  if ( prmaxall(k) .gt. F_conv ) then
                     prmod = prfact * ( sumall(2,k) - G_ni * F_field(1,l_nj+1,k) ) * maskall(2,k)
                     prmod = prmod * 4.0 / G_ni    
                     prmax(k) = max ( prmax(k), abs(prmod) )
                     F_field (:,l_nj+1,k) = F_field (:,l_nj+1,k) + prmod
                  endif
               end do
!$omp enddo
            endif
         endif
         
!$omp do
         do k=1,Nk
            if ( prmaxall(k) .gt. F_conv ) then
               do j=j0,jn,jc
               do i=i0,in,ic
                  prmod = prfact * F_mask(i,j,k)               * &
                          (F_field(i-1,j,k) + F_field(i+1,j,k) + &
                           F_field(i,j-1,k) + F_field(i,j+1,k) - &
                           4.*F_field(i,j,k))
                  prmax(k) = max ( prmax(k), abs(prmod) )
                  F_field(i,j,k) = F_field(i,j,k) + prmod
               enddo
               enddo
            endif
         enddo
!$omp enddo

!$omp single
         if ((i0.eq.1   ).and.(j0.eq.1   )) then
            i0=l_ni ; in=1 ; ic=-1
         endif
         if ((i0.eq.l_ni).and.(j0.eq.1   )) then
            j0=l_nj ; jn=1 ; jc=-1
         endif
         if ((i0.eq.l_ni).and.(j0.eq.l_nj)) then
            i0 = 1    ; in = l_ni ; ic =  1
            j0 = l_nj ; jn = 1    ; jc = -1
         endif
         if ((i0.eq.1   ).and.(j0.eq.l_nj)) then
            i0 = 1 ; in = l_ni ; ic = 1
            j0 = 1 ; jn = l_nj ; jc = 1
         endif

         call rpn_comm_ALLREDUCE (prmax,prmaxall,Nk,"MPI_REAL","MPI_MAX","grid",ier)

         count = 0
         do k=1,Nk
            if ( prmaxall(k) .lt. F_conv ) count = count + 1
         end do
!$omp end single

         if ( count.eq.Nk ) exit

!$omp single
         if (mod(ite,4).eq.0) &
         call rpn_comm_xch_halo( F_field, Minx,Maxx,Miny,Maxy,l_ni,l_nj,Nk, &
                                 G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
!$omp end single
         if (.not.G_lam) then
!$omp do
            do k=1,Nk
               sum0(:,k) = 0.0
            end do
!$omp enddo
            if (Ptopo_myrow.eq.0) then
!$omp do
               do k=1,Nk
                  if ( prmaxall(k) .gt. F_conv ) then
                     do i=1,L_ni
                        sum0 (1,k) = sum0(1,k) + F_field(i,   1,k)
                     end do
                  endif
               end do
!$omp enddo
            endif
            if (Ptopo_myrow.eq.(Ptopo_npey-1)) then
!$omp do
               do k=1,Nk
                  if ( prmaxall(k) .gt. F_conv ) then
                     do i=1,L_ni
                        sum0 (2,k) = sum0(2,k) + F_field(i,l_nj,k)
                     end do
                  endif
               end do
!$omp enddo
            endif
!$omp single
            call rpn_comm_ALLREDUCE (sum0,sumall,2*nk,"MPI_double_precision","MPI_SUM","grid",ier)
!$omp end single
         endif

      end do
!$omp end parallel
!     __________________________________________________________________
!
      return
      end
