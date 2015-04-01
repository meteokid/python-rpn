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

!**s/r sid3df
!

!
      function sid3df2(xpaq,ypaq,xpau,ypav,unf,done, &
                               nia,nja,nka_m,nka_t,presstype) result(F_istat)
      implicit none
#include <arch_specific.hf>
!author
!     Michel Desgagne - rpn - MC2 2001
!
!revision
! v4_03 - Lee V.            - Adapt to using new pressure functions
!
      logical done
      integer unf,nia,nja,nka_m,nka_t,presstype
      real*8 xpaq(nia), ypaq(nja), xpau(nia), ypav(nja)

      integer :: F_istat
!
#include "ifd.cdk"
#include "bcsgrds.cdk"
#include "ptopo.cdk"
#include "lun.cdk"
#include "grd.cdk"

      real, parameter :: EPS = 1.E-6

      character*4 nomvar
      integer i,j,k,ni1,nj1,nk1,err
      real*8, dimension (:), allocatable :: xp1,yp1,xu1,yv1
      real :: xlon1,xlat1,xlon2,xlat2
!-----------------------------------------------------------------------
!
      F_istat = -1

      read (unf,end=33,err=33)  &
           nomvar,ni1,nj1,nka_m,nka_t,presstype
      read (unf,end=33,err=33) xlon1,xlat1,xlon2,xlat2

      if (abs(xlon1-Grd_xlon1) > EPS .or.  &
           abs(xlat1-Grd_xlat1) > EPS  .or.  &
           abs(xlon2-Grd_xlon2) > EPS  .or.  &
           abs(xlat2-Grd_xlat2) > EPS) then
         if (Lun_out>0) then
            write(Lun_out,'(A)') 'ERROR: Data in 3df file should be on grid with same rotation.'
            write(Lun_out,'(A,2F8.2,A,2F8.2,A)') 'Model: (',Grd_xlon1,Grd_xlat1,') (',Grd_xlon2,Grd_xlat2,')'
            write(Lun_out,'(A,2F8.2,A,2F8.2,A)') '3df  : (',xlon1,xlat1,') (',xlon2,xlat2,')'
         endif
         F_istat = -2
         return
      endif

      if (.not.done) then         

         allocate (xp1(ni1),yp1(nj1),xu1(ni1),yv1(nj1))
         read (unf,end=33,err=33) xp1,yp1,xu1,yv1

         do i=1,nia
            xpaq(i) = xp1(ifd_niad+i-1)
            xpau(i) = xu1(ifd_niad+i-1)
         end do
         do j=1,nja
            ypaq(j) = yp1(ifd_njad+j-1)
            ypav(j) = yv1(ifd_njad+j-1)
         end do
         deallocate (xp1,yp1,xu1,yv1)
!
         if (associated(ana_am_8)) deallocate(ana_am_8,stat=err)
         if (associated(ana_bm_8)) deallocate(ana_bm_8,stat=err)
         if (associated(ana_at_8)) deallocate(ana_at_8,stat=err)
         if (associated(ana_bt_8)) deallocate(ana_bt_8,stat=err)
!
         allocate(ana_am_8(nka_m), ana_bm_8(nka_m), &
                  ana_at_8(nka_t), ana_bt_8(nka_t) )
!
         read (unf,end=33,err=33)  &
               ana_am_8,ana_bm_8,ana_at_8,ana_bt_8
         if (Lun_debug_L) then
             write(Lun_out,*)'sid3df:'
             write(Lun_out,*)'ana_am_8=',ana_am_8
             write(Lun_out,*)'ana_bm_8=',ana_bm_8
             write(Lun_out,*)'ana_at_8=',ana_at_8
             write(Lun_out,*)'ana_bt_8=',ana_bt_8
         endif
!
      else
         read (unf,end=33,err=33)
         read (unf,end=33,err=33)
      endif

      F_istat = 0
!
!-----------------------------------------------------------------------
 33   return
      end
