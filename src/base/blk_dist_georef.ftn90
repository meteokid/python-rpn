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

      subroutine blk_dist_georef (F_xpaq    ,F_ypaq    ,F_xpau    ,F_ypav, &
                              F_ana_am_8,F_ana_bm_8,F_ana_at_8,F_ana_bt_8, &
                              F_nia, F_nja, F_nka_t, F_nka_m, F_diag_lvl , &
                              F_ntra, F_vtype, F_datev_S, F_unf, F_err)
      implicit none
#include <arch_specific.hf>
      character* (*) F_datev_S
      integer F_nia,F_nja,F_nka_t,F_nka_m,F_diag_lvl,F_ntra,F_vtype,F_unf,F_err
      real*8, dimension (:), pointer :: F_xpaq,F_ypaq,F_xpau,F_ypav, &
                         F_ana_am_8,F_ana_bm_8,F_ana_at_8,F_ana_bt_8
!
#include "glb_ld.cdk"
#include "grd.cdk"
#include "lun.cdk"
#include "ptopo.cdk"
#include "ifd.cdk"
#include "path.cdk"
#include "dcst.cdk"
#include "blk_input.cdk"
!
      character*4    nomvar
      character*8    dynophy
      character*1024 fn
      integer i,j,ni1,nj1,ideb,ifin,jdeb,jfin
      real    xlon1,xlat1,xlon2,xlat2
      real, parameter :: EPS = 1.E-6

      integer nia,nja,nka_t,nka_m,ntra,vtyp,diag_lvl
      common /blkcom_i/ nia,nja,nka_t,nka_m,ntra,vtyp,diag_lvl

      integer n, iproc, tag, err, status, dim, extra
      real*8  xri,xrf,yri,yrf,deg2rad_8,resax,resay,rapres
      real*8, dimension (:), pointer :: xp1,yp1,xu1,yv1,buf
      data tag /210/
!
!----------------------------------------------------------------------
!
      deg2rad_8 = Dcst_pi_8/180.d0
      F_err  = -1 ; F_nia  = -1 ; F_nja  = -1
      F_nka_t= -1 ; F_nka_m= -1 ; F_ntra = -1 ; F_vtype= -1

      if (associated(F_ana_am_8)) deallocate(F_ana_am_8)
      if (associated(F_ana_bm_8)) deallocate(F_ana_bm_8)
      if (associated(F_ana_at_8)) deallocate(F_ana_at_8)
      if (associated(F_ana_bt_8)) deallocate(F_ana_bt_8)
      if (associated(F_xpaq    )) deallocate(F_xpaq)
      if (associated(F_ypaq    )) deallocate(F_ypaq)
      if (associated(F_xpau    )) deallocate(F_xpau)
      if (associated(F_ypav    )) deallocate(F_ypav)

      if (Ptopo_blocme.eq.0) then

         do n=1,ifd_nf
            if (ifd_needit(n)) then
               fn = trim(Path_ind_S)//'/3df'//'_'//trim(F_datev_S)//'_'//ifd_fnext(n)
               exit
            endif
         end do

         open (F_unf,file=trim(fn),access='SEQUENTIAL', &
                      form='UNFORMATTED',status='OLD',iostat=err)

         if (err.ne.0) then
            write (6,998) trim(fn)
            goto 33
         endif

         read (F_unf,end=33,err=33) ni1,nj1,F_nka_t,F_nka_m,F_vtype,F_ntra,F_diag_lvl
         read (F_unf,end=33,err=33) xlon1,xlat1,xlon2,xlat2

         allocate (xp1(ni1),yp1(nj1),xu1(ni1),yv1(nj1))

         read (F_unf,end=33,err=33) xp1,yp1,xu1,yv1

         allocate( F_ana_am_8(F_nka_m+1), F_ana_bm_8(F_nka_m+1),&
                   F_ana_at_8(F_nka_t+1), F_ana_bt_8(F_nka_t+1) )

         read (F_unf,end=33,err=33) F_ana_am_8,F_ana_bm_8,F_ana_at_8,F_ana_bt_8

         close (F_unf)

         if (abs(xlon1-Grd_xlon1) > EPS  .or.  &
             abs(xlat1-Grd_xlat1) > EPS  .or.  &
             abs(xlon2-Grd_xlon2) > EPS  .or.  &
             abs(xlat2-Grd_xlat2) > EPS) then
            if (Lun_out>0) then
            write(Lun_out,'(A)') 'ERROR: Data in 3df file should be on grid with same rotation.'
            write(Lun_out,'(A,2F8.2,A,2F8.2,A)') 'Model: (',Grd_xlon1,Grd_xlat1,') (',Grd_xlon2,Grd_xlat2,')'
            write(Lun_out,'(A,2F8.2,A,2F8.2,A)') '3df  : (',xlon1,xlat1,') (',xlon2,xlat2,')'
            endif
            goto 33
         endif

         F_err = 0

 33      if (F_err .eq. 0) then

            resax = Grd_dx*deg2rad_8/1000.
            resay = Grd_dy*deg2rad_8/1000.
            rapres= max( 1.0d0, Grd_dx*deg2rad_8/(xp1(2)-xp1(1)) )
            extra = max( 4, nint(rapres*2) )

            do iproc = 0, Ptopo_numpe_perb-1
               xri = blk_xg_8(iproc,1) + resax
               yri = blk_yg_8(iproc,1) + resay
               xrf = blk_xg_8(iproc,2) - resax
               yrf = blk_yg_8(iproc,2) - resax
               
               ideb = 2*ni1
               ifin = -1
               do i = 1, ni1
                  if (xp1(i).ge.xri) ideb= min(ideb,i)
                  if (xp1(i).le.xrf) ifin= max(ifin,i)
               end do
               blk_indx(iproc,1) = max(1  ,ideb-extra)
               blk_indx(iproc,2) = min(ni1,ifin+extra)
               
               jdeb = 2*nj1
               jfin = -1
               do i = 1, nj1
                  if (yp1(i).ge.yri) jdeb= min(jdeb,i)
                  if (yp1(i).le.yrf) jfin= max(jfin,i)
               end do
               blk_indx(iproc,3) = max(1  ,jdeb-extra)
               blk_indx(iproc,4) = min(nj1,jfin+extra)
            end do

            F_nia  =  blk_indx(0,2) - blk_indx(0,1) + 1
            F_nja  =  blk_indx(0,4) - blk_indx(0,3) + 1
            blk_indx(0,5) = F_nia * F_nja

            if ((F_nia.gt.0).and.(F_nja.gt.0)) then

               allocate (F_xpaq(F_nia), F_ypaq(F_nja),&
                         F_xpau(F_nia), F_ypav(F_nja) )

               do i=1,F_nia
                  F_xpaq(i) = xp1(blk_indx(0,1)+i-1)
                  F_xpau(i) = xu1(blk_indx(0,1)+i-1)
               end do
               do j=1,F_nja
                  F_ypaq(j) = yp1(blk_indx(0,3)+j-1)
                  F_ypav(j) = yv1(blk_indx(0,3)+j-1)
               end do

            endif

         endif

         do iproc = 1, Ptopo_numpe_perb-1
            if (F_err .eq. 0) then
               nia= blk_indx(iproc,2) - blk_indx(iproc,1) + 1
               nja= blk_indx(iproc,4) - blk_indx(iproc,3) + 1
               blk_indx(iproc,5) = nia * nja
            else
               nia= -1
               nja= -1
            endif
            nka_t   = F_nka_t
            nka_m   = F_nka_m
            ntra    = F_ntra
            vtyp    = F_vtype
            diag_lvl= F_diag_lvl
            call RPN_COMM_send (nia, 7, 'MPI_INTEGER', iproc, &
                                            tag, 'BLOC', err )
            if (F_err .eq. 0) then
               dim= 2*nia+2*nja
               allocate (buf(dim))
               buf(          1:  nia    ) = xp1 (blk_indx(iproc,1):blk_indx(iproc,2))
               buf(  nia    +1:2*nia    ) = xu1 (blk_indx(iproc,1):blk_indx(iproc,2))
               buf(2*nia    +1:2*nia+nja) = yp1 (blk_indx(iproc,3):blk_indx(iproc,4))
               buf(2*nia+nja+1:dim      ) = yv1 (blk_indx(iproc,3):blk_indx(iproc,4))
               call RPN_COMM_send (buf, dim, 'MPI_DOUBLE_PRECISION', iproc, &
                                            tag, 'BLOC', err )
               deallocate (buf)
            endif
         end do

         if (associated(xp1)) deallocate(xp1)
         if (associated(yp1)) deallocate(yp1)
         if (associated(xu1)) deallocate(xu1)
         if (associated(yv1)) deallocate(yv1)

      else
!
! Send local data (LD) segment to processor 0 of mybloc
!
         call RPN_COMM_recv ( nia, 7, 'MPI_INTEGER', 0, &
                              tag, 'BLOC', status, err )
         if ((nia.gt.0).and.(nja.gt.0)) then
            F_nia  = nia  ; F_nja  = nja  ; F_nka_t  = nka_t ; F_nka_m  = nka_m 
            F_ntra = ntra ; F_vtype= vtyp ; F_diag_lvl = diag_lvl
            allocate (F_ana_am_8(F_nka_m+1), F_ana_bm_8(F_nka_m+1), &
                      F_ana_at_8(F_nka_t+1), F_ana_bt_8(F_nka_t+1) )
            allocate (F_xpaq(F_nia), F_ypaq(F_nja),&
                      F_xpau(F_nia), F_ypav(F_nja) )
            dim = 2*F_nia+2*F_nja
            allocate (buf(dim))
            call RPN_COMM_recv ( buf, dim, 'MPI_DOUBLE_PRECISION', 0, &
                                 tag, 'BLOC', status, err )
            F_xpaq(1:F_nia) = buf(              1  :F_nia      )
            F_xpau(1:F_nia) = buf(  F_nia      +1:2*F_nia      )
            F_ypaq(1:F_nja) = buf(2*F_nia      +1:2*F_nia+F_nja)
            F_ypav(1:F_nja) = buf(2*F_nia+F_nja+1:dim          )
            F_err = 0
            deallocate (buf)
         endif

      endif

      call handle_error ( F_err,'blk_dist_georef','Problem reading 3DF file '//trim(fn) )

      call RPN_COMM_bcast (F_ana_am_8, F_nka_m+1, "MPI_DOUBLE_PRECISION",0,"BLOC",err)
      call RPN_COMM_bcast (F_ana_bm_8, F_nka_m+1, "MPI_DOUBLE_PRECISION",0,"BLOC",err)
      call RPN_COMM_bcast (F_ana_at_8, F_nka_t+1, "MPI_DOUBLE_PRECISION",0,"BLOC",err)
      call RPN_COMM_bcast (F_ana_bt_8, F_nka_t+1, "MPI_DOUBLE_PRECISION",0,"BLOC",err)

      if (Ptopo_blocme.eq.0) then
         do iproc = 0, Ptopo_numpe_perb-1
            blk_indx(iproc,1) = blk_indx(iproc,1) - ifd_niad + 1
            blk_indx(iproc,2) = blk_indx(iproc,2) - ifd_niad + 1
            blk_indx(iproc,3) = blk_indx(iproc,3) - ifd_njad + 1
            blk_indx(iproc,4) = blk_indx(iproc,4) - ifd_njad + 1
         end do
      endif

 998  format (/' PROBLEM OPENING FILE: ',a/)
!
!----------------------------------------------------------------------
!
      return
      end
!
