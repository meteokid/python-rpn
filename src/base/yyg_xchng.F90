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

!**s/r yyg_xchng - Interpolate and exchange scalars

      subroutine yyg_xchng ( F_src, Minx,Maxx,Miny,Maxy, Nk, &
                             mono_L, F_interpo_S )
      use gem_options
      use geomh
      implicit none
#include <arch_specific.hf>

      character* (*) F_interpo_S
      logical mono_L
      integer Minx,Maxx,Miny,Maxy, Nk
      real F_src (Minx:Maxx,Miny:Maxy,Nk)

!author
!      Abdessamad Qaddouri/V.Lee - October 2009
!revision
! v4_60 - Qaddouri A.   - initial version
! v4_70 - Desgagne M.   - major revision

#include "ptopo.cdk"
#include "glb_ld.cdk"
#include "glb_pil.cdk"
#include "yyg_pil.cdk"
#include "yyg_pil0.cdk"

      integer ni,nj, numproc, status
      integer ierr,i,j,k,kk,kk_proc,m,mm,adr
      integer tag2,recvlen,sendlen,ireq,sendmaxproc,recvmaxproc
      integer request(Ptopo_numproc*2)
      real, dimension (:,:), allocatable :: recv_pil,send_pil
      real sent,recv
      real*8 wrk1(Minx:Maxx,Miny:Maxy,Nk), send_pil_8

      integer, dimension (:  ), pointer :: &
               sendproc ,  recvproc,  recv_len,  send_len, &
               recv_adr, send_adr, &
               recv_i   , recv_j   , send_imx , send_imy
      real*8,  dimension (: ), pointer :: send_xxr, send_yyr
!
!----------------------------------------------------------------------
!
      call timing_start2 ( 6, 'YYG_XCHNG', 0)
      tag2=14 ; sendlen=0 ; recvlen=0 ; ireq=0

      if (trim(F_interpo_S) == 'CUBIC') then
         sendmaxproc= Pil_sendmaxproc
         recvmaxproc= Pil_recvmaxproc
         send_len  => Pil_send_len
         recv_len  => Pil_recv_len
         sendproc  => Pil_sendproc
         send_len  => Pil_send_len
         send_adr  => Pil_send_adr
         send_imx  => Pil_send_imx
         send_imy  => Pil_send_imy
         send_xxr  => Pil_send_xxr
         send_yyr  => Pil_send_yyr
         recvproc  => Pil_recvproc
         recv_len  => Pil_recv_len
         recv_adr  => Pil_recv_adr
         recv_i    => Pil_recv_i
         recv_j    => Pil_recv_j
      else
! this need to be checked ...
         sendmaxproc= Pil0_sendmaxproc
         recvmaxproc= Pil0_recvmaxproc
         send_len  => Pil0_send_len
         recv_len  => Pil0_recv_len
         sendproc  => Pil0_sendproc
         send_len  => Pil0_send_len
         send_adr  => Pil0_send_adr
         send_imx  => Pil0_send_imx
         send_imy  => Pil0_send_imy
         send_xxr  => Pil0_send_xxr
         send_yyr  => Pil0_send_yyr
         recvproc  => Pil0_recvproc
         recv_len  => Pil0_recv_len
         recv_adr  => Pil0_recv_adr
         recv_i    => Pil0_recv_i
         recv_j    => Pil0_recv_j
      endif

      do kk= 1, sendmaxproc
         sendlen=max(sendlen,send_len(kk))
      enddo
      do kk= 1, recvmaxproc
         recvlen=max(recvlen,recv_len(kk))
      enddo

      call rpn_comm_xch_halo(f_src, Minx,Maxx,Miny,Maxy,l_ni,l_nj,Nk, &
                             G_halox,G_haloy,G_periodx,G_periody,l_ni,0)

      if (sendlen.gt.0) then
         allocate(send_pil(sendlen*Nk,sendmaxproc))
!     Double the precision on all values, including inside halo
!     If halo has undefined values, Intel will find floating invalid
         wrk1(:,:,:)= dble(F_src(:,:,:))
      endif
      if (recvlen.gt.0) then
          allocate(recv_pil(recvlen*NK,recvmaxproc))
      endif

      do 100 kk= 1, sendmaxproc

!        For each processor (in other colour)

         if (Ptopo_couleur.eq.0) then
            kk_proc = sendproc(kk)+Ptopo_numproc-1
         else
            kk_proc = sendproc(kk)-1
         endif

!        prepare to send to other colour processor
         if (send_len(kk).gt.0) then
!            prepare something to send

             adr=send_adr(kk)+1

             call yyg_interp1( send_pil(1,KK), wrk1, &
                       send_imx(adr), send_imy(adr), geomh_x_8,geomh_y_8,&
                       Minx,Maxx,Miny,Maxy,Nk,&
                       send_xxr(adr),send_yyr(adr),send_len(KK),&
                       mono_l,F_interpo_S )

             ireq = ireq+1
             call RPN_COMM_ISend(send_pil (1,KK),send_len(kk)*NK,&
                         'MPI_REAL',kk_proc,tag2+Ptopo_world_myproc,&
                         'MULTIGRID',request(ireq),ierr)
         endif

 100  continue
!
!        check to receive from other colour processors
!
      do 200 kk= 1, recvmaxproc
!        For each processor (in other colour)

         if (Ptopo_couleur.eq.0) then
             kk_proc = recvproc(kk)+Ptopo_numproc-1
         else
             kk_proc = recvproc(kk)-1
         endif

         if (recv_len(kk).gt.0) then
!            detect something to receive

            ireq = ireq+1
            call RPN_COMM_IRecv (recv_pil(1,KK),recv_len(kk)*NK  , &
                                 'MPI_REAL', kk_proc,tag2+kk_proc, &
                                 'MULTIGRID',request(ireq),ierr)
         endif

 200  continue

! Wait for all done sending and receiving

      call RPN_COMM_waitall_nostat(ireq,request,ierr)

! Now fill my results if I have received something

      if (recvlen.gt.0) then

         do 300 kk=1, recvmaxproc
           mm=0
           do m=1,recv_len(kk)
                 adr=recv_adr(kk)+m
              do k=1,Nk
                 mm=mm+1
                 F_src(recv_i(adr),recv_j(adr),k) = recv_pil(mm,KK)
              enddo
           enddo

 300     continue

      endif

      if (recvlen.gt.0) deallocate(recv_pil)
      if (sendlen.gt.0) deallocate(send_pil)
      call timing_stop (6)
!
!----------------------------------------------------------------------
!
      return
      end

