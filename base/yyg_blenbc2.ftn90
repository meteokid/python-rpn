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
!**s/r yyg_blenbc2- to interpolate (cubic) and exchange or blend scalar data
!


      Subroutine yyg_blenbc2(tab_dst,tab_src,Minx,Maxx,Miny,Maxy,NK)
      implicit none
#include <arch_specific.hf>
!
!author
!           Abdessamad Qaddouri/V.Lee - October 2009
!
!     include 'mpif.h'
#include "ptopo.cdk"
#include "glb_ld.cdk"
#include "geomn.cdk"
#include "geomg.cdk"
#include "glb_pil.cdk"
#include "yyg_bln.cdk"

      integer Minx,Maxx,Miny,Maxy,Ni,Nj,NK,numproc
      real  tab_src (Minx:Maxx,Miny:Maxy,Nk)
      real  tab_dst (Minx:Maxx,Miny:Maxy,Nk)
      real*8  tab_src_8(Minx:Maxx,Miny:Maxy,NK)
      integer ierr,i,j,k,kk,kk_proc,m,mm,adr
      real, dimension (:,:), allocatable :: recv_pil,send_pil
      real sent,recv
!     integer status(MPI_STATUS_SIZE)
!     integer stat(MPI_STATUS_SIZE,Ptopo_numproc)
      integer status
      integer request(Ptopo_numproc*2)
      real*8  send_pil_8
      integer tag2,recvlen,sendlen,tag1,ireq
      tag2=14
      tag1=13
      
      sendlen=0
      recvlen=0
      ireq=0
      do kk=1,Bln_sendmaxproc
         sendlen=max(sendlen,Bln_send_len(kk))
      enddo
      do kk=1,Bln_recvmaxproc
         recvlen=max(recvlen,Bln_recv_len(kk))
      enddo
      

!     print *,'sendlen=',sendlen,' recvlen=',recvlen
      if (sendlen.gt.0) then
          allocate(send_pil(sendlen*NK,Bln_sendmaxproc))
!         assume rpn_comm_xch_halo already done on tab_src
          tab_src_8(:,:,:)=dble(tab_src(:,:,:))
      endif
      if (recvlen.gt.0) then
          allocate(recv_pil(recvlen*NK,Bln_recvmaxproc))
      endif
 
!
      do 100 kk=1,Bln_sendmaxproc
!
!        For each processor (in other colour)
      
         if (Ptopo_couleur.eq.0) then
             kk_proc = Bln_sendproc(kk)+Ptopo_numproc-1
         else
             kk_proc = Bln_sendproc(kk)-1
         endif

!        prepare to send to other colour processor
         if (Bln_send_len(kk).gt.0) then
!            prepare something to send

             mm=0

! make for west
             do m=1,Bln_send_len(kk)
                adr=Bln_send_adr(kk)+m

!$omp parallel private (mm,send_pil_8,i,j) &
!$omp          shared (tab_src_8,send_pil)
!$omp do
                do k=1,NK
                   mm=(m-1)*NK+k
                   call int_cub_lag2(send_pil_8,tab_src_8(l_minx,l_miny,k), &
                             Bln_send_imx(adr),Bln_send_imy(adr), &
                             Geomg_x_8,Geomg_y_8,l_minx,l_maxx,l_miny,l_maxy,                  &
                             Bln_send_xxr(adr),Bln_send_yyr(adr))
                   send_pil(mm,KK)=real(send_pil_8)
                enddo
!$omp enddo
!$omp end parallel
             enddo

             ireq = ireq+1
!            print *,'scalbc: sending',Bln_send_len(kk)*NK,' to ',kk_proc
!            call MPI_ISend(send_pil (1,KK),Bln_send_len(kk)*NK,MPI_REAL, &
!                                       kk_proc,tag2+Ptopo_world_myproc, &
!                                       MPI_COMM_WORLD,request(ireq),ierr)
             call RPN_COMM_ISend(send_pil (1,KK),Bln_send_len(kk)*NK,&
                                 'MPI_REAL',kk_proc,tag2+Ptopo_world_myproc, &
                                 'MULTIGRID',request(ireq),ierr)
         endif
 100  continue
!
!        check to receive from other colour processors
!
      do 200 kk=1,Bln_recvmaxproc
!        For each processor (in other colour)
      
         if (Ptopo_couleur.eq.0) then
             kk_proc = Bln_recvproc(kk)+Ptopo_numproc-1
         else
             kk_proc = Bln_recvproc(kk)-1
         endif
         if (Bln_recv_len(kk).gt.0) then
!            detect something to receive

             ireq = ireq+1
!            print *,'scalbc: receiving',Bln_recv_len(kk)*NK,' from ',kk_proc
!            call MPI_IRecv(recv_pil(1,KK),Bln_recv_len(kk)*NK,MPI_REAL, &
!                   kk_proc,tag2+kk_proc,MPI_COMM_WORLD,request(ireq),ierr)
             call RPN_COMM_IRecv(recv_pil(1,KK),Bln_recv_len(kk)*NK,'MPI_REAL',&
                    kk_proc,tag2+kk_proc, 'MULTIGRID',request(ireq),ierr)
         endif

 200  continue

! Wait for all done sending and receiving

!     call mpi_waitall(ireq,request,stat,ierr)
      call RPN_COMM_waitall_nostat(ireq,request,ierr)

! Now fill my results if I have received something

      if (recvlen.gt.0) then

          do 300 kk=1, Bln_recvmaxproc

             mm=0
             do m=1,Bln_recv_len(kk)
             adr=Bln_recv_adr(kk)+m
             do k=1,NK
                mm=mm+1
                tab_dst(Bln_recv_i(adr),Bln_recv_j(adr),k)= &
                     0.5*tab_dst(Bln_recv_i(adr),Bln_recv_j(adr),k) &
                   + 0.5*recv_pil(mm,KK)
             enddo
             enddo

 300  continue

       
      endif
      if (recvlen.gt.0)deallocate(recv_pil)
      if (sendlen.gt.0) deallocate(send_pil)

      return
      end

