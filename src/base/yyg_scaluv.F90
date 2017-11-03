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
!**s/r yyg_scaluv - to interpolate and exchange scalar winds for Yin-Yang
!
      Subroutine yyg_scaluv(tabu_src,tabv_src,Minx,Maxx,Miny,Maxy,NK)
      use gem_options
      use geomh
      use glb_ld
      use glb_pil
      use ptopo
      use yyg_pil
      implicit none
#include <arch_specific.hf>
!
!author
!           Abdessamad Qaddouri/V.Lee - October 2009
!
!  PLEASE consult Abdessamad or Vivian before modifying this routine.
!
!revision
!  v4.8  V.Lee - Added NK to arguments for call to int_cubuv_lag
!

      integer Minx,Maxx,Miny,Maxy,NK
      real  tabu_src (Minx:Maxx,Miny:Maxy,Nk)
      real  tabv_src (Minx:Maxx,Miny:Maxy,Nk)
      real*8  tabu_src_8(Minx:Maxx,Miny:Maxy,NK)
      real*8  tabv_src_8(Minx:Maxx,Miny:Maxy,NK)
      integer ierr,k,kk,kk_proc,m,mm,adr
      real, dimension (:,:), allocatable :: recv_pil,send_pil
!     integer status(MPI_STATUS_SIZE)
!     integer stat(MPI_STATUS_SIZE,Ptopo_numproc)
      integer request(Ptopo_numproc*2)
      integer tag2,recvlen,sendlen,tag1,ireq
      tag2=14
      tag1=13

      sendlen=0
      recvlen=0
      ireq=0
      do kk=1,Pil_sendmaxproc
         sendlen=max(sendlen,Pil_send_len(kk))
      enddo
      do kk=1,Pil_recvmaxproc
         recvlen=max(recvlen,Pil_recv_len(kk))
      enddo

!     Will not assume rpn_comm_xch_halo is already done on tab_src
      call rpn_comm_xch_halo(tabu_src,l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj,Nk, &
                  G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      call rpn_comm_xch_halo(tabv_src,l_minx,l_maxx,l_miny,l_maxy,l_ni,l_nj,Nk, &
                  G_halox,G_haloy,G_periodx,G_periody,l_ni,0 )
      if (sendlen > 0) then
          allocate(send_pil(sendlen*NK*2,Pil_sendmaxproc))
          tabu_src_8(:,:,:)=dble(tabu_src(:,:,:))
          tabv_src_8(:,:,:)=dble(tabv_src(:,:,:))
      endif
      if (recvlen > 0) then
          allocate(recv_pil(recvlen*NK*2,Pil_recvmaxproc))
      endif

!
      do 100 kk=1,Pil_sendmaxproc
!
!        For each processor (in other colour)
         if (Ptopo_couleur == 0) then
             kk_proc = Pil_sendproc(kk)+Ptopo_numproc-1
         else
             kk_proc = Pil_sendproc(kk)-1
         endif

!        prepare to send to other colour processor
         if (Pil_send_len(kk) > 0) then
!            prepare something to send

             adr=Pil_send_adr(kk)+1

             call int_cubuv_lag(send_pil(1,KK),tabu_src_8,tabv_src_8,  &
                  Pil_send_imx(adr:adr+Pil_send_len(kk)),              &
                  Pil_send_imy(adr:adr+Pil_send_len(kk)),              &
                  geomh_x_8,geomh_y_8,l_minx,l_maxx,l_miny,l_maxy,NK,  &
                  Pil_send_xxr(adr:adr+Pil_send_len(kk)),              &
                  Pil_send_yyr(adr:adr+Pil_send_len(kk)),              &
                  Pil_send_len(kk),                                    &
                  Pil_send_s1(adr:adr+Pil_send_len(kk)),               &
                  Pil_send_s2(adr:adr+Pil_send_len(kk)),               &
                  Pil_send_s3(adr:adr+Pil_send_len(kk)),               &
                  Pil_send_s4(adr:adr+Pil_send_len(kk))            )

             ireq = ireq+1
!            print *,'scaluv: sending',Pil_send_len(kk)*NK*2,' to ',kk_proc
!            call MPI_ISend (send_pil(1,KK),Pil_send_len(kk)*NK*2,MPI_REAL, &
!                                         kk_proc,tag2+Ptopo_world_myproc, &
!                                         MPI_COMM_WORLD,request(ireq),ierr)
             call RPN_COMM_ISend (send_pil(1,KK),Pil_send_len(kk)*NK*2, &
                                  'MPI_REAL',kk_proc,tag2+Ptopo_world_myproc, &
                                          'MULTIGRID',request(ireq),ierr)
         endif
 100  continue
!
!        check to receive from other colour processor
!
      do 200 kk=1,Pil_recvmaxproc
!        For each processor (in other colour)

         if (Ptopo_couleur == 0) then
             kk_proc = Pil_recvproc(kk)+Ptopo_numproc-1
         else
             kk_proc = Pil_recvproc(kk)-1
         endif
         if (Pil_recv_len(kk) > 0) then
!            detect something to receive

             ireq = ireq+1
!            print *,'scaluv: receiving',Pil_send_len(kk)*NK*2,' from ',kk_proc
!            call MPI_IRecv(recv_pil(1,KK),Pil_recv_len(kk)*NK*2,MPI_REAL, &
!                   kk_proc,tag2+kk_proc,MPI_COMM_WORLD,request(ireq),ierr)
             call RPN_COMM_IRecv(recv_pil(1,KK),Pil_recv_len(kk)*NK*2, &
                                 'MPI_REAL', kk_proc,tag2+kk_proc, &
                                 'MULTIGRID',request(ireq),ierr)
         endif

 200  continue

!Wait for all done sending and receiving

!     call mpi_waitall(ireq,request,stat,ierr)
      call RPN_COMM_waitall_nostat(ireq,request,ierr)

! Now fill my results if I have received something

      if (recvlen > 0) then

          do 300 kk=1, Pil_recvmaxproc
             mm=0
             do m=1,Pil_recv_len(kk)
             adr=Pil_recv_adr(kk)+m
             do k=1,NK
                mm=mm+1
                tabu_src(Pil_recv_i(adr),Pil_recv_j(adr),k)=recv_pil(mm,KK)
                mm=mm+1
                tabv_src(Pil_recv_i(adr),Pil_recv_j(adr),k)=recv_pil(mm,KK)
             enddo
             enddo
 300  continue


      endif
      if (recvlen > 0) deallocate(recv_pil)
      if (sendlen > 0) deallocate(send_pil)

!
!
      return
      end

