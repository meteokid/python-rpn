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
!**s/r yyg_vecbc2 - to interpolate and exchange V wind
!


      Subroutine yyg_vecbc2(tab_dst,tabv_src,tabu_src,Minx,Maxx,Miny,Maxy,NK)
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
#include "yyg_pilv.cdk"

      integer Minx,Maxx,Miny,Maxy,Ni,Nj,NK,numproc
      real  tabu_src (Minx:Maxx,Miny:Maxy,Nk), tabv_src (Minx:Maxx,Miny:Maxy,Nk)
      real  tab_dst (Minx:Maxx,Miny:Maxy,Nk)
      real*8  tabu_src_8(Minx:Maxx,Miny:Maxy,NK)
      real*8  tabv_src_8(Minx:Maxx,Miny:Maxy,NK)
      integer ierr,i,j,k,kk,kk_proc,m,mm,adr
      real, dimension (:,:), allocatable :: recv_pil,send_pil
      real sent,recv
!     integer status(MPI_STATUS_SIZE)
!     integer stat(MPI_STATUS_SIZE,Ptopo_numproc)
      integer status
      integer request(Ptopo_numproc*2)
      real*8  send_pil_8,tabu_8,tabv_8
      integer tag2,recvlen,sendlen,tag1,ireq
      tag2=14
      tag1=13
      
      sendlen=0
      recvlen=0
      ireq=0
      do kk=1,Pil_vsendmaxproc
         sendlen=max(sendlen,Pil_vsend_len(kk))
      enddo
      do kk=1,Pil_vrecvmaxproc
         recvlen=max(recvlen,Pil_vrecv_len(kk))
      enddo
      

!     print *,'sendlen=',sendlen,' recvlen=',recvlen
      if (sendlen.gt.0) then
          allocate(send_pil(sendlen*NK,Pil_vsendmaxproc))
!         assume rpn_comm_xch_halo already done on tab_src
          tabu_src_8(:,:,:)=dble(tabu_src(:,:,:))
          tabv_src_8(:,:,:)=dble(tabv_src(:,:,:))
      endif
      if (recvlen.gt.0) then
          allocate(recv_pil(recvlen*NK,Pil_vrecvmaxproc))
      endif
 
!
      do 100 kk=1,Pil_vsendmaxproc
!
!        For each processor (in other colour)
         if (Ptopo_couleur.eq.0) then
             kk_proc = Pil_vsendproc(kk)+Ptopo_numproc-1
         else
             kk_proc = Pil_vsendproc(kk)-1
         endif

!        prepare to send to other colour processor
         if (Pil_vsend_len(kk).gt.0) then
!            prepare something to send

             mm=0

! make for west
             do m=1,Pil_vsendw_len(kk)
                adr=Pil_vsendw_adr(kk)+m

!$omp parallel private (mm,tabu_8,tabv_8) &
!$omp          shared (tabu_src_8,tabv_src_8,send_pil)
!$omp do
                do k=1,NK
!                  mm=mm+1
                   mm=(m-1)*NK+k
                   call int_cub_lag2(tabv_8,tabv_src_8(l_minx,l_miny,k),  &
                             Pil_vsend_imx1(adr),Pil_vsend_imy1(adr), &
                             Geomg_x_8,Geomg_yv_8,l_minx,l_maxx,l_miny,l_maxy,              & 
                             Pil_vsend_xxr(adr),Pil_vsend_yyr(adr))
                   call int_cub_lag2(tabu_8,tabu_src_8(l_minx,l_miny,k),  &
                             Pil_vsend_imx2(adr),Pil_vsend_imy2(adr), &
                             Geomg_xu_8,Geomg_y_8,l_minx,l_maxx,l_miny,l_maxy,              &
                             Pil_vsend_xxr(adr),Pil_vsend_yyr(adr))
                   send_pil(mm,KK)=                                       &
                   real(Pil_vsend_s1(adr)*tabv_8 + Pil_vsend_s2(adr)*tabu_8)
                enddo
!$omp enddo
!$omp end parallel
             enddo
! make for east
             do m=1,Pil_vsende_len(kk)
                adr=Pil_vsende_adr(kk)+m

!$omp parallel private (mm,tabu_8,tabv_8) &
!$omp          shared (tabu_src_8,tabv_src_8,send_pil)
!$omp do
                do k=1,NK
                   mm=(Pil_vsendw_len(kk)+m-1)*NK+k
                   call int_cub_lag2(tabv_8,tabv_src_8(l_minx,l_miny,k), &
                             Pil_vsend_imx1(adr),Pil_vsend_imy1(adr),&
                             Geomg_x_8,Geomg_yv_8,l_minx,l_maxx,l_miny,l_maxy,             &
                             Pil_vsend_xxr(adr),Pil_vsend_yyr(adr))
                   call int_cub_lag2(tabu_8,tabu_src_8(l_minx,l_miny,k), &
                             Pil_vsend_imx2(adr),Pil_vsend_imy2(adr),&
                             Geomg_xu_8,Geomg_y_8,l_minx,l_maxx,l_miny,l_maxy,             &
                             Pil_vsend_xxr(adr),Pil_vsend_yyr(adr))
                   send_pil(mm,KK)=                                      &
                   real(Pil_vsend_s1(adr)*tabv_8 + Pil_vsend_s2(adr)*tabu_8)
                enddo
!$omp enddo
!$omp end parallel
             enddo
! make for south
             do m=1,Pil_vsends_len(kk)
                adr=Pil_vsends_adr(kk)+m

!$omp parallel private (mm,tabu_8,tabv_8) &
!$omp          shared (tabu_src_8,tabv_src_8,send_pil)
!$omp do
                do k=1,NK
                   mm=(Pil_vsendw_len(kk)+Pil_vsende_len(kk)+m-1)*NK+k
                   call int_cub_lag2(tabv_8,tabv_src_8(l_minx,l_miny,k), &
                             Pil_vsend_imx1(adr),Pil_vsend_imy1(adr),&
                             Geomg_x_8,Geomg_yv_8,l_minx,l_maxx,l_miny,l_maxy,             &
                             Pil_vsend_xxr(adr),Pil_vsend_yyr(adr))
                   call int_cub_lag2(tabu_8,tabu_src_8(l_minx,l_miny,k), &
                             Pil_vsend_imx2(adr),Pil_vsend_imy2(adr),&
                             Geomg_xu_8,Geomg_y_8,l_minx,l_maxx,l_miny,l_maxy,             &
                             Pil_vsend_xxr(adr),Pil_vsend_yyr(adr))
                   send_pil(mm,KK)=                                      & 
                   real(Pil_vsend_s1(adr)*tabv_8 + Pil_vsend_s2(adr)*tabu_8)
                enddo
!$omp enddo
!$omp end parallel
             enddo
! make for north
             do m=1,Pil_vsendn_len(kk)
                adr=Pil_vsendn_adr(kk)+m

!$omp parallel private (mm,tabu_8,tabv_8) &
!$omp          shared (tabu_src_8,tabv_src_8,send_pil)
!$omp do
                do k=1,NK
                   mm=(Pil_vsendw_len(kk)+Pil_vsende_len(kk)+Pil_vsends_len(kk)+m-1)*NK+k
                   call int_cub_lag2(tabv_8,tabv_src_8(l_minx,l_miny,k), &
                             Pil_vsend_imx1(adr),Pil_vsend_imy1(adr),&
                             Geomg_x_8,Geomg_yv_8,l_minx,l_maxx,l_miny,l_maxy,             &
                             Pil_vsend_xxr(adr),Pil_vsend_yyr(adr))
                   call int_cub_lag2(tabu_8,tabu_src_8(l_minx,l_miny,k), &
                             Pil_vsend_imx2(adr),Pil_vsend_imy2(adr),&
                             Geomg_xu_8,Geomg_y_8,l_minx,l_maxx,l_miny,l_maxy,             &
                             Pil_vsend_xxr(adr),Pil_vsend_yyr(adr))
                   send_pil(mm,KK)=                                      &
                   real(Pil_vsend_s1(adr)*tabv_8 + Pil_vsend_s2(adr)*tabu_8)
                enddo
!$omp enddo
!$omp end parallel
             enddo

             ireq = ireq+1
!            print *,'vecbc2: sending',Pil_vsend_len(kk)*NK,' to ',kk_proc
!            call MPI_ISend (send_pil(1,KK),Pil_vsend_len(kk)*NK,MPI_REAL, &
!                                        kk_proc,tag2+Ptopo_world_myproc, &
!                                        MPI_COMM_WORLD,request(ireq),ierr)
             call RPN_COMM_ISend (send_pil(1,KK),Pil_vsend_len(kk)*NK, &
                                  'MPI_REAL',kk_proc,tag2+Ptopo_world_myproc, &
                                  'MULTIGRID',request(ireq),ierr)
         endif
 100  continue
!
!        check to receive from other colour processor
!
      do 200 kk=1,Pil_vrecvmaxproc
!        For each processor (in other colour)

         if (Ptopo_couleur.eq.0) then
             kk_proc = Pil_vrecvproc(kk)+Ptopo_numproc-1
         else
             kk_proc = Pil_vrecvproc(kk)-1
         endif
         if (Pil_vrecv_len(kk).gt.0) then
!            detect something to receive

             ireq = ireq+1
!            print *,'vecbc2: receiving',Pil_vrecv_len(kk)*NK,' from ',kk_proc
!            call MPI_IRecv(recv_pil(1,KK),Pil_vrecv_len(kk)*NK,MPI_REAL, &
!                   kk_proc,tag2+kk_proc,MPI_COMM_WORLD,request(ireq),ierr)
             call RPN_COMM_IRecv(recv_pil(1,KK),Pil_vrecv_len(kk)*NK,'MPI_REAL',&
                    kk_proc,tag2+kk_proc,'MULTIGRID',request(ireq),ierr)
         endif

 200  continue

!Wait for all done sending and receiving

!     call mpi_waitall(ireq,request,stat,ierr)
      call RPN_COMM_waitall_nostat(ireq,request,ierr)

! Now fill my results if I have received something

      if (recvlen.gt.0) then

          do 300 kk=1,Pil_vrecvmaxproc
! fill my west
             mm=0
             do m=1,Pil_vrecvw_len(kk)
             adr=Pil_vrecvw_adr(kk)+m
             do k=1,NK
                mm=mm+1
                tab_dst(Pil_vrecv_i(adr),Pil_vrecv_j(adr),k)=recv_pil(mm,KK)
             enddo
             enddo
! fill my east
             do m=1,Pil_vrecve_len(kk)
             adr=Pil_vrecve_adr(kk)+m
             do k=1,NK
                mm=mm+1
                tab_dst(Pil_vrecv_i(adr),Pil_vrecv_j(adr),k)=recv_pil(mm,KK)
             enddo
             enddo
! fill my south
             do m=1,Pil_vrecvs_len(kk)
             adr=Pil_vrecvs_adr(kk)+m
             do k=1,NK
                mm=mm+1
                tab_dst(Pil_vrecv_i(adr),Pil_vrecv_j(adr),k)=recv_pil(mm,KK)
             enddo
             enddo
! fill my north
             do m=1,Pil_vrecvn_len(kk)
             adr=Pil_vrecvn_adr(kk)+m
             do k=1,NK
                mm=mm+1
                tab_dst(Pil_vrecv_i(adr),Pil_vrecv_j(adr),k)=recv_pil(mm,KK)
             enddo
             enddo

 300  continue

       
      endif
      if (recvlen.gt.0)deallocate(recv_pil)
      if (sendlen.gt.0) deallocate(send_pil)

!
!
      return
      end

