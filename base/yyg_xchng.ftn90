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
#include "geomn.cdk"
#include "geomg.cdk"
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
               recvw_len, recve_len, recvs_len, recvn_len, &
               sendw_len, sende_len, sends_len, sendn_len, &
               recvw_adr, recve_adr, recvs_adr, recvn_adr, &
               sendw_adr, sende_adr, sends_adr, sendn_adr, &
               recv_i   , recv_j   , send_imx , send_imy
      real*8,  dimension (: ), pointer :: send_xxr, send_yyr
!
!----------------------------------------------------------------------
!
      tag2=14 ; sendlen=0 ; recvlen=0 ; ireq=0

      if (trim(F_interpo_S) == 'CUBIC') then
         sendmaxproc= Pil_sendmaxproc
         recvmaxproc= Pil_recvmaxproc
         send_len  => Pil_send_len
         recv_len  => Pil_recv_len
         sendproc  => Pil_sendproc
         send_len  => Pil_send_len
         sendw_len => Pil_sendw_len
         sendw_adr => Pil_sendw_adr
         sende_len => Pil_sende_len
         sende_adr => Pil_sende_adr
         sendn_len => Pil_sendn_len
         sendn_adr => Pil_sendn_adr
         sends_len => Pil_sends_len
         sends_adr => Pil_sends_adr
         send_imx  => Pil_send_imx
         send_imy  => Pil_send_imy
         send_xxr  => Pil_send_xxr
         send_yyr  => Pil_send_yyr
         recvproc  => Pil_recvproc
         recv_len  => Pil_recv_len
         recvw_len => Pil_recvw_len
         recvw_adr => Pil_recvw_adr
         recve_len => Pil_recve_len
         recve_adr => Pil_recve_adr
         recvn_len => Pil_recvn_len
         recvn_adr => Pil_recvn_adr
         recvs_len => Pil_recvs_len
         recvs_adr => Pil_recvs_adr
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
         sendw_len => Pil0_sendw_len
         sendw_adr => Pil0_sendw_adr
         sende_len => Pil0_sende_len
         sende_adr => Pil0_sende_adr
         sendn_len => Pil0_sendn_len
         sendn_adr => Pil0_sendn_adr
         sends_len => Pil0_sends_len
         sends_adr => Pil0_sends_adr
         send_imx  => Pil0_send_imx
         send_imy  => Pil0_send_imy
         send_xxr  => Pil0_send_xxr
         send_yyr  => Pil0_send_yyr
         recvproc  => Pil0_recvproc
         recv_len  => Pil0_recv_len
         recvw_len => Pil0_recvw_len
         recvw_adr => Pil0_recvw_adr
         recve_len => Pil0_recve_len
         recve_adr => Pil0_recve_adr
         recvn_len => Pil0_recvn_len
         recvn_adr => Pil0_recvn_adr
         recvs_len => Pil0_recvs_len
         recvs_adr => Pil0_recvs_adr
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
         wrk1(:,:,:)= dble(F_src(:,:,:))
      endif
      if (recvlen.gt.0) then
          allocate(recv_pil(recvlen*NK,recvmaxproc))
      endif

      do kk= 1, sendmaxproc

!        For each processor (in other colour)
      
         if (Ptopo_couleur.eq.0) then
            kk_proc = sendproc(kk)+Ptopo_numproc-1
         else
            kk_proc = sendproc(kk)-1
         endif

!        prepare to send to other colour processor
         if (send_len(kk).gt.0) then
!            prepare something to send

            mm=0

! make for west
            do m= 1, sendw_len(kk)
               adr=sendw_adr(kk)+m

!$omp parallel private (mm,send_pil_8,i,j) &
!$omp          shared (wrk1,send_pil)
!$omp do
               do k= 1, Nk
                  mm= (m-1)*NK+k
                  call yyg_interp ( send_pil_8, wrk1(l_minx,l_miny,k),&
                                    send_imx(adr), send_imy(adr)     ,&
                              Geomg_x_8,Geomg_y_8,Minx,Maxx,Miny,Maxy,&
                       send_xxr(adr),send_yyr(adr),mono_l,F_interpo_S )
                  send_pil(mm,KK)= real(send_pil_8)
               enddo
!$omp enddo
!$omp end parallel
            enddo

! make for east
            do m= 1, sende_len(kk)
               adr=sende_adr(kk)+m

!$omp parallel private (mm,send_pil_8,i,j) &
!$omp          shared (wrk1,send_pil)
!$omp do
               do k= 1, Nk
                  mm= (sendw_len(kk)+m-1)*Nk+k
                  call yyg_interp ( send_pil_8, wrk1(l_minx,l_miny,k),&
                                    send_imx(adr), send_imy(adr)     ,&
                              Geomg_x_8,Geomg_y_8,Minx,Maxx,Miny,Maxy,&
                       send_xxr(adr),send_yyr(adr),mono_l,F_interpo_S )
                  send_pil(mm,KK)= real(send_pil_8)
               enddo
!$omp enddo
!$omp end parallel
            enddo

! make for south
            do m= 1, sends_len(kk)
               adr=sends_adr(kk)+m

!$omp parallel private (mm,send_pil_8,i,j) &
!$omp          shared (wrk1,send_pil)
!$omp do
               do k= 1, Nk
                  mm= (sendw_len(kk)+sende_len(kk)+m-1)*Nk+k
                  call yyg_interp ( send_pil_8, wrk1(l_minx,l_miny,k),&
                                    send_imx(adr),send_imy(adr)      ,&
                              Geomg_x_8,Geomg_y_8,Minx,Maxx,Miny,Maxy,&
                       send_xxr(adr),send_yyr(adr),mono_l,F_interpo_S )
                  send_pil(mm,KK)= real(send_pil_8)
               enddo
!$omp enddo
!$omp end parallel
            enddo

! make for north
            do m= 1, sendn_len(kk)
               adr=sendn_adr(kk)+m

!$omp parallel private (mm,send_pil_8,i,j) &
!$omp          shared (wrk1,send_pil)
!$omp do
               do k=1,Nk
                  mm=(sendw_len(kk)+sende_len(kk)+sends_len(kk)+m-1)*Nk+k
                  call yyg_interp ( send_pil_8, wrk1(l_minx,l_miny,k),&
                                    send_imx(adr),send_imy(adr)      ,&
                              Geomg_x_8,Geomg_y_8,Minx,Maxx,Miny,Maxy,&
                       send_xxr(adr),send_yyr(adr),mono_l,F_interpo_S )
                  send_pil(mm,KK)= real(send_pil_8)
               enddo
!$omp enddo
!$omp end parallel
            enddo

            ireq = ireq+1
            call RPN_COMM_ISend(send_pil (1,KK),send_len(kk)*NK,&
                         'MPI_REAL',kk_proc,tag2+Ptopo_world_myproc,&
                         'MULTIGRID',request(ireq),ierr)
         endif

      end do
!
!        check to receive from other colour processors
!
      do kk= 1, recvmaxproc
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

      end do

! Wait for all done sending and receiving

      call RPN_COMM_waitall_nostat(ireq,request,ierr)

! Now fill my results if I have received something

      if (recvlen.gt.0) then

         do kk=1, recvmaxproc
! fill my west
            mm=0
            do m= 1, recvw_len(kk)
               adr=recvw_adr(kk)+m
               do k=1,Nk
                  mm=mm+1
                  F_src(recv_i(adr),recv_j(adr),k)= recv_pil(mm,KK)
               enddo
            enddo
! fill my east
            do m=1,recve_len(kk)
               adr=recve_adr(kk)+m
               do k=1,Nk
                  mm=mm+1
                  F_src(recv_i(adr),recv_j(adr),k)= recv_pil(mm,KK)
               enddo
            enddo
! fill my south
            do m=1,recvs_len(kk)
               adr=recvs_adr(kk)+m
               do k=1,Nk
                  mm=mm+1
                  F_src(recv_i(adr),recv_j(adr),k)= recv_pil(mm,KK)
               enddo
            enddo
! fill my north
            do m=1,recvn_len(kk)
               adr=recvn_adr(kk)+m
               do k=1,Nk
                  mm=mm+1
                  F_src(recv_i(adr),recv_j(adr),k)= recv_pil(mm,KK)
               enddo
            enddo
            
         end do
       
      endif

      if (recvlen.gt.0) deallocate(recv_pil)
      if (sendlen.gt.0) deallocate(send_pil)
!
!----------------------------------------------------------------------
!
      return
      end

