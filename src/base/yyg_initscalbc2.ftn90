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
!***s/r yyg_initscalbc2 - to initialize communication pattern for cubic
!                            interpolation of scalar fields
!


      Subroutine yyg_initscalbc2()
      implicit none
#include <arch_specific.hf>
!
!author
!           Abdessamad Qaddouri/ V.lee - September 2011
!
#include "ptopo.cdk"
#include "glb_ld.cdk"
#include "geomn.cdk"
#include "glb_pil.cdk"
#include "yyg_pil.cdk"

      integer err,Ndim,i,j,k,imx,imy,kk,ii,jj,ki,ksend,krecv
      integer kkproc
      integer, dimension (:), pointer :: recv_len,recvw_len,recve_len,recvs_len,recvn_len
      integer, dimension (:), pointer :: send_len,sendw_len,sende_len,sends_len,sendn_len
      real*8  xx_8(G_ni,G_nj),yy_8(G_ni,G_nj)
      real*8  t,p,s(2,2),h1,h2
      real*8  x_d,y_d,x_a,y_a   
!
!     Take a global copy of xg,yg
      do j=1,G_nj
      do i=1,G_ni
         xx_8(i,j)=G_xg_8(i)
      enddo
      enddo
      do j=1,G_nj
      do i=1,G_ni
         yy_8(i,j)=G_yg_8(j)
      enddo
      enddo
      h1=G_xg_8(2)-G_xg_8(1)
      h2=G_yg_8(2)-G_yg_8(1)
!
!

! And allocate temp vectors needed for counting for each processor
!
      allocate (recv_len (Ptopo_numproc))
      allocate (recvw_len(Ptopo_numproc))
      allocate (recve_len(Ptopo_numproc))
      allocate (recvs_len(Ptopo_numproc))
      allocate (recvn_len(Ptopo_numproc))
      allocate (send_len (Ptopo_numproc))
      allocate (sendw_len(Ptopo_numproc))
      allocate (sende_len(Ptopo_numproc))
      allocate (sends_len(Ptopo_numproc))
      allocate (sendn_len(Ptopo_numproc))
      recv_len (:)=0
      recvw_len(:)=0
      recve_len(:)=0
      recvs_len(:)=0
      recvn_len(:)=0
      send_len (:)=0
      sendw_len(:)=0
      sende_len(:)=0
      sends_len(:)=0
      sendn_len(:)=0
!
! FIRST PASS is to find the number of processor to tag for
! communication and the number of items to send and receive for each
! processor before allocating the vectors
!
! WEST section

      do j=1, G_nj
      do i=1,Glb_pil_w
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx,imy,x_a,y_a, &
                          G_xg_8(1),G_yg_8(1),h1,h2,1,1)
         imx = min(max(imx-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy = min(max(imy-1,glb_pil_s+1),G_nj-glb_pil_n-3)
! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (imx.ge.Ptopo_gindx(1,kk).and.imx.le.Ptopo_gindx(2,kk).and. &
                    imy.ge.Ptopo_gindx(3,kk).and.imy.le.Ptopo_gindx(4,kk)) then
                    recvw_len(kk)=recvw_len(kk)+1
                endif
             enddo       
         endif

! check to send to who
         if (imx.ge.l_i0.and.imx.le.l_i0+l_ni-1 .and. &
             imy.ge.l_j0.and.imy.le.l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (i  .ge.Ptopo_gindx(1,kk).and.i  .le.Ptopo_gindx(2,kk).and. &
                    j  .ge.Ptopo_gindx(3,kk).and.j  .le.Ptopo_gindx(4,kk))then
                    sendw_len(kk)=sendw_len(kk)+1
                endif
             enddo       
         endif
      enddo   
      enddo   
!
!
! East section
      do j=1, G_nj
      do i=G_ni-Glb_pil_e+1,G_ni
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx,imy,x_a,y_a, &
                          G_xg_8(1),G_yg_8(1),h1,h2,1,1)
         imx = min(max(imx-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy = min(max(imy-1,glb_pil_s+1),G_nj-glb_pil_n-3)
! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (imx.ge.Ptopo_gindx(1,kk).and.imx.le.Ptopo_gindx(2,kk).and. &
                    imy.ge.Ptopo_gindx(3,kk).and.imy.le.Ptopo_gindx(4,kk))then
                    recve_len(kk)=recve_len(kk)+1
                endif
             enddo       
         endif

! check to send to who
         if (imx.ge.l_i0.and.imx.le.l_i0+l_ni-1 .and. &
             imy.ge.l_j0.and.imy.le.l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (i  .ge.Ptopo_gindx(1,kk).and.i  .le.Ptopo_gindx(2,kk).and. &
                    j  .ge.Ptopo_gindx(3,kk).and.j  .le.Ptopo_gindx(4,kk))then
                    sende_len(kk)=sende_len(kk)+1
                endif
             enddo       
         endif
      enddo   
      enddo   
!
! South section
      do j=1,Glb_pil_s
      do i=1+Glb_pil_w,G_ni-Glb_pil_e

         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx,imy,x_a,y_a, &
                          G_xg_8(1),G_yg_8(1),h1,h2,1,1)
         imx = min(max(imx-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy = min(max(imy-1,glb_pil_s+1),G_nj-glb_pil_n-3)
! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (imx.ge.Ptopo_gindx(1,kk).and.imx.le.Ptopo_gindx(2,kk).and. &
                    imy.ge.Ptopo_gindx(3,kk).and.imy.le.Ptopo_gindx(4,kk))then
                    recvs_len(kk)=recvs_len(kk)+1
                endif
             enddo       
         endif

! check to send to who
         if (imx.ge.l_i0.and.imx.le.l_i0+l_ni-1 .and. &
             imy.ge.l_j0.and.imy.le.l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (i  .ge.Ptopo_gindx(1,kk).and.i  .le.Ptopo_gindx(2,kk).and. &
                    j  .ge.Ptopo_gindx(3,kk).and.j  .le.Ptopo_gindx(4,kk))then
                    sends_len(kk)=sends_len(kk)+1
                endif
             enddo       
         endif
      enddo   
      enddo   
!
! North section
      do j=G_nj-Glb_pil_n+1,G_nj
      do i=1+Glb_pil_w,G_ni-Glb_pil_e

         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx,imy,x_a,y_a, &
                          G_xg_8(1),G_yg_8(1),h1,h2,1,1)
         imx = min(max(imx-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy = min(max(imy-1,glb_pil_s+1),G_nj-glb_pil_n-3)

! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (imx.ge.Ptopo_gindx(1,kk).and.imx.le.Ptopo_gindx(2,kk).and. &
                    imy.ge.Ptopo_gindx(3,kk).and.imy.le.Ptopo_gindx(4,kk))then
                    recvn_len(kk)=recvn_len(kk)+1
                endif
             enddo       
         endif

! check to send to who
         if (imx.ge.l_i0.and.imx.le.l_i0+l_ni-1 .and. &
             imy.ge.l_j0.and.imy.le.l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (i  .ge.Ptopo_gindx(1,kk).and.i  .le.Ptopo_gindx(2,kk).and. &
                    j  .ge.Ptopo_gindx(3,kk).and.j  .le.Ptopo_gindx(4,kk))then
                    sendn_len(kk)=sendn_len(kk)+1
                endif
             enddo       
         endif
      enddo   
      enddo   
!
! Obtain sum of elements to send and receive for each processor
! and the total memory needed to store and receive for each processor
!
     Pil_send_all=0 
     Pil_recv_all=0 
     Pil_sendmaxproc=0
     Pil_recvmaxproc=0
     
     do kk=1,Ptopo_numproc
        send_len(kk)=sendw_len(kk)+sende_len(kk) + sends_len(kk)+sendn_len(kk)
        recv_len(kk)=recvw_len(kk)+recve_len(kk) + recvs_len(kk)+recvn_len(kk)
        Pil_send_all=send_len(kk)+Pil_send_all
        Pil_recv_all=recv_len(kk)+Pil_recv_all

        if (send_len(kk).gt.0) Pil_sendmaxproc=Pil_sendmaxproc+1
        if (recv_len(kk).gt.0) Pil_recvmaxproc=Pil_recvmaxproc+1
     enddo
!
!     print *,'Allocate common vectors'
      allocate (Pil_recvproc(Pil_recvmaxproc))
      allocate (Pil_recv_len(Pil_recvmaxproc))
      allocate (Pil_recvw_len(Pil_recvmaxproc))
      allocate (Pil_recve_len(Pil_recvmaxproc))
      allocate (Pil_recvs_len(Pil_recvmaxproc))
      allocate (Pil_recvn_len(Pil_recvmaxproc))
      allocate (Pil_recvw_adr(Pil_recvmaxproc))
      allocate (Pil_recve_adr(Pil_recvmaxproc))
      allocate (Pil_recvs_adr(Pil_recvmaxproc))
      allocate (Pil_recvn_adr(Pil_recvmaxproc))

      allocate (Pil_sendproc(Pil_sendmaxproc))
      allocate (Pil_send_len(Pil_sendmaxproc))
      allocate (Pil_sendw_len(Pil_sendmaxproc))
      allocate (Pil_sende_len(Pil_sendmaxproc))
      allocate (Pil_sends_len(Pil_sendmaxproc))
      allocate (Pil_sendn_len(Pil_sendmaxproc))
      allocate (Pil_sendw_adr(Pil_sendmaxproc))
      allocate (Pil_sende_adr(Pil_sendmaxproc))
      allocate (Pil_sends_adr(Pil_sendmaxproc))
      allocate (Pil_sendn_adr(Pil_sendmaxproc))
      Pil_recvw_len(:) = 0
      Pil_recve_len(:) = 0
      Pil_recvs_len(:) = 0
      Pil_recvn_len(:) = 0
      Pil_sendw_len(:) = 0
      Pil_sende_len(:) = 0
      Pil_sends_len(:) = 0
      Pil_sendn_len(:) = 0
      Pil_recvw_adr(:) = 0
      Pil_recve_adr(:) = 0
      Pil_recvs_adr(:) = 0
      Pil_recvn_adr(:) = 0
      Pil_sendw_adr(:) = 0
      Pil_sende_adr(:) = 0
      Pil_sends_adr(:) = 0
      Pil_sendn_adr(:) = 0

!    print*,'Pil_sendmaxproc=',Pil_sendmaxproc,'recvmaxproc=',Pil_recvmaxproc
       
     ksend=0
     krecv=0
     Pil_send_all=0
     Pil_recv_all=0
!
! Fill the lengths and addresses for selected processors to communicate
!
     do kk=1,Ptopo_numproc
        if (send_len(kk).gt.0) then
            ksend=ksend+1
            Pil_sendproc(ksend)=kk
            Pil_send_len(ksend)=send_len(kk)
            Pil_sendw_len(ksend)=sendw_len(kk)
            Pil_sende_len(ksend)=sende_len(kk)
            Pil_sends_len(ksend)=sends_len(kk)
            Pil_sendn_len(ksend)=sendn_len(kk)

            Pil_sendw_adr(ksend)= Pil_send_all
            Pil_send_all= Pil_send_all + Pil_send_len(ksend)
            Pil_sende_adr(ksend)= Pil_sendw_adr(ksend)+Pil_sendw_len(ksend)
            Pil_sends_adr(ksend)= Pil_sende_adr(ksend)+Pil_sende_len(ksend)
            Pil_sendn_adr(ksend)= Pil_sends_adr(ksend)+Pil_sends_len(ksend)
        endif
        if (recv_len(kk).gt.0) then
            krecv=krecv+1
            Pil_recvproc(krecv)=kk
            Pil_recv_len(krecv)=recv_len(kk)
            Pil_recvw_len(krecv)=recvw_len(kk)
            Pil_recve_len(krecv)=recve_len(kk)
            Pil_recvs_len(krecv)=recvs_len(kk)
            Pil_recvn_len(krecv)=recvn_len(kk)

            Pil_recvw_adr(krecv)= Pil_recv_all
            Pil_recv_all= Pil_recv_all + Pil_recv_len(krecv)
            Pil_recve_adr(krecv)= Pil_recvw_adr(krecv)+Pil_recvw_len(krecv)
            Pil_recvs_adr(krecv)= Pil_recve_adr(krecv)+Pil_recve_len(krecv)
            Pil_recvn_adr(krecv)= Pil_recvs_adr(krecv)+Pil_recvs_len(krecv)
        endif
            
     enddo
!    print *,'krecv=',krecv,'Pil_recvmaxproc=',Pil_recvmaxproc
!    print *,'ksend=',ksend,'Pil_sendmaxproc=',Pil_sendmaxproc

!     print *,'Summary of comm procs'
!     do kk=1,Pil_recvmaxproc
!       print *,'From proc:',Pil_recvproc(kk),'Pil_recv_len',Pil_recvw_len(kk),Pil_recve_len(kk),Pil_recvs_len(kk),Pil_recvn_len(kk),'adr',Pil_recvw_adr(kk),Pil_recve_adr(kk),Pil_recvs_adr(kk),Pil_recvn_adr(kk)
!     enddo
!     do kk=1,Pil_sendmaxproc
!       print *,'To proc:',Pil_sendproc(kk),'Pil_send_len',Pil_sendw_len(kk),Pil_sende_len(kk),Pil_sends_len(kk),Pil_sendn_len(kk),'adr',Pil_sendw_adr(kk),Pil_sende_adr(kk),Pil_sends_adr(kk),Pil_sendn_adr(kk)
!     enddo

!
! Now allocate the vectors needed for sending and receiving each processor
!
      if (Pil_recv_all.gt.0) then
          allocate (Pil_recv_i(Pil_recv_all))
          allocate (Pil_recv_j(Pil_recv_all))
          Pil_recv_i(:) = 0
          Pil_recv_j(:) = 0
      endif

      if (Pil_send_all.gt.0) then
          allocate (Pil_send_imx(Pil_send_all))
          allocate (Pil_send_imy(Pil_send_all))
          allocate (Pil_send_xxr(Pil_send_all))
          allocate (Pil_send_yyr(Pil_send_all))
          allocate (Pil_send_s1(Pil_send_all))
          allocate (Pil_send_s2(Pil_send_all))
          allocate (Pil_send_s3(Pil_send_all))
          allocate (Pil_send_s4(Pil_send_all))
          Pil_send_imx(:) = 0
          Pil_send_imy(:) = 0
          Pil_send_xxr(:) = 0.0
          Pil_send_yyr(:) = 0.0
          Pil_send_s1(:) = 0.0
          Pil_send_s2(:) = 0.0
          Pil_send_s3(:) = 0.0
          Pil_send_s4(:) = 0.0
      endif
!

      recvw_len(:)=0
      recve_len(:)=0
      recvs_len(:)=0
      recvn_len(:)=0
      sendw_len(:)=0
      sende_len(:)=0
      sends_len(:)=0
      sendn_len(:)=0
!
! SECOND PASS is to initialize the vectors with information for communication
!
! WEST section

      do j=1, G_nj
      do i=1,Glb_pil_w
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx,imy,x_a,y_a, &
                          G_xg_8(1),G_yg_8(1),h1,h2,1,1)
         imx = min(max(imx-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy = min(max(imy-1,glb_pil_s+1),G_nj-glb_pil_n-3)
! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Pil_recvmaxproc
                ki=Pil_recvproc(kk)
                if (imx.ge.Ptopo_gindx(1,ki).and.imx.le.Ptopo_gindx(2,ki).and. &
                    imy.ge.Ptopo_gindx(3,ki).and.imy.le.Ptopo_gindx(4,ki))then
                    recvw_len(kk)=recvw_len(kk)+1
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil_recv_i(Pil_recvw_adr(kk)+recvw_len(kk))=ii
                    Pil_recv_j(Pil_recvw_adr(kk)+recvw_len(kk))=jj
                endif
             enddo       
         endif

! check to send to who
         if (imx.ge.l_i0.and.imx.le.l_i0+l_ni-1 .and. &
             imy.ge.l_j0.and.imy.le.l_j0+l_nj-1      ) then
             do kk=1,Pil_sendmaxproc
                ki=Pil_sendproc(kk)
                if (i  .ge.Ptopo_gindx(1,ki).and.i  .le.Ptopo_gindx(2,ki).and. &
                    j  .ge.Ptopo_gindx(3,ki).and.j  .le.Ptopo_gindx(4,ki))then
                    sendw_len(kk)=sendw_len(kk)+1
                    Pil_send_imx(Pil_sendw_adr(kk)+sendw_len(kk))=imx-l_i0+1
                    Pil_send_imy(Pil_sendw_adr(kk)+sendw_len(kk))=imy-l_j0+1
                    Pil_send_xxr(Pil_sendw_adr(kk)+sendw_len(kk))=x_a
                    Pil_send_yyr(Pil_sendw_adr(kk)+sendw_len(kk))=y_a
                    Pil_send_s1(Pil_sendw_adr(kk)+sendw_len(kk))=s(1,1)
                    Pil_send_s2(Pil_sendw_adr(kk)+sendw_len(kk))=s(1,2)
                    Pil_send_s3(Pil_sendw_adr(kk)+sendw_len(kk))=s(2,1)
                    Pil_send_s4(Pil_sendw_adr(kk)+sendw_len(kk))=s(2,2)
                endif
             enddo       
         endif
      enddo   
      enddo   
!
!
! East section
      do j=1, G_nj
      do i=G_ni-Glb_pil_e+1,G_ni
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx,imy,x_a,y_a, &
                          G_xg_8(1),G_yg_8(1),h1,h2,1,1)
         imx = min(max(imx-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy = min(max(imy-1,glb_pil_s+1),G_nj-glb_pil_n-3)
! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Pil_recvmaxproc
                ki=Pil_recvproc(kk)
                if (imx.ge.Ptopo_gindx(1,ki).and.imx.le.Ptopo_gindx(2,ki).and. &
                    imy.ge.Ptopo_gindx(3,ki).and.imy.le.Ptopo_gindx(4,ki))then
                    recve_len(kk)=recve_len(kk)+1
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil_recv_i(Pil_recve_adr(kk)+recve_len(kk))=ii
                    Pil_recv_j(Pil_recve_adr(kk)+recve_len(kk))=jj
                endif
             enddo       
         endif

! check to send to who
         if (imx.ge.l_i0.and.imx.le.l_i0+l_ni-1 .and. &
             imy.ge.l_j0.and.imy.le.l_j0+l_nj-1      ) then
             do kk=1,Pil_sendmaxproc
                ki=Pil_sendproc(kk)
                if (i  .ge.Ptopo_gindx(1,ki).and.i  .le.Ptopo_gindx(2,ki).and. &
                    j  .ge.Ptopo_gindx(3,ki).and.j  .le.Ptopo_gindx(4,ki))then
                    sende_len(kk)=sende_len(kk)+1
                    Pil_send_imx(Pil_sende_adr(kk)+sende_len(kk))=imx-l_i0+1
                    Pil_send_imy(Pil_sende_adr(kk)+sende_len(kk))=imy-l_j0+1
                    Pil_send_xxr(Pil_sende_adr(kk)+sende_len(kk))=x_a
                    Pil_send_yyr(Pil_sende_adr(kk)+sende_len(kk))=y_a
                    Pil_send_s1(Pil_sende_adr(kk)+sende_len(kk))=s(1,1)
                    Pil_send_s2(Pil_sende_adr(kk)+sende_len(kk))=s(1,2)
                    Pil_send_s3(Pil_sende_adr(kk)+sende_len(kk))=s(2,1)
                    Pil_send_s4(Pil_sende_adr(kk)+sende_len(kk))=s(2,2)
                endif
             enddo       
         endif
      enddo   
      enddo   
!
! South section
      do j=1,Glb_pil_s
      do i=1+Glb_pil_w,G_ni-Glb_pil_e

         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx,imy,x_a,y_a, &
                          G_xg_8(1),G_yg_8(1),h1,h2,1,1)
         imx = min(max(imx-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy = min(max(imy-1,glb_pil_s+1),G_nj-glb_pil_n-3)
! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Pil_recvmaxproc
                ki=Pil_recvproc(kk)
                if (imx.ge.Ptopo_gindx(1,ki).and.imx.le.Ptopo_gindx(2,ki).and. &
                    imy.ge.Ptopo_gindx(3,ki).and.imy.le.Ptopo_gindx(4,ki))then
                    recvs_len(kk)=recvs_len(kk)+1
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil_recv_i(Pil_recvs_adr(kk)+recvs_len(kk))=ii
                    Pil_recv_j(Pil_recvs_adr(kk)+recvs_len(kk))=jj
                endif
             enddo       
         endif

! check to send to who
         if (imx.ge.l_i0.and.imx.le.l_i0+l_ni-1 .and. &
             imy.ge.l_j0.and.imy.le.l_j0+l_nj-1      ) then
             do kk=1,Pil_sendmaxproc
                ki=Pil_sendproc(kk)
                if (i  .ge.Ptopo_gindx(1,ki).and.i  .le.Ptopo_gindx(2,ki).and. &
                    j  .ge.Ptopo_gindx(3,ki).and.j  .le.Ptopo_gindx(4,ki))then
                    sends_len(kk)=sends_len(kk)+1
                    Pil_send_imx(Pil_sends_adr(kk)+sends_len(kk))=imx-l_i0+1
                    Pil_send_imy(Pil_sends_adr(kk)+sends_len(kk))=imy-l_j0+1
                    Pil_send_xxr(Pil_sends_adr(kk)+sends_len(kk))=x_a
                    Pil_send_yyr(Pil_sends_adr(kk)+sends_len(kk))=y_a
                    Pil_send_s1(Pil_sends_adr(kk)+sends_len(kk))=s(1,1)
                    Pil_send_s2(Pil_sends_adr(kk)+sends_len(kk))=s(1,2)
                    Pil_send_s3(Pil_sends_adr(kk)+sends_len(kk))=s(2,1)
                    Pil_send_s4(Pil_sends_adr(kk)+sends_len(kk))=s(2,2)
                endif
             enddo       
         endif
      enddo   
      enddo   
!
! North section
      do j=G_nj-Glb_pil_n+1,G_nj
      do i=1+Glb_pil_w,G_ni-Glb_pil_e

         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx,imy,x_a,y_a, &
                          G_xg_8(1),G_yg_8(1),h1,h2,1,1)
         imx = min(max(imx-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy = min(max(imy-1,glb_pil_s+1),G_nj-glb_pil_n-3)

! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Pil_recvmaxproc
                ki=Pil_recvproc(kk)
                if (imx.ge.Ptopo_gindx(1,ki).and.imx.le.Ptopo_gindx(2,ki).and. &
                    imy.ge.Ptopo_gindx(3,ki).and.imy.le.Ptopo_gindx(4,ki))then
                    recvn_len(kk)=recvn_len(kk)+1
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil_recv_i(Pil_recvn_adr(kk)+recvn_len(kk))=ii
                    Pil_recv_j(Pil_recvn_adr(kk)+recvn_len(kk))=jj
                endif
             enddo       
         endif

! check to send to who
         if (imx.ge.l_i0.and.imx.le.l_i0+l_ni-1 .and. &
             imy.ge.l_j0.and.imy.le.l_j0+l_nj-1      ) then
             do kk=1,Pil_sendmaxproc
                ki=Pil_sendproc(kk)
                if (i  .ge.Ptopo_gindx(1,ki).and.i  .le.Ptopo_gindx(2,ki).and. &
                    j  .ge.Ptopo_gindx(3,ki).and.j  .le.Ptopo_gindx(4,ki))then
                    sendn_len(kk)=sendn_len(kk)+1
                    Pil_send_imx(Pil_sendn_adr(kk)+sendn_len(kk))=imx-l_i0+1
                    Pil_send_imy(Pil_sendn_adr(kk)+sendn_len(kk))=imy-l_j0+1
                    Pil_send_xxr(Pil_sendn_adr(kk)+sendn_len(kk))=x_a
                    Pil_send_yyr(Pil_sendn_adr(kk)+sendn_len(kk))=y_a
                    Pil_send_s1(Pil_sendn_adr(kk)+sendn_len(kk))=s(1,1)
                    Pil_send_s2(Pil_sendn_adr(kk)+sendn_len(kk))=s(1,2)
                    Pil_send_s3(Pil_sendn_adr(kk)+sendn_len(kk))=s(2,1)
                    Pil_send_s4(Pil_sendn_adr(kk)+sendn_len(kk))=s(2,2)
                endif
             enddo       
         endif
      enddo   
      enddo   
!Check receive lengths from each processor
!     do ki=1,Pil_recvmaxproc
!        kk=Pil_recvproc(ki)
!        if (Ptopo_couleur.eq.0) then
!            kkproc = kk+Ptopo_numproc-1
!        else
!            kkproc = kk -1
!        endif
!    write(6,1000) 'Pil_recvw_len',kkproc,Pil_recvw_len(kk),Pil_recvw_adr(kk)
!    write(6,1000) 'Pil_recve_len',kkproc,Pil_recve_len(kk),Pil_recve_adr(kk)
!    write(6,1000) 'Pil_recvs_len',kkproc,Pil_recvs_len(kk),Pil_recvs_adr(kk)
!    write(6,1000) 'Pil_recvn_len',kkproc,Pil_recvn_len(kk),Pil_recvn_adr(kk)
!   enddo
!Check send lengths to each processor

!     do ki=1,Pil_sendmaxproc
!        kk=Pil_sendproc(ki)
!        if (Ptopo_couleur.eq.0) then
!            kkproc = kk+Ptopo_numproc-1
!        else
!            kkproc = kk -1
!        endif
! write(6,1000) 'Pil_sendw_len',kkproc,Pil_sendw_len(kk),Pil_sendw_adr(kk)
! write(6,1000) 'Pil_sende_len',kkproc,Pil_sende_len(kk),Pil_sende_adr(kk)
! write(6,1000) 'Pil_sends_len',kkproc,Pil_sends_len(kk),Pil_sends_adr(kk)
! write(6,1000) 'Pil_sendn_len',kkproc,Pil_sendn_len(kk),Pil_sendn_adr(kk)
!     enddo
      deallocate (recv_len,recvw_len,recve_len,recvs_len,recvn_len)
      deallocate (send_len,sendw_len,sende_len,sends_len,sendn_len)

 1000 format(a15,i3,'=',i5,'bytes, addr=',i5)
 1001 format(a15,i3,'=',i4,'bytes   i:', i3,' j:',i3)
       

!
      return
      end

