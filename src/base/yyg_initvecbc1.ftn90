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
!***s/r yyg_initvecbc1 - to initialize communication pattern for U field 
!


      Subroutine yyg_initvecbc1()
      implicit none
#include <arch_specific.hf>
!
!author
!     Abdessamad Qaddouri/V.Lee - October 2009
!
#include "ptopo.cdk"
#include "glb_ld.cdk"
#include "geomn.cdk"
#include "glb_pil.cdk"
#include "yyg_pilu.cdk"

      integer err,Ndim,i,j,k,kk,ii,jj,ki,ksend,krecv
      integer imx1,imx2
      integer imy1,imy2
      integer kkproc,adr
      integer, dimension (:), pointer :: recv_len,recvw_len,recve_len,recvs_len,recvn_len
      integer, dimension (:), pointer :: send_len,sendw_len,sende_len,sends_len,sendn_len
      real*8  xx_8(G_niu,G_nj),yy_8(G_niu,G_nj)
      real*8  xgu_8(G_niu),ygv_8(G_nj-1)
      real*8  t,p,s(2,2),h1,h2
      real*8  x_d,y_d,x_a,y_a   
!
!     Get global xgu,ygv,xx,yy
       do i=1,G_niu
       xgu_8(i)=0.5D0 *(G_xg_8(i+1)+G_xg_8(i))
       enddo
       do j=1,G_nj-1
       ygv_8(j)= 0.5D0*(G_yg_8(j+1)+G_yg_8(j))
       enddo
!
      do j=1,G_nj
      do i=1,G_niu
         xx_8(i,j)=xgu_8(i)
         yy_8(i,j)=G_yg_8(j)
      enddo
      enddo
      h1=G_xg_8(2)-G_xg_8(1)
      h2=G_yg_8(2)-G_yg_8(1)
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
!
! WEST section
!
      do j=1, G_nj
      do i=1,glb_pil_w
!        U vector
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx1,imy1,x_a,y_a, &
                          xgu_8(1),G_yg_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          G_xg_8(1),ygv_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_nj-glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy2 = min(max(imy2-1,glb_pil_s+1),G_nj-glb_pil_n-3)
!

! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (max(imx1,imx2).ge.Ptopo_gindx(1,kk).and. &
                    max(imx1,imx2).le.Ptopo_gindx(2,kk).and. &
                    max(imy1,imy2).ge.Ptopo_gindx(3,kk).and. &
                    max(imy1,imy2).le.Ptopo_gindx(4,kk) ) then
                    recvw_len(kk)=recvw_len(kk)+1
                endif
             enddo       
              
         endif

! check to send to who
         if (max(imx1,imx2).ge.l_i0.and.         &
             max(imx1,imx2).le.l_i0+l_ni-1 .and. &
             max(imy1,imy2).ge.l_j0.and.         &
             max(imy1,imy2).le.l_j0+l_nj-1) then
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
      do i=G_niu-glb_pil_e+1,G_niu
!        U vector
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx1,imy1,x_a,y_a, &
                          xgu_8(1),G_yg_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          G_xg_8(1),ygv_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_nj-glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy2 = min(max(imy2-1,glb_pil_s+1),G_nj-glb_pil_n-3)
!
! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (max(imx1,imx2).ge.Ptopo_gindx(1,kk).and. &
                    max(imx1,imx2).le.Ptopo_gindx(2,kk).and. &
                    max(imy1,imy2).ge.Ptopo_gindx(3,kk).and. &
                    max(imy1,imy2).le.Ptopo_gindx(4,kk) ) then
                    recve_len(kk)=recve_len(kk)+1
                endif
             enddo       
         endif

! check to send to who
         if (max(imx1,imx2).ge.l_i0.and.         &
             max(imx1,imx2).le.l_i0+l_ni-1 .and. &
             max(imy1,imy2).ge.l_j0.and.         &
             max(imy1,imy2).le.l_j0+l_nj-1) then
             do kk=1,Ptopo_numproc
                if (i  .ge.Ptopo_gindx(1,kk).and.i .le.Ptopo_gindx(2,kk).and. &
                    j  .ge.Ptopo_gindx(3,kk).and.j .le.Ptopo_gindx(4,kk))then
                    sende_len(kk)=sende_len(kk)+1
                endif
             enddo       
         endif
      enddo   
      enddo   
!
! South section
      do j=1,glb_pil_s
      do i=1+glb_pil_w,G_niu-glb_pil_e
!        U vector
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx1,imy1,x_a,y_a, &
                          xgu_8(1),G_yg_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          G_xg_8(1),ygv_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_nj-glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy2 = min(max(imy2-1,glb_pil_s+1),G_nj-glb_pil_n-3)
!
! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (max(imx1,imx2).ge.Ptopo_gindx(1,kk).and. &
                    max(imx1,imx2).le.Ptopo_gindx(2,kk).and. &
                    max(imy1,imy2).ge.Ptopo_gindx(3,kk).and. &
                    max(imy1,imy2).le.Ptopo_gindx(4,kk) ) then
                    recvs_len(kk)=recvs_len(kk)+1
                endif
             enddo       
         endif

! check to send to who
         if (max(imx1,imx2).ge.l_i0.and.         &
             max(imx1,imx2).le.l_i0+l_ni-1 .and. &
             max(imy1,imy2).ge.l_j0.and.         &
             max(imy1,imy2).le.l_j0+l_nj-1) then
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
      do j=G_nj-glb_pil_n+1,G_nj
      do i=1+glb_pil_w,G_niu-glb_pil_e
!        U vector
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx1,imy1,x_a,y_a, &
                          xgu_8(1),G_yg_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          G_xg_8(1),ygv_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_nj-glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy2 = min(max(imy2-1,glb_pil_s+1),G_nj-glb_pil_n-3)
!

! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (max(imx1,imx2).ge.Ptopo_gindx(1,kk).and. &
                    max(imx1,imx2).le.Ptopo_gindx(2,kk).and. &
                    max(imy1,imy2).ge.Ptopo_gindx(3,kk).and. &
                    max(imy1,imy2).le.Ptopo_gindx(4,kk) ) then
                    recvn_len(kk)=recvn_len(kk)+1
                endif
             enddo       
         endif

! check to send to who
         if (max(imx1,imx2).ge.l_i0.and.         &
             max(imx1,imx2).le.l_i0+l_ni-1 .and. &
             max(imy1,imy2).ge.l_j0.and.         &
             max(imy1,imy2).le.l_j0+l_nj-1) then
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
     Pil_usend_all=0
     Pil_urecv_all=0
     Pil_usendmaxproc=0
     Pil_urecvmaxproc=0

     do kk=1,Ptopo_numproc
        send_len(kk)=sendw_len(kk)+sende_len(kk) + sends_len(kk)+sendn_len(kk)
        recv_len(kk)=recvw_len(kk)+recve_len(kk) + recvs_len(kk)+recvn_len(kk)
        Pil_usend_all=send_len(kk)+Pil_usend_all
        Pil_urecv_all=recv_len(kk)+Pil_urecv_all

        if (send_len(kk).gt.0) Pil_usendmaxproc=Pil_usendmaxproc+1
        if (recv_len(kk).gt.0) Pil_urecvmaxproc=Pil_urecvmaxproc+1
     enddo

!
!     print *,'Allocate common vectors'
      allocate (Pil_urecvproc(Pil_urecvmaxproc))
      allocate (Pil_urecv_len(Pil_urecvmaxproc))
      allocate (Pil_urecvw_len(Pil_urecvmaxproc))
      allocate (Pil_urecve_len(Pil_urecvmaxproc))
      allocate (Pil_urecvs_len(Pil_urecvmaxproc))
      allocate (Pil_urecvn_len(Pil_urecvmaxproc))
      allocate (Pil_urecvw_adr(Pil_urecvmaxproc))
      allocate (Pil_urecve_adr(Pil_urecvmaxproc))
      allocate (Pil_urecvs_adr(Pil_urecvmaxproc))
      allocate (Pil_urecvn_adr(Pil_urecvmaxproc))

      allocate (Pil_usendproc(Pil_usendmaxproc))
      allocate (Pil_usend_len(Pil_usendmaxproc))
      allocate (Pil_usendw_len(Pil_usendmaxproc))
      allocate (Pil_usende_len(Pil_usendmaxproc))
      allocate (Pil_usends_len(Pil_usendmaxproc))
      allocate (Pil_usendn_len(Pil_usendmaxproc))
      allocate (Pil_usendw_adr(Pil_usendmaxproc))
      allocate (Pil_usende_adr(Pil_usendmaxproc))
      allocate (Pil_usends_adr(Pil_usendmaxproc))
      allocate (Pil_usendn_adr(Pil_usendmaxproc))
      Pil_urecvw_len(:) = 0
      Pil_urecve_len(:) = 0
      Pil_urecvs_len(:) = 0
      Pil_urecvn_len(:) = 0
      Pil_usendw_len(:) = 0
      Pil_usende_len(:) = 0
      Pil_usends_len(:) = 0
      Pil_usendn_len(:) = 0
      Pil_urecvw_adr(:) = 0
      Pil_urecve_adr(:) = 0
      Pil_urecvs_adr(:) = 0
      Pil_urecvn_adr(:) = 0
      Pil_usendw_adr(:) = 0
      Pil_usende_adr(:) = 0
      Pil_usends_adr(:) = 0
      Pil_usendn_adr(:) = 0

!    print*,'Pil_usendmaxproc=',Pil_usendmaxproc,'recvmaxproc=',Pil_urecvmaxproc

     ksend=0
     krecv=0
     Pil_usend_all=0
     Pil_urecv_all=0
!
! Fill the lengths and addresses for selected processors to communicate
!
     do kk=1,Ptopo_numproc
        if (send_len(kk).gt.0) then
            ksend=ksend+1
            Pil_usendproc(ksend)=kk
            Pil_usend_len(ksend)=send_len(kk)
            Pil_usendw_len(ksend)=sendw_len(kk)
            Pil_usende_len(ksend)=sende_len(kk)
            Pil_usends_len(ksend)=sends_len(kk)
            Pil_usendn_len(ksend)=sendn_len(kk)

            Pil_usendw_adr(ksend)= Pil_usend_all
            Pil_usend_all= Pil_usend_all + Pil_usend_len(ksend)
            Pil_usende_adr(ksend)= Pil_usendw_adr(ksend)+Pil_usendw_len(ksend)
            Pil_usends_adr(ksend)= Pil_usende_adr(ksend)+Pil_usende_len(ksend)
            Pil_usendn_adr(ksend)= Pil_usends_adr(ksend)+Pil_usends_len(ksend)
        endif
        if (recv_len(kk).gt.0) then
            krecv=krecv+1
            Pil_urecvproc(krecv)=kk
            Pil_urecv_len(krecv)=recv_len(kk)
            Pil_urecvw_len(krecv)=recvw_len(kk)
            Pil_urecve_len(krecv)=recve_len(kk)
            Pil_urecvs_len(krecv)=recvs_len(kk)
            Pil_urecvn_len(krecv)=recvn_len(kk)

            Pil_urecvw_adr(krecv)= Pil_urecv_all
            Pil_urecv_all= Pil_urecv_all + Pil_urecv_len(krecv)
            Pil_urecve_adr(krecv)= Pil_urecvw_adr(krecv)+Pil_urecvw_len(krecv)
            Pil_urecvs_adr(krecv)= Pil_urecve_adr(krecv)+Pil_urecve_len(krecv)
            Pil_urecvn_adr(krecv)= Pil_urecvs_adr(krecv)+Pil_urecvs_len(krecv)
        endif

     enddo
!    print *,'krecv=',krecv,'Pil_urecvmaxproc=',Pil_urecvmaxproc
!    print *,'ksend=',ksend,'Pil_usendmaxproc=',Pil_usendmaxproc

!     print *,'Summary of comm procs'
!     do kk=1,Pil_urecvmaxproc
!       print *,'From proc:',Pil_urecvproc(kk),'Pil_urecv_len',Pil_urecvw_len(kk),Pil_urecve_len(kk),Pil_urecvs_len(kk),Pil_urecvn_len(kk),'adr',Pil_urecvw_adr(kk),Pil_urecve_adr(kk),Pil_urecvs_adr(kk),Pil_urecvn_adr(kk)
!     enddo
!     do kk=1,Pil_usendmaxproc
!       print *,'To proc:',Pil_usendproc(kk),'Pil_usend_len',Pil_usendw_len(kk),Pil_usende_len(kk),Pil_usends_len(kk),Pil_usendn_len(kk),'adr',Pil_usendw_adr(kk),Pil_usende_adr(kk),Pil_usends_adr(kk),Pil_usendn_adr(kk)
!     enddo
!     print *,'Pil_urecv_all=',Pil_urecv_all, 'Pil_usend_all=',Pil_usend_all

!
! Now allocate the vectors needed for sending and receiving each processor
!
      if (Pil_urecv_all.gt.0) then
          allocate (Pil_urecv_i(Pil_urecv_all))
          allocate (Pil_urecv_j(Pil_urecv_all))
          Pil_urecv_i(:) = 0
          Pil_urecv_j(:) = 0
      endif

      if (Pil_usend_all.gt.0) then
          allocate (Pil_usend_imx1(Pil_usend_all))
          allocate (Pil_usend_imy1(Pil_usend_all))
          allocate (Pil_usend_imx2(Pil_usend_all))
          allocate (Pil_usend_imy2(Pil_usend_all))
          allocate (Pil_usend_xxr(Pil_usend_all))
          allocate (Pil_usend_yyr(Pil_usend_all))
          allocate (Pil_usend_s1(Pil_usend_all))
          allocate (Pil_usend_s2(Pil_usend_all))
          Pil_usend_imx1(:) = 0
          Pil_usend_imy1(:) = 0
          Pil_usend_imx2(:) = 0
          Pil_usend_imy2(:) = 0
          Pil_usend_xxr(:) = 0.0
          Pil_usend_yyr(:) = 0.0
          Pil_usend_s1(:) = 0.0
          Pil_usend_s2(:) = 0.0
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
!
      do j=1, G_nj
      do i=1,glb_pil_w
!        U vector
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx1,imy1,x_a,y_a, &
                          xgu_8(1),G_yg_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          G_xg_8(1),ygv_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_nj-glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy2 = min(max(imy2-1,glb_pil_s+1),G_nj-glb_pil_n-3)
!

! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Pil_urecvmaxproc
                ki=Pil_urecvproc(kk)
                if (max(imx1,imx2).ge.Ptopo_gindx(1,ki).and. &
                    max(imx1,imx2).le.Ptopo_gindx(2,ki).and. &
                    max(imy1,imy2).ge.Ptopo_gindx(3,ki).and. &
                    max(imy1,imy2).le.Ptopo_gindx(4,ki) ) then
                    recvw_len(kk)=recvw_len(kk)+1
                    adr=Pil_urecvw_adr(kk)+recvw_len(kk)
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil_urecv_i(adr)=ii
                    Pil_urecv_j(adr)=jj
                endif
             enddo       
         endif

! check to send to who
         if (max(imx1,imx2).ge.l_i0.and.         &
             max(imx1,imx2).le.l_i0+l_ni-1 .and. &
             max(imy1,imy2).ge.l_j0.and.         &
             max(imy1,imy2).le.l_j0+l_nj-1) then
             do kk=1,Pil_usendmaxproc
                ki=Pil_usendproc(kk)
                if (i  .ge.Ptopo_gindx(1,ki).and.i  .le.Ptopo_gindx(2,ki).and. &
                    j  .ge.Ptopo_gindx(3,ki).and.j  .le.Ptopo_gindx(4,ki))then
                    sendw_len(kk)=sendw_len(kk)+1
                    adr=Pil_usendw_adr(kk)+sendw_len(kk)
                    Pil_usend_imx1(adr)=imx1-l_i0+1
                    Pil_usend_imy1(adr)=imy1-l_j0+1
                    Pil_usend_imx2(adr)=imx2-l_i0+1
                    Pil_usend_imy2(adr)=imy2-l_j0+1
                    Pil_usend_xxr(adr)=x_a
                    Pil_usend_yyr(adr)=y_a
                    Pil_usend_s1(adr)=s(1,1)
                    Pil_usend_s2(adr)=s(1,2)
                endif
             enddo       
         endif
      enddo   
      enddo   

!
!
! East section
      do j=1, G_nj
      do i=G_niu-glb_pil_e+1,G_niu
!        U vector
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx1,imy1,x_a,y_a, &
                          xgu_8(1),G_yg_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          G_xg_8(1),ygv_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_nj-glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy2 = min(max(imy2-1,glb_pil_s+1),G_nj-glb_pil_n-3)
!
! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Pil_urecvmaxproc
                ki=Pil_urecvproc(kk)
                if (max(imx1,imx2).ge.Ptopo_gindx(1,ki).and. &
                    max(imx1,imx2).le.Ptopo_gindx(2,ki).and. &
                    max(imy1,imy2).ge.Ptopo_gindx(3,ki).and. &
                    max(imy1,imy2).le.Ptopo_gindx(4,ki) ) then
                    recve_len(kk)=recve_len(kk)+1
                    adr=Pil_urecve_adr(kk)+recve_len(kk)
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil_urecv_i(adr)=ii
                    Pil_urecv_j(adr)=jj
                endif
             enddo       
         endif

! check to send to who
         if (max(imx1,imx2).ge.l_i0.and.         &
             max(imx1,imx2).le.l_i0+l_ni-1 .and. &
             max(imy1,imy2).ge.l_j0.and.         &
             max(imy1,imy2).le.l_j0+l_nj-1) then
             do kk=1,Pil_usendmaxproc
                ki=Pil_usendproc(kk)
                if (i  .ge.Ptopo_gindx(1,ki).and.i .le.Ptopo_gindx(2,ki).and. &
                    j  .ge.Ptopo_gindx(3,ki).and.j .le.Ptopo_gindx(4,ki))then
                    sende_len(kk)=sende_len(kk)+1
                    adr=Pil_usende_adr(kk)+sende_len(kk)
                    Pil_usend_imx1(adr)=imx1-l_i0+1
                    Pil_usend_imy1(adr)=imy1-l_j0+1
                    Pil_usend_imx2(adr)=imx2-l_i0+1
                    Pil_usend_imy2(adr)=imy2-l_j0+1
                    Pil_usend_xxr(adr)=x_a
                    Pil_usend_yyr(adr)=y_a
                    Pil_usend_s1(adr)=s(1,1)
                    Pil_usend_s2(adr)=s(1,2)
                endif
             enddo       
         endif
      enddo   
      enddo   
!
! South section
      do j=1,glb_pil_s
      do i=1+glb_pil_w,G_niu-glb_pil_e
!        U vector
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx1,imy1,x_a,y_a, &
                          xgu_8(1),G_yg_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          G_xg_8(1),ygv_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_nj-glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy2 = min(max(imy2-1,glb_pil_s+1),G_nj-glb_pil_n-3)
!
! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Pil_urecvmaxproc
                ki=Pil_urecvproc(kk)
                if (max(imx1,imx2).ge.Ptopo_gindx(1,ki).and. &
                    max(imx1,imx2).le.Ptopo_gindx(2,ki).and. &
                    max(imy1,imy2).ge.Ptopo_gindx(3,ki).and. &
                    max(imy1,imy2).le.Ptopo_gindx(4,ki) ) then
                    recvs_len(kk)=recvs_len(kk)+1
                    adr=Pil_urecvs_adr(kk)+recvs_len(kk)
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil_urecv_i(adr)=ii
                    Pil_urecv_j(adr)=jj
                endif
             enddo       
         endif

! check to send to who
         if (max(imx1,imx2).ge.l_i0.and.         &
             max(imx1,imx2).le.l_i0+l_ni-1 .and. &
             max(imy1,imy2).ge.l_j0.and.         &
             max(imy1,imy2).le.l_j0+l_nj-1) then
             do kk=1,Pil_usendmaxproc
                ki=Pil_usendproc(kk)
                if (i  .ge.Ptopo_gindx(1,ki).and.i  .le.Ptopo_gindx(2,ki).and. &
                    j  .ge.Ptopo_gindx(3,ki).and.j  .le.Ptopo_gindx(4,ki))then
                    sends_len(kk)=sends_len(kk)+1
                    adr=Pil_usends_adr(kk)+sends_len(kk)
                    Pil_usend_imx1(adr)=imx1-l_i0+1
                    Pil_usend_imy1(adr)=imy1-l_j0+1
                    Pil_usend_imx2(adr)=imx2-l_i0+1
                    Pil_usend_imy2(adr)=imy2-l_j0+1
                    Pil_usend_xxr(adr)=x_a
                    Pil_usend_yyr(adr)=y_a
                    Pil_usend_s1(adr)=s(1,1)
                    Pil_usend_s2(adr)=s(1,2)
                endif
             enddo       
         endif
      enddo   
      enddo   
!
! North section
      do j=G_nj-glb_pil_n+1,G_nj
      do i=1+glb_pil_w,G_niu-glb_pil_e
!        U vector
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx1,imy1,x_a,y_a, &
                          xgu_8(1),G_yg_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          G_xg_8(1),ygv_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_nj-glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy2 = min(max(imy2-1,glb_pil_s+1),G_nj-glb_pil_n-3)
!

! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Pil_urecvmaxproc
                ki=Pil_urecvproc(kk)
                if (max(imx1,imx2).ge.Ptopo_gindx(1,ki).and. &
                    max(imx1,imx2).le.Ptopo_gindx(2,ki).and. &
                    max(imy1,imy2).ge.Ptopo_gindx(3,ki).and. &
                    max(imy1,imy2).le.Ptopo_gindx(4,ki) ) then
                    recvn_len(kk)=recvn_len(kk)+1
                    adr=Pil_urecvn_adr(kk)+recvn_len(kk)
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil_urecv_i(adr)=ii
                    Pil_urecv_j(adr)=jj
                endif
             enddo       
         endif

! check to send to who
         if (max(imx1,imx2).ge.l_i0.and.         &
             max(imx1,imx2).le.l_i0+l_ni-1 .and. &
             max(imy1,imy2).ge.l_j0.and.         &
             max(imy1,imy2).le.l_j0+l_nj-1) then
             do kk=1,Pil_usendmaxproc
                ki=Pil_usendproc(kk)
                if (i  .ge.Ptopo_gindx(1,ki).and.i  .le.Ptopo_gindx(2,ki).and. &
                    j  .ge.Ptopo_gindx(3,ki).and.j  .le.Ptopo_gindx(4,ki))then
                    sendn_len(kk)=sendn_len(kk)+1
                    adr=Pil_usendn_adr(kk)+sendn_len(kk)
                    Pil_usend_imx1(adr)=imx1-l_i0+1
                    Pil_usend_imy1(adr)=imy1-l_j0+1
                    Pil_usend_imx2(adr)=imx2-l_i0+1
                    Pil_usend_imy2(adr)=imy2-l_j0+1
                    Pil_usend_xxr(adr)=x_a
                    Pil_usend_yyr(adr)=y_a
                    Pil_usend_s1(adr)=s(1,1)
                    Pil_usend_s2(adr)=s(1,2)
                endif
             enddo       
         endif
      enddo   
      enddo   
!Check receive lengths from each processor
!     do ki=1,Pil_urecvmaxproc
!        kk=Pil_urecvproc(ki)
!        if (Ptopo_couleur.eq.0) then
!            kkproc = kk+Ptopo_numproc-1
!        else
!            kkproc = kk -1
!        endif
!    write(6,1000) 'Pil_urecvw_len',kkproc,Pil_urecvw_len(kk),Pil_urecvw_adr(kk)
!    write(6,1000) 'Pil_urecve_len',kkproc,Pil_urecve_len(kk),Pil_urecve_adr(kk)
!    write(6,1000) 'Pil_urecvs_len',kkproc,Pil_urecvs_len(kk),Pil_urecvs_adr(kk)
!    write(6,1000) 'Pil_urecvn_len',kkproc,Pil_urecvn_len(kk),Pil_urecvn_adr(kk)
!   enddo

!Check send lengths to each processor

!     do ki=1,Pil_usendmaxproc
!        kk=Pil_usendproc(ki)
!        if (Ptopo_couleur.eq.0) then
!            kkproc = kk+Ptopo_numproc-1
!        else
!            kkproc = kk -1
!        endif
! write(6,1000) 'Pil_usendw_len',kkproc,Pil_usendw_len(kk),Pil_usendw_adr(kk)
! write(6,1000) 'Pil_usende_len',kkproc,Pil_usende_len(kk),Pil_usende_adr(kk)
! write(6,1000) 'Pil_usends_len',kkproc,Pil_usends_len(kk),Pil_usends_adr(kk)
! write(6,1000) 'Pil_usendn_len',kkproc,Pil_usendn_len(kk),Pil_usendn_adr(kk)
!     enddo
      deallocate (recv_len,recvw_len,recve_len,recvs_len,recvn_len)
      deallocate (send_len,sendw_len,sende_len,sends_len,sendn_len)

 1000 format(a15,i3,'=',i5,'bytes, addr=',i5)
 1001 format(a15,i3,'=',i4,'bytes   i:', i3,' j:',i3)
       
!
      return
      end

