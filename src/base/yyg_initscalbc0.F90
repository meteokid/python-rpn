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
!***s/r yyg_initscalbc0 - to initialize communication pattern for nearest
!                            interpolation of scalar fields
!


      Subroutine yyg_initscalbc0()
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
#include "yyg_pil0.cdk"

      integer err,Ndim,i,j,k,imx,imy,kk,ii,jj,ki,ksend,krecv
      integer kkproc, minx,maxx,miny,maxy
      integer, dimension (:), pointer :: recv_len,send_len
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
      allocate (send_len (Ptopo_numproc))
      recv_len (:)=0
      send_len (:)=0
!
! FIRST PASS is to find the number of processor to tag for
! communication and the number of items to send and receive for each
! processor before allocating the vectors
!
! WEST section

      minx = lbound(G_xg_8,1)
      maxx = ubound(G_xg_8,1)
      miny = lbound(G_yg_8,1)
      maxy = ubound(G_yg_8,1)
      do j=1, G_nj
      do i=1,Glb_pil_w
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise1(imx,imy,x_a,y_a, &
                          G_xg_8,G_yg_8,h1,h2,1,1, &
                          minx,maxx,miny,maxy)
         imx = min(max(imx,glb_pil_w+1),G_ni-glb_pil_e-1)
         imy = min(max(imy,glb_pil_s+1),G_nj-glb_pil_n-1)
! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (imx.ge.Ptopo_gindx(1,kk).and.imx.le.Ptopo_gindx(2,kk).and. &
                    imy.ge.Ptopo_gindx(3,kk).and.imy.le.Ptopo_gindx(4,kk)) then
                    recv_len(kk)=recv_len(kk)+1
                endif
             enddo       
         endif

! check to send to who
         if (imx.ge.l_i0.and.imx.le.l_i0+l_ni-1 .and. &
             imy.ge.l_j0.and.imy.le.l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (i  .ge.Ptopo_gindx(1,kk).and.i  .le.Ptopo_gindx(2,kk).and. &
                    j  .ge.Ptopo_gindx(3,kk).and.j  .le.Ptopo_gindx(4,kk))then
                    send_len(kk)=send_len(kk)+1
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
         call localise1(imx,imy,x_a,y_a, &
                          G_xg_8,G_yg_8,h1,h2,1,1, &
                          minx,maxx,miny,maxy)
         imx = min(max(imx,glb_pil_w+1),G_ni-glb_pil_e-1)
         imy = min(max(imy,glb_pil_s+1),G_nj-glb_pil_n-1)
! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (imx.ge.Ptopo_gindx(1,kk).and.imx.le.Ptopo_gindx(2,kk).and. &
                    imy.ge.Ptopo_gindx(3,kk).and.imy.le.Ptopo_gindx(4,kk))then
                    recv_len(kk)=recv_len(kk)+1
                endif
             enddo       
         endif

! check to send to who
         if (imx.ge.l_i0.and.imx.le.l_i0+l_ni-1 .and. &
             imy.ge.l_j0.and.imy.le.l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (i  .ge.Ptopo_gindx(1,kk).and.i  .le.Ptopo_gindx(2,kk).and. &
                    j  .ge.Ptopo_gindx(3,kk).and.j  .le.Ptopo_gindx(4,kk))then
                    send_len(kk)=send_len(kk)+1
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
         call localise1(imx,imy,x_a,y_a, &
                          G_xg_8,G_yg_8,h1,h2,1,1, &
                          minx,maxx,miny,maxy)
         imx = min(max(imx,glb_pil_w+1),G_ni-glb_pil_e-1)
         imy = min(max(imy,glb_pil_s+1),G_nj-glb_pil_n-1)
! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (imx.ge.Ptopo_gindx(1,kk).and.imx.le.Ptopo_gindx(2,kk).and. &
                    imy.ge.Ptopo_gindx(3,kk).and.imy.le.Ptopo_gindx(4,kk))then
                    recv_len(kk)=recv_len(kk)+1
                endif
             enddo       
         endif

! check to send to who
         if (imx.ge.l_i0.and.imx.le.l_i0+l_ni-1 .and. &
             imy.ge.l_j0.and.imy.le.l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (i  .ge.Ptopo_gindx(1,kk).and.i  .le.Ptopo_gindx(2,kk).and. &
                    j  .ge.Ptopo_gindx(3,kk).and.j  .le.Ptopo_gindx(4,kk))then
                    send_len(kk)=send_len(kk)+1
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
         call localise1(imx,imy,x_a,y_a, &
                          G_xg_8,G_yg_8,h1,h2,1,1, &
                          minx,maxx,miny,maxy)
         imx = min(max(imx,glb_pil_w+1),G_ni-glb_pil_e-1)
         imy = min(max(imy,glb_pil_s+1),G_nj-glb_pil_n-1)

! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (imx.ge.Ptopo_gindx(1,kk).and.imx.le.Ptopo_gindx(2,kk).and. &
                    imy.ge.Ptopo_gindx(3,kk).and.imy.le.Ptopo_gindx(4,kk))then
                    recv_len(kk)=recv_len(kk)+1
                endif
             enddo       
         endif

! check to send to who
         if (imx.ge.l_i0.and.imx.le.l_i0+l_ni-1 .and. &
             imy.ge.l_j0.and.imy.le.l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (i  .ge.Ptopo_gindx(1,kk).and.i  .le.Ptopo_gindx(2,kk).and. &
                    j  .ge.Ptopo_gindx(3,kk).and.j  .le.Ptopo_gindx(4,kk))then
                    send_len(kk)=send_len(kk)+1
                endif
             enddo       
         endif
      enddo   
      enddo   
!
! Obtain sum of elements to send and receive for each processor
! and the total memory needed to store and receive for each processor
!
     Pil0_send_all=0 
     Pil0_recv_all=0 
     Pil0_sendmaxproc=0
     Pil0_recvmaxproc=0
     
     do kk=1,Ptopo_numproc
        Pil0_send_all=send_len(kk)+Pil0_send_all
        Pil0_recv_all=recv_len(kk)+Pil0_recv_all

        if (send_len(kk).gt.0) Pil0_sendmaxproc=Pil0_sendmaxproc+1
        if (recv_len(kk).gt.0) Pil0_recvmaxproc=Pil0_recvmaxproc+1
     enddo
!
!     print *,'Allocate common vectors'
      allocate (Pil0_recvproc(Pil0_recvmaxproc))
      allocate (Pil0_recv_len(Pil0_recvmaxproc))
      allocate (Pil0_recv_adr(Pil0_recvmaxproc))

      allocate (Pil0_sendproc(Pil0_sendmaxproc))
      allocate (Pil0_send_len(Pil0_sendmaxproc))
      allocate (Pil0_send_adr(Pil0_sendmaxproc))
      Pil0_recv_len(:) = 0
      Pil0_send_len(:) = 0
      Pil0_recv_adr(:) = 0
      Pil0_send_adr(:) = 0

!    print*,'Pil0_sendmaxproc=',Pil0_sendmaxproc,'recvmaxproc=',Pil0_recvmaxproc
       
     ksend=0
     krecv=0
     Pil0_send_all=0
     Pil0_recv_all=0
!
! Fill the lengths and addresses for selected processors to communicate
!
     do kk=1,Ptopo_numproc
        if (send_len(kk).gt.0) then
            ksend=ksend+1
            Pil0_sendproc(ksend)=kk
            Pil0_send_len(ksend)=send_len(kk)

            Pil0_send_adr(ksend)= Pil0_send_all
            Pil0_send_all= Pil0_send_all + Pil0_send_len(ksend)
        endif
        if (recv_len(kk).gt.0) then
            krecv=krecv+1
            Pil0_recvproc(krecv)=kk
            Pil0_recv_len(krecv)=recv_len(kk)

            Pil0_recv_adr(krecv)= Pil0_recv_all
            Pil0_recv_all= Pil0_recv_all + Pil0_recv_len(krecv)
        endif
            
     enddo
!    print *,'krecv=',krecv,'Pil0_recvmaxproc=',Pil0_recvmaxproc
!    print *,'ksend=',ksend,'Pil0_sendmaxproc=',Pil0_sendmaxproc

!     print *,'Summary of comm procs'
!     do kk=1,Pil0_recvmaxproc
!       print *,'From proc:',Pil0_recvproc(kk),'Pil0_recv_len=',Pil0_recv_len(kk),'Pil0_recv_adr=',Pil0_recv_adr(kk)
!     enddo
!     do kk=1,Pil0_sendmaxproc
!       print *,'To proc:',Pil0_sendproc(kk),'Pil0_send_len=',Pil0_send_len(kk),'Pil0_send_adr=',Pil0_send_adr(kk)
!     enddo

!
! Now allocate the vectors needed for sending and receiving each processor
!
      if (Pil0_recv_all.gt.0) then
          allocate (Pil0_recv_i(Pil0_recv_all))
          allocate (Pil0_recv_j(Pil0_recv_all))
          Pil0_recv_i(:) = 0
          Pil0_recv_j(:) = 0
      endif

      if (Pil0_send_all.gt.0) then
          allocate (Pil0_send_imx(Pil0_send_all))
          allocate (Pil0_send_imy(Pil0_send_all))
          allocate (Pil0_send_xxr(Pil0_send_all))
          allocate (Pil0_send_yyr(Pil0_send_all))
          allocate (Pil0_send_s1(Pil0_send_all))
          allocate (Pil0_send_s2(Pil0_send_all))
          allocate (Pil0_send_s3(Pil0_send_all))
          allocate (Pil0_send_s4(Pil0_send_all))
          Pil0_send_imx(:) = 0
          Pil0_send_imy(:) = 0
          Pil0_send_xxr(:) = 0.0
          Pil0_send_yyr(:) = 0.0
          Pil0_send_s1(:) = 0.0
          Pil0_send_s2(:) = 0.0
          Pil0_send_s3(:) = 0.0
          Pil0_send_s4(:) = 0.0
      endif
!

      recv_len(:)=0
      send_len(:)=0
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
         call localise1(imx,imy,x_a,y_a, &
                          G_xg_8,G_yg_8,h1,h2,1,1, &
                          minx,maxx,miny,maxy)
         imx = min(max(imx,glb_pil_w+1),G_ni-glb_pil_e-1)
         imy = min(max(imy,glb_pil_s+1),G_nj-glb_pil_n-1)
! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Pil0_recvmaxproc
                ki=Pil0_recvproc(kk)
                if (imx.ge.Ptopo_gindx(1,ki).and.imx.le.Ptopo_gindx(2,ki).and. &
                    imy.ge.Ptopo_gindx(3,ki).and.imy.le.Ptopo_gindx(4,ki))then
                    recv_len(kk)=recv_len(kk)+1
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil0_recv_i(Pil0_recv_adr(kk)+recv_len(kk))=ii
                    Pil0_recv_j(Pil0_recv_adr(kk)+recv_len(kk))=jj
                endif
             enddo       
         endif

! check to send to who
         if (imx.ge.l_i0.and.imx.le.l_i0+l_ni-1 .and. &
             imy.ge.l_j0.and.imy.le.l_j0+l_nj-1      ) then
             do kk=1,Pil0_sendmaxproc
                ki=Pil0_sendproc(kk)
                if (i  .ge.Ptopo_gindx(1,ki).and.i  .le.Ptopo_gindx(2,ki).and. &
                    j  .ge.Ptopo_gindx(3,ki).and.j  .le.Ptopo_gindx(4,ki))then
                    send_len(kk)=send_len(kk)+1
                    Pil0_send_imx(Pil0_send_adr(kk)+send_len(kk))=imx-l_i0+1
                    Pil0_send_imy(Pil0_send_adr(kk)+send_len(kk))=imy-l_j0+1
                    Pil0_send_xxr(Pil0_send_adr(kk)+send_len(kk))=x_a
                    Pil0_send_yyr(Pil0_send_adr(kk)+send_len(kk))=y_a
                    Pil0_send_s1(Pil0_send_adr(kk)+send_len(kk))=s(1,1)
                    Pil0_send_s2(Pil0_send_adr(kk)+send_len(kk))=s(1,2)
                    Pil0_send_s3(Pil0_send_adr(kk)+send_len(kk))=s(2,1)
                    Pil0_send_s4(Pil0_send_adr(kk)+send_len(kk))=s(2,2)
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
         call localise1(imx,imy,x_a,y_a, &
                          G_xg_8,G_yg_8,h1,h2,1,1, &
                          minx,maxx,miny,maxy)
         imx = min(max(imx,glb_pil_w+1),G_ni-glb_pil_e-1)
         imy = min(max(imy,glb_pil_s+1),G_nj-glb_pil_n-1)
! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Pil0_recvmaxproc
                ki=Pil0_recvproc(kk)
                if (imx.ge.Ptopo_gindx(1,ki).and.imx.le.Ptopo_gindx(2,ki).and. &
                    imy.ge.Ptopo_gindx(3,ki).and.imy.le.Ptopo_gindx(4,ki))then
                    recv_len(kk)=recv_len(kk)+1
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil0_recv_i(Pil0_recv_adr(kk)+recv_len(kk))=ii
                    Pil0_recv_j(Pil0_recv_adr(kk)+recv_len(kk))=jj
                endif
             enddo       
         endif

! check to send to who
         if (imx.ge.l_i0.and.imx.le.l_i0+l_ni-1 .and. &
             imy.ge.l_j0.and.imy.le.l_j0+l_nj-1      ) then
             do kk=1,Pil0_sendmaxproc
                ki=Pil0_sendproc(kk)
                if (i  .ge.Ptopo_gindx(1,ki).and.i  .le.Ptopo_gindx(2,ki).and. &
                    j  .ge.Ptopo_gindx(3,ki).and.j  .le.Ptopo_gindx(4,ki))then
                    send_len(kk)=send_len(kk)+1
                    Pil0_send_imx(Pil0_send_adr(kk)+send_len(kk))=imx-l_i0+1
                    Pil0_send_imy(Pil0_send_adr(kk)+send_len(kk))=imy-l_j0+1
                    Pil0_send_xxr(Pil0_send_adr(kk)+send_len(kk))=x_a
                    Pil0_send_yyr(Pil0_send_adr(kk)+send_len(kk))=y_a
                    Pil0_send_s1(Pil0_send_adr(kk)+send_len(kk))=s(1,1)
                    Pil0_send_s2(Pil0_send_adr(kk)+send_len(kk))=s(1,2)
                    Pil0_send_s3(Pil0_send_adr(kk)+send_len(kk))=s(2,1)
                    Pil0_send_s4(Pil0_send_adr(kk)+send_len(kk))=s(2,2)
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
         call localise1(imx,imy,x_a,y_a, &
                          G_xg_8,G_yg_8,h1,h2,1,1, &
                          minx,maxx,miny,maxy)
         imx = min(max(imx,glb_pil_w+1),G_ni-glb_pil_e-1)
         imy = min(max(imy,glb_pil_s+1),G_nj-glb_pil_n-1)
! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Pil0_recvmaxproc
                ki=Pil0_recvproc(kk)
                if (imx.ge.Ptopo_gindx(1,ki).and.imx.le.Ptopo_gindx(2,ki).and. &
                    imy.ge.Ptopo_gindx(3,ki).and.imy.le.Ptopo_gindx(4,ki))then
                    recv_len(kk)=recv_len(kk)+1
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil0_recv_i(Pil0_recv_adr(kk)+recv_len(kk))=ii
                    Pil0_recv_j(Pil0_recv_adr(kk)+recv_len(kk))=jj
                endif
             enddo       
         endif

! check to send to who
         if (imx.ge.l_i0.and.imx.le.l_i0+l_ni-1 .and. &
             imy.ge.l_j0.and.imy.le.l_j0+l_nj-1      ) then
             do kk=1,Pil0_sendmaxproc
                ki=Pil0_sendproc(kk)
                if (i  .ge.Ptopo_gindx(1,ki).and.i  .le.Ptopo_gindx(2,ki).and. &
                    j  .ge.Ptopo_gindx(3,ki).and.j  .le.Ptopo_gindx(4,ki))then
                    send_len(kk)=send_len(kk)+1
                    Pil0_send_imx(Pil0_send_adr(kk)+send_len(kk))=imx-l_i0+1
                    Pil0_send_imy(Pil0_send_adr(kk)+send_len(kk))=imy-l_j0+1
                    Pil0_send_xxr(Pil0_send_adr(kk)+send_len(kk))=x_a
                    Pil0_send_yyr(Pil0_send_adr(kk)+send_len(kk))=y_a
                    Pil0_send_s1(Pil0_send_adr(kk)+send_len(kk))=s(1,1)
                    Pil0_send_s2(Pil0_send_adr(kk)+send_len(kk))=s(1,2)
                    Pil0_send_s3(Pil0_send_adr(kk)+send_len(kk))=s(2,1)
                    Pil0_send_s4(Pil0_send_adr(kk)+send_len(kk))=s(2,2)
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
         call localise1(imx,imy,x_a,y_a, &
                          G_xg_8,G_yg_8,h1,h2,1,1, &
                          minx,maxx,miny,maxy)
         imx = min(max(imx,glb_pil_w+1),G_ni-glb_pil_e-1)
         imy = min(max(imy,glb_pil_s+1),G_nj-glb_pil_n-1)

! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Pil0_recvmaxproc
                ki=Pil0_recvproc(kk)
                if (imx.ge.Ptopo_gindx(1,ki).and.imx.le.Ptopo_gindx(2,ki).and. &
                    imy.ge.Ptopo_gindx(3,ki).and.imy.le.Ptopo_gindx(4,ki))then
                    recv_len(kk)=recv_len(kk)+1
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil0_recv_i(Pil0_recv_adr(kk)+recv_len(kk))=ii
                    Pil0_recv_j(Pil0_recv_adr(kk)+recv_len(kk))=jj
                endif
             enddo       
         endif

! check to send to who
         if (imx.ge.l_i0.and.imx.le.l_i0+l_ni-1 .and. &
             imy.ge.l_j0.and.imy.le.l_j0+l_nj-1      ) then
             do kk=1,Pil0_sendmaxproc
                ki=Pil0_sendproc(kk)
                if (i  .ge.Ptopo_gindx(1,ki).and.i  .le.Ptopo_gindx(2,ki).and. &
                    j  .ge.Ptopo_gindx(3,ki).and.j  .le.Ptopo_gindx(4,ki))then
                    send_len(kk)=send_len(kk)+1
                    Pil0_send_imx(Pil0_send_adr(kk)+send_len(kk))=imx-l_i0+1
                    Pil0_send_imy(Pil0_send_adr(kk)+send_len(kk))=imy-l_j0+1
                    Pil0_send_xxr(Pil0_send_adr(kk)+send_len(kk))=x_a
                    Pil0_send_yyr(Pil0_send_adr(kk)+send_len(kk))=y_a
                    Pil0_send_s1(Pil0_send_adr(kk)+send_len(kk))=s(1,1)
                    Pil0_send_s2(Pil0_send_adr(kk)+send_len(kk))=s(1,2)
                    Pil0_send_s3(Pil0_send_adr(kk)+send_len(kk))=s(2,1)
                    Pil0_send_s4(Pil0_send_adr(kk)+send_len(kk))=s(2,2)
                endif
             enddo       
         endif
      enddo   
      enddo   
!Check receive lengths from each processor
!     do ki=1,Pil0_recvmaxproc
!        kk=Pil0_recvproc(ki)
!        if (Ptopo_couleur.eq.0) then
!            kkproc = kk+Ptopo_numproc-1
!        else
!            kkproc = kk -1
!        endif
!    write(6,1000) 'Pil0_recv_len',kkproc,Pil0_recv_len(kk),Pil0_recv_adr(kk)
!   enddo
!Check send lengths to each processor

!     do ki=1,Pil0_sendmaxproc
!        kk=Pil0_sendproc(ki)
!        if (Ptopo_couleur.eq.0) then
!            kkproc = kk+Ptopo_numproc-1
!        else
!            kkproc = kk -1
!        endif
! write(6,1000) 'Pil0_send_len',kkproc,Pil0_send_len(kk),Pil0_send_adr(kk)
!     enddo
      deallocate (recv_len,send_len)

 1000 format(a15,i3,'=',i5,'bytes, addr=',i5)
 1001 format(a15,i3,'=',i4,'bytes   i:', i3,' j:',i3)
       

!
      return
      end

