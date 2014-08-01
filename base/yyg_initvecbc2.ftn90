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
!***s/r yyg_initvecbc2 - to initialize communication pattern for V field 
!


      Subroutine yyg_initvecbc2()
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
#include "yyg_pilv.cdk"

      integer err,Ndim,i,j,k,kk,ii,jj,ki,ksend,krecv
      integer imx1,imx2
      integer imy1,imy2
      integer kkproc,adr
      integer, dimension (:), pointer :: recv_len,recvw_len,recve_len,recvs_len,recvn_len
      integer, dimension (:), pointer :: send_len,sendw_len,sende_len,sends_len,sendn_len
      real*8  xx_8(G_ni,G_njv),yy_8(G_ni,G_njv)
      real*8  xgu_8(G_ni-1),ygv_8(G_njv)
      real*8  t,p,s(2,2),h1,h2
      real*8  x_d,y_d,x_a,y_a   
!
!     Get global xgu,ygv,xx,yy
       do i=1,G_ni-1
       xgu_8(i)=0.5D0 *(G_xg_8(i+1)+G_xg_8(i))
       enddo
       do j=1,G_njv
       ygv_8(j)= 0.5D0*(G_yg_8(j+1)+G_yg_8(j))
       enddo
!
      do j=1,G_njv
      do i=1,G_ni
         xx_8(i,j)=G_xg_8(i)
         yy_8(i,j)=ygv_8(j)
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
      do j=1, G_njv
      do i=1,glb_pil_w
!        V vector
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx1,imy1,x_a,y_a, &
                          G_xg_8(1),ygv_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          xgu_8(1),G_yg_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_njv-glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy2 = min(max(imy2-1,glb_pil_s+1),G_njv-glb_pil_n-3)
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
      do j=1, G_njv
      do i=G_ni-glb_pil_e+1,G_ni
!        V vector
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx1,imy1,x_a,y_a, &
                          G_xg_8(1),ygv_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          xgu_8(1),G_yg_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_njv-glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy2 = min(max(imy2-1,glb_pil_s+1),G_njv-glb_pil_n-3)
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
      do i=1+glb_pil_w,G_ni-glb_pil_e
!        V vector
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx1,imy1,x_a,y_a, &
                          G_xg_8(1),ygv_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          xgu_8(1),G_yg_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_njv-glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy2 = min(max(imy2-1,glb_pil_s+1),G_njv-glb_pil_n-3)
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
      do j=G_njv-glb_pil_n+1,G_njv
      do i=1+glb_pil_w,G_ni-glb_pil_e
!        V vector
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx1,imy1,x_a,y_a, &
                          G_xg_8(1),ygv_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          xgu_8(1),G_yg_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_njv-glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy2 = min(max(imy2-1,glb_pil_s+1),G_njv-glb_pil_n-3)
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
     Pil_vsend_all=0
     Pil_vrecv_all=0
     Pil_vsendmaxproc=0
     Pil_vrecvmaxproc=0

     do kk=1,Ptopo_numproc
        send_len(kk)=sendw_len(kk)+sende_len(kk) + sends_len(kk)+sendn_len(kk)
        recv_len(kk)=recvw_len(kk)+recve_len(kk) + recvs_len(kk)+recvn_len(kk)
        Pil_vsend_all=send_len(kk)+Pil_vsend_all
        Pil_vrecv_all=recv_len(kk)+Pil_vrecv_all

        if (send_len(kk).gt.0) Pil_vsendmaxproc=Pil_vsendmaxproc+1
        if (recv_len(kk).gt.0) Pil_vrecvmaxproc=Pil_vrecvmaxproc+1
     enddo
!
!     print *,'Allocate common vectors'
      allocate (Pil_vrecvproc(Pil_vrecvmaxproc))
      allocate (Pil_vrecv_len(Pil_vrecvmaxproc))
      allocate (Pil_vrecvw_len(Pil_vrecvmaxproc))
      allocate (Pil_vrecve_len(Pil_vrecvmaxproc))
      allocate (Pil_vrecvs_len(Pil_vrecvmaxproc))
      allocate (Pil_vrecvn_len(Pil_vrecvmaxproc))
      allocate (Pil_vrecvw_adr(Pil_vrecvmaxproc))
      allocate (Pil_vrecve_adr(Pil_vrecvmaxproc))
      allocate (Pil_vrecvs_adr(Pil_vrecvmaxproc))
      allocate (Pil_vrecvn_adr(Pil_vrecvmaxproc))

      allocate (Pil_vsendproc(Pil_vsendmaxproc))
      allocate (Pil_vsend_len(Pil_vsendmaxproc))
      allocate (Pil_vsendw_len(Pil_vsendmaxproc))
      allocate (Pil_vsende_len(Pil_vsendmaxproc))
      allocate (Pil_vsends_len(Pil_vsendmaxproc))
      allocate (Pil_vsendn_len(Pil_vsendmaxproc))
      allocate (Pil_vsendw_adr(Pil_vsendmaxproc))
      allocate (Pil_vsende_adr(Pil_vsendmaxproc))
      allocate (Pil_vsends_adr(Pil_vsendmaxproc))
      allocate (Pil_vsendn_adr(Pil_vsendmaxproc))
      Pil_vrecvw_len(:) = 0
      Pil_vrecve_len(:) = 0
      Pil_vrecvs_len(:) = 0
      Pil_vrecvn_len(:) = 0
      Pil_vsendw_len(:) = 0
      Pil_vsende_len(:) = 0
      Pil_vsends_len(:) = 0
      Pil_vsendn_len(:) = 0
      Pil_vrecvw_adr(:) = 0
      Pil_vrecve_adr(:) = 0
      Pil_vrecvs_adr(:) = 0
      Pil_vrecvn_adr(:) = 0
      Pil_vsendw_adr(:) = 0
      Pil_vsende_adr(:) = 0
      Pil_vsends_adr(:) = 0
      Pil_vsendn_adr(:) = 0

!    print*,'Pil_vsendmaxproc=',Pil_vsendmaxproc,'recvmaxproc=',Pil_vrecvmaxproc

     ksend=0
     krecv=0
     Pil_vsend_all=0
     Pil_vrecv_all=0
!
! Fill the lengths and addresses for selected processors to communicate
!
     do kk=1,Ptopo_numproc
        if (send_len(kk).gt.0) then
            ksend=ksend+1
            Pil_vsendproc(ksend)=kk
            Pil_vsend_len(ksend)=send_len(kk)
            Pil_vsendw_len(ksend)=sendw_len(kk)
            Pil_vsende_len(ksend)=sende_len(kk)
            Pil_vsends_len(ksend)=sends_len(kk)
            Pil_vsendn_len(ksend)=sendn_len(kk)

            Pil_vsendw_adr(ksend)= Pil_vsend_all
            Pil_vsend_all= Pil_vsend_all + Pil_vsend_len(ksend)
            Pil_vsende_adr(ksend)= Pil_vsendw_adr(ksend)+Pil_vsendw_len(ksend)
            Pil_vsends_adr(ksend)= Pil_vsende_adr(ksend)+Pil_vsende_len(ksend)
            Pil_vsendn_adr(ksend)= Pil_vsends_adr(ksend)+Pil_vsends_len(ksend)
        endif
        if (recv_len(kk).gt.0) then
            krecv=krecv+1
            Pil_vrecvproc(krecv)=kk
            Pil_vrecv_len(krecv)=recv_len(kk)
            Pil_vrecvw_len(krecv)=recvw_len(kk)
            Pil_vrecve_len(krecv)=recve_len(kk)
            Pil_vrecvs_len(krecv)=recvs_len(kk)
            Pil_vrecvn_len(krecv)=recvn_len(kk)

            Pil_vrecvw_adr(krecv)= Pil_vrecv_all
            Pil_vrecv_all= Pil_vrecv_all + Pil_vrecv_len(krecv)
            Pil_vrecve_adr(krecv)= Pil_vrecvw_adr(krecv)+Pil_vrecvw_len(krecv)
            Pil_vrecvs_adr(krecv)= Pil_vrecve_adr(krecv)+Pil_vrecve_len(krecv)
            Pil_vrecvn_adr(krecv)= Pil_vrecvs_adr(krecv)+Pil_vrecvs_len(krecv)
        endif

     enddo
!    print *,'krecv=',krecv,'Pil_vrecvmaxproc=',Pil_vrecvmaxproc
!    print *,'ksend=',ksend,'Pil_vsendmaxproc=',Pil_vsendmaxproc

!     print *,'Summary of comm procs'
!     do kk=1,Pil_vrecvmaxproc
!  print *,'From proc:',Pil_vrecvproc(kk),'Pil_vrecv_len',Pil_vrecvw_len(kk),Pil_vrecve_len(kk),Pil_vrecvs_len(kk),Pil_vrecvn_len(kk),'adr',Pil_vrecvw_adr(kk),Pil_vrecve_adr(kk),Pil_vrecvs_adr(kk),Pil_vrecvn_adr(kk)
!     enddo
!     do kk=1,Pil_vsendmaxproc
!       print *,'To proc:',Pil_vsendproc(kk),'Pil_vsend_len',Pil_vsendw_len(kk),Pil_vsende_len(kk),Pil_vsends_len(kk),Pil_vsendn_len(kk),'adr',Pil_vsendw_adr(kk),Pil_vsende_adr(kk),Pil_vsends_adr(kk),Pil_vsendn_adr(kk)
!     enddo

!
! Now allocate the vectors needed for sending and receiving each processor
!
      if (Pil_vrecv_all.gt.0) then
          allocate (Pil_vrecv_i(Pil_vrecv_all))
          allocate (Pil_vrecv_j(Pil_vrecv_all))
          Pil_vrecv_i(:) = 0
          Pil_vrecv_j(:) = 0
      endif

      if (Pil_vsend_all.gt.0) then
          allocate (Pil_vsend_imx1(Pil_vsend_all))
          allocate (Pil_vsend_imy1(Pil_vsend_all))
          allocate (Pil_vsend_imx2(Pil_vsend_all))
          allocate (Pil_vsend_imy2(Pil_vsend_all))
          allocate (Pil_vsend_xxr(Pil_vsend_all))
          allocate (Pil_vsend_yyr(Pil_vsend_all))
          allocate (Pil_vsend_s1(Pil_vsend_all))
          allocate (Pil_vsend_s2(Pil_vsend_all))
          Pil_vsend_imx1(:) = 0
          Pil_vsend_imy1(:) = 0
          Pil_vsend_imx2(:) = 0
          Pil_vsend_imy2(:) = 0
          Pil_vsend_xxr(:) = 0.0
          Pil_vsend_yyr(:) = 0.0
          Pil_vsend_s1(:) = 0.0
          Pil_vsend_s2(:) = 0.0
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
      do j=1, G_njv
      do i=1,glb_pil_w
!        V vector
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx1,imy1,x_a,y_a, &
                          G_xg_8(1),ygv_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          xgu_8(1),G_yg_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_njv-glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy2 = min(max(imy2-1,glb_pil_s+1),G_njv-glb_pil_n-3)
!

! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Pil_vrecvmaxproc
                ki=Pil_vrecvproc(kk)
                if (max(imx1,imx2).ge.Ptopo_gindx(1,ki).and. &
                    max(imx1,imx2).le.Ptopo_gindx(2,ki).and. &
                    max(imy1,imy2).ge.Ptopo_gindx(3,ki).and. &
                    max(imy1,imy2).le.Ptopo_gindx(4,ki) ) then
                    recvw_len(kk)=recvw_len(kk)+1
                    adr=Pil_vrecvw_adr(kk)+recvw_len(kk)
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil_vrecv_i(adr)=ii
                    Pil_vrecv_j(adr)=jj
                endif
             enddo       
         endif

! check to send to who
         if (max(imx1,imx2).ge.l_i0.and.         &
             max(imx1,imx2).le.l_i0+l_ni-1 .and. &
             max(imy1,imy2).ge.l_j0.and.         &
             max(imy1,imy2).le.l_j0+l_nj-1) then
             do kk=1,Pil_vsendmaxproc
                ki=Pil_vsendproc(kk)
                if (i  .ge.Ptopo_gindx(1,ki).and.i  .le.Ptopo_gindx(2,ki).and. &
                    j  .ge.Ptopo_gindx(3,ki).and.j  .le.Ptopo_gindx(4,ki))then
                    sendw_len(kk)=sendw_len(kk)+1
                    adr=Pil_vsendw_adr(kk)+sendw_len(kk)
                    Pil_vsend_imx1(adr)=imx1-l_i0+1
                    Pil_vsend_imy1(adr)=imy1-l_j0+1
                    Pil_vsend_imx2(adr)=imx2-l_i0+1
                    Pil_vsend_imy2(adr)=imy2-l_j0+1
                    Pil_vsend_xxr(adr)=x_a
                    Pil_vsend_yyr(adr)=y_a
                    Pil_vsend_s1(adr)=s(2,2)
                    Pil_vsend_s2(adr)=s(2,1)
                endif
             enddo       
         endif
      enddo   
      enddo   
!
!
! East section
      do j=1, G_njv
      do i=G_ni-glb_pil_e+1,G_ni
!        V vector
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx1,imy1,x_a,y_a, &
                          G_xg_8(1),ygv_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          xgu_8(1),G_yg_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_njv-glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy2 = min(max(imy2-1,glb_pil_s+1),G_njv-glb_pil_n-3)
!
! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Pil_vrecvmaxproc
                ki=Pil_vrecvproc(kk)
                if (max(imx1,imx2).ge.Ptopo_gindx(1,ki).and. &
                    max(imx1,imx2).le.Ptopo_gindx(2,ki).and. &
                    max(imy1,imy2).ge.Ptopo_gindx(3,ki).and. &
                    max(imy1,imy2).le.Ptopo_gindx(4,ki) ) then
                    recve_len(kk)=recve_len(kk)+1
                    adr=Pil_vrecve_adr(kk)+recve_len(kk)
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil_vrecv_i(adr)=ii
                    Pil_vrecv_j(adr)=jj
                endif
             enddo       
         endif

! check to send to who
         if (max(imx1,imx2).ge.l_i0.and.         &
             max(imx1,imx2).le.l_i0+l_ni-1 .and. &
             max(imy1,imy2).ge.l_j0.and.         &
             max(imy1,imy2).le.l_j0+l_nj-1) then
             do kk=1,Pil_vsendmaxproc
                ki=Pil_vsendproc(kk)
                if (i  .ge.Ptopo_gindx(1,ki).and.i .le.Ptopo_gindx(2,ki).and. &
                    j  .ge.Ptopo_gindx(3,ki).and.j .le.Ptopo_gindx(4,ki))then
                    sende_len(kk)=sende_len(kk)+1
                    adr=Pil_vsende_adr(kk)+sende_len(kk)
                    Pil_vsend_imx1(adr)=imx1-l_i0+1
                    Pil_vsend_imy1(adr)=imy1-l_j0+1
                    Pil_vsend_imx2(adr)=imx2-l_i0+1
                    Pil_vsend_imy2(adr)=imy2-l_j0+1
                    Pil_vsend_xxr(adr)=x_a
                    Pil_vsend_yyr(adr)=y_a
                    Pil_vsend_s1(adr)=s(2,2)
                    Pil_vsend_s2(adr)=s(2,1)
                endif
             enddo       
         endif
      enddo   
      enddo   
!
! South section
      do j=1,glb_pil_s
      do i=1+glb_pil_w,G_ni-glb_pil_e
!        V vector
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx1,imy1,x_a,y_a,  &
                          G_xg_8(1),ygv_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a,  &
                          xgu_8(1),G_yg_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_njv-glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy2 = min(max(imy2-1,glb_pil_s+1),G_njv-glb_pil_n-3)
!
! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Pil_vrecvmaxproc
                ki=Pil_vrecvproc(kk)
                if (max(imx1,imx2).ge.Ptopo_gindx(1,ki).and. &
                    max(imx1,imx2).le.Ptopo_gindx(2,ki).and. &
                    max(imy1,imy2).ge.Ptopo_gindx(3,ki).and. &
                    max(imy1,imy2).le.Ptopo_gindx(4,ki) ) then
                    recvs_len(kk)=recvs_len(kk)+1
                    adr=Pil_vrecvs_adr(kk)+recvs_len(kk)
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil_vrecv_i(adr)=ii
                    Pil_vrecv_j(adr)=jj
                endif
             enddo       
         endif

! check to send to who
         if (max(imx1,imx2).ge.l_i0.and.         &
             max(imx1,imx2).le.l_i0+l_ni-1 .and. &
             max(imy1,imy2).ge.l_j0.and.         &
             max(imy1,imy2).le.l_j0+l_nj-1) then
             do kk=1,Pil_vsendmaxproc
                ki=Pil_vsendproc(kk)
                if (i  .ge.Ptopo_gindx(1,ki).and.i  .le.Ptopo_gindx(2,ki).and. &
                    j  .ge.Ptopo_gindx(3,ki).and.j  .le.Ptopo_gindx(4,ki))then
                    sends_len(kk)=sends_len(kk)+1
                    adr=Pil_vsends_adr(kk)+sends_len(kk)
                    Pil_vsend_imx1(adr)=imx1-l_i0+1
                    Pil_vsend_imy1(adr)=imy1-l_j0+1
                    Pil_vsend_imx2(adr)=imx2-l_i0+1
                    Pil_vsend_imy2(adr)=imy2-l_j0+1
                    Pil_vsend_xxr(adr)=x_a
                    Pil_vsend_yyr(adr)=y_a
                    Pil_vsend_s1(adr)=s(2,2)
                    Pil_vsend_s2(adr)=s(2,1)
                endif
             enddo       
         endif
      enddo   
      enddo   
!
! North section
      do j=G_njv-glb_pil_n+1,G_njv
      do i=1+glb_pil_w,G_ni-glb_pil_e
!        V vector
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx1,imy1,x_a,y_a, &
                          G_xg_8(1),ygv_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          xgu_8(1),G_yg_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_njv-glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy2 = min(max(imy2-1,glb_pil_s+1),G_njv-glb_pil_n-3)
!

! check to collect from who
         if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
             j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
             do kk=1,Pil_vrecvmaxproc
                ki=Pil_vrecvproc(kk)
                if (max(imx1,imx2).ge.Ptopo_gindx(1,ki).and. &
                    max(imx1,imx2).le.Ptopo_gindx(2,ki).and. &
                    max(imy1,imy2).ge.Ptopo_gindx(3,ki).and. &
                    max(imy1,imy2).le.Ptopo_gindx(4,ki) ) then
                    recvn_len(kk)=recvn_len(kk)+1
                    adr=Pil_vrecvn_adr(kk)+recvn_len(kk)
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil_vrecv_i(adr)=ii
                    Pil_vrecv_j(adr)=jj
                endif
             enddo       
         endif

! check to send to who
         if (max(imx1,imx2).ge.l_i0.and.         &
             max(imx1,imx2).le.l_i0+l_ni-1 .and. &
             max(imy1,imy2).ge.l_j0.and.         &
             max(imy1,imy2).le.l_j0+l_nj-1) then
             do kk=1,Pil_vsendmaxproc
                ki=Pil_vsendproc(kk)
                if (i  .ge.Ptopo_gindx(1,ki).and.i  .le.Ptopo_gindx(2,ki).and. &
                    j  .ge.Ptopo_gindx(3,ki).and.j  .le.Ptopo_gindx(4,ki))then
                    sendn_len(kk)=sendn_len(kk)+1
                    adr=Pil_vsendn_adr(kk)+sendn_len(kk)
                    Pil_vsend_imx1(adr)=imx1-l_i0+1
                    Pil_vsend_imy1(adr)=imy1-l_j0+1
                    Pil_vsend_imx2(adr)=imx2-l_i0+1
                    Pil_vsend_imy2(adr)=imy2-l_j0+1
                    Pil_vsend_xxr(adr)=x_a
                    Pil_vsend_yyr(adr)=y_a
                    Pil_vsend_s1(adr)=s(2,2)
                    Pil_vsend_s2(adr)=s(2,1)
                endif
             enddo       
         endif
      enddo   
      enddo   
!Check receive lengths from each processor
!     do ki=1,Pil_vrecvmaxproc
!        kk=Pil_vrecvproc(ki)
!        if (Ptopo_couleur.eq.0) then
!            kkproc = kk+Ptopo_numproc-1
!        else
!            kkproc = kk -1
!        endif
!    write(6,1000) 'Pil_vrecvw_len',kkproc,Pil_vrecvw_len(kk),Pil_vrecvw_adr(kk)
!    write(6,1000) 'Pil_vrecve_len',kkproc,Pil_vrecve_len(kk),Pil_vrecve_adr(kk)
!    write(6,1000) 'Pil_vrecvs_len',kkproc,Pil_vrecvs_len(kk),Pil_vrecvs_adr(kk)
!    write(6,1000) 'Pil_vrecvn_len',kkproc,Pil_vrecvn_len(kk),Pil_vrecvn_adr(kk)
!   enddo

!Check send lengths to each processor

!     do ki=1,Pil_vsendmaxproc
!        kk=Pil_vsendproc(ki)
!        if (Ptopo_couleur.eq.0) then
!            kkproc = kk+Ptopo_numproc-1
!        else
!            kkproc = kk -1
!        endif
! write(6,1000) 'Pil_vsendw_len',kkproc,Pil_vsendw_len(kk),Pil_vsendw_adr(kk)
! write(6,1000) 'Pil_vsende_len',kkproc,Pil_vsende_len(kk),Pil_vsende_adr(kk)
! write(6,1000) 'Pil_vsends_len',kkproc,Pil_vsends_len(kk),Pil_vsends_adr(kk)
! write(6,1000) 'Pil_vsendn_len',kkproc,Pil_vsendn_len(kk),Pil_vsendn_adr(kk)
!     enddo
      deallocate (recv_len,recvw_len,recve_len,recvs_len,recvn_len)
      deallocate (send_len,sendw_len,sende_len,sends_len,sendn_len)

 1000 format(a15,i3,'=',i5,'bytes, addr=',i5)
 1001 format(a15,i3,'=',i4,'bytes   i:', i3,' j:',i3)
       
!
      return
      end

