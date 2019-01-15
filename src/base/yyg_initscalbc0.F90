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
      use tdpack
      use glb_ld
      use glb_pil
      use ptopo
      use yyg_pil0
      implicit none
#include <arch_specific.hf>
!
!author
!           Abdessamad Qaddouri/ V.lee - September 2011
!  PLEASE consult Abdessamad or Vivian before modifying this routine.
!

      integer i,j,imx,imy,kk,ii,jj,ki,ksend,krecv
      integer, dimension (:), pointer :: recv_len,send_len
      real*8  xx_8(G_ni,G_nj),yy_8(G_ni,G_nj)
      real*8  xg_8(1-G_ni:2*G_ni),yg_8(1-G_nj:2*G_nj)
      real*8  s(2,2),h1,h2
      real*8  x_d,y_d,x_a,y_a
      real*8 TWO_8
      parameter( TWO_8   = 2.0d0 )
!
!     Localise could get point way outside of the actual grid in search
!     So extend all global arrays: xg_8,yg_8

      do i=1,G_ni
         xg_8(i) = G_xg_8(i)
      end do
      do j=1,G_nj
         yg_8(j) = G_yg_8(j)
      enddo

      do i=-G_ni+1,0
         xg_8(i) = xg_8(i+G_ni) - TWO_8*pi_8
      end do
      do i=G_ni+1,2*G_ni
         xg_8(i) = xg_8(i-G_ni) + TWO_8*pi_8
      end do

      yg_8( 0    ) = -(yg_8(1) + pi_8)
      yg_8(-1    ) = -TWO_8*pi_8 -  &
           (yg_8(0)+yg_8(1)+yg_8(2))
      yg_8(G_nj+1) =  pi_8 - yg_8(G_nj)
      yg_8(G_nj+2) =  TWO_8*pi_8 - &
           (yg_8(G_nj+1)+yg_8(G_nj)+yg_8(G_nj-1))
      do j=-2,-G_nj+1,-1
         yg_8(j) = 1.01*yg_8(j+1)
      end do
      do j=G_nj+3,2*G_nj
         yg_8(j) = 1.01*yg_8(j-1)
      end do

      do j=1,G_nj
      do i=1,G_ni
         xx_8(i,j)=xg_8(i)
      enddo
      enddo
      do j=1,G_nj
      do i=1,G_ni
         yy_8(i,j)=yg_8(j)
      enddo
      enddo

!Delta xg, yg is not identical between xg(i) and xg(i+1)
!h1, h2 used in this routine is ok as it is a close estimate for
!creating YY pattern exchange and it works on the global tile

      !do i=2,G_ni
      !   h1=xg_8(i)-xg_8(i-1)
      !   print *,'h1=',h1
      !enddo
      !h1=xg_8(2)-xg_8(1)
      !print *,'h1 final=',h1
      !do j=2,G_nj
      !   h2=yg_8(j)-yg_8(j-1)
      !   print *,'h2=',h2
      !enddo
      !h2=yg_8(2)-yg_8(1)
      !print *,'h2 final=',h2

      h1=xg_8(2)-xg_8(1)
      h2=yg_8(2)-yg_8(1)
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

      do j=1, G_nj
      do i=1,Glb_pil_w
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx,imy,x_a,y_a, &
                          xg_8(1),yg_8(1),h1,h2,1,1)
         imx = min(max(imx,glb_pil_w+1),G_ni-glb_pil_e-1)
         imy = min(max(imy,glb_pil_s+1),G_nj-glb_pil_n-1)
! check to collect from who
         if (i >= l_i0.and.i <= l_i0+l_ni-1 .and. &
             j >= l_j0.and.j <= l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (imx >= Ptopo_gindx(1,kk).and.imx <= Ptopo_gindx(2,kk).and. &
                    imy >= Ptopo_gindx(3,kk).and.imy <= Ptopo_gindx(4,kk)) then
                    recv_len(kk)=recv_len(kk)+1
                endif
             enddo
         endif

! check to send to who
         if (imx >= l_i0.and.imx <= l_i0+l_ni-1 .and. &
             imy >= l_j0.and.imy <= l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (i >= Ptopo_gindx(1,kk).and.i <= Ptopo_gindx(2,kk).and. &
                    j >= Ptopo_gindx(3,kk).and.j <= Ptopo_gindx(4,kk))then
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
         call localise(imx,imy,x_a,y_a, &
                          xg_8(1),yg_8(1),h1,h2,1,1)
         imx = min(max(imx,glb_pil_w+1),G_ni-glb_pil_e-1)
         imy = min(max(imy,glb_pil_s+1),G_nj-glb_pil_n-1)
! check to collect from who
         if (i >= l_i0.and.i <= l_i0+l_ni-1 .and. &
             j >= l_j0.and.j <= l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (imx >= Ptopo_gindx(1,kk).and.imx <= Ptopo_gindx(2,kk).and. &
                    imy >= Ptopo_gindx(3,kk).and.imy <= Ptopo_gindx(4,kk))then
                    recv_len(kk)=recv_len(kk)+1
                endif
             enddo
         endif

! check to send to who
         if (imx >= l_i0.and.imx <= l_i0+l_ni-1 .and. &
             imy >= l_j0.and.imy <= l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (i >= Ptopo_gindx(1,kk).and.i <= Ptopo_gindx(2,kk).and. &
                    j >= Ptopo_gindx(3,kk).and.j <= Ptopo_gindx(4,kk))then
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
         call localise(imx,imy,x_a,y_a, &
                          xg_8(1),yg_8(1),h1,h2,1,1)
         imx = min(max(imx,glb_pil_w+1),G_ni-glb_pil_e-1)
         imy = min(max(imy,glb_pil_s+1),G_nj-glb_pil_n-1)
! check to collect from who
         if (i >= l_i0.and.i <= l_i0+l_ni-1 .and. &
             j >= l_j0.and.j <= l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (imx >= Ptopo_gindx(1,kk).and.imx <= Ptopo_gindx(2,kk).and. &
                    imy >= Ptopo_gindx(3,kk).and.imy <= Ptopo_gindx(4,kk))then
                    recv_len(kk)=recv_len(kk)+1
                endif
             enddo
         endif

! check to send to who
         if (imx >= l_i0.and.imx <= l_i0+l_ni-1 .and. &
             imy >= l_j0.and.imy <= l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (i >= Ptopo_gindx(1,kk).and.i <= Ptopo_gindx(2,kk).and. &
                    j >= Ptopo_gindx(3,kk).and.j <= Ptopo_gindx(4,kk))then
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
         call localise(imx,imy,x_a,y_a, &
                          xg_8(1),yg_8(1),h1,h2,1,1)
         imx = min(max(imx,glb_pil_w+1),G_ni-glb_pil_e-1)
         imy = min(max(imy,glb_pil_s+1),G_nj-glb_pil_n-1)

! check to collect from who
         if (i >= l_i0.and.i <= l_i0+l_ni-1 .and. &
             j >= l_j0.and.j <= l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (imx >= Ptopo_gindx(1,kk).and.imx <= Ptopo_gindx(2,kk).and. &
                    imy >= Ptopo_gindx(3,kk).and.imy <= Ptopo_gindx(4,kk))then
                    recv_len(kk)=recv_len(kk)+1
                endif
             enddo
         endif

! check to send to who
         if (imx >= l_i0.and.imx <= l_i0+l_ni-1 .and. &
             imy >= l_j0.and.imy <= l_j0+l_nj-1      ) then
             do kk=1,Ptopo_numproc
                if (i >= Ptopo_gindx(1,kk).and.i <= Ptopo_gindx(2,kk).and. &
                    j >= Ptopo_gindx(3,kk).and.j <= Ptopo_gindx(4,kk))then
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

        if (send_len(kk) > 0) Pil0_sendmaxproc=Pil0_sendmaxproc+1
        if (recv_len(kk) > 0) Pil0_recvmaxproc=Pil0_recvmaxproc+1
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
        if (send_len(kk) > 0) then
            ksend=ksend+1
            Pil0_sendproc(ksend)=kk
            Pil0_send_len(ksend)=send_len(kk)

            Pil0_send_adr(ksend)= Pil0_send_all
            Pil0_send_all= Pil0_send_all + Pil0_send_len(ksend)
        endif
        if (recv_len(kk) > 0) then
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
      if (Pil0_recv_all > 0) then
          allocate (Pil0_recv_i(Pil0_recv_all))
          allocate (Pil0_recv_j(Pil0_recv_all))
          Pil0_recv_i(:) = 0
          Pil0_recv_j(:) = 0
      endif

      if (Pil0_send_all > 0) then
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
         call localise(imx,imy,x_a,y_a, &
                          xg_8(1),yg_8(1),h1,h2,1,1)
         imx = min(max(imx,glb_pil_w+1),G_ni-glb_pil_e-1)
         imy = min(max(imy,glb_pil_s+1),G_nj-glb_pil_n-1)
! check to collect from who
         if (i >= l_i0.and.i <= l_i0+l_ni-1 .and. &
             j >= l_j0.and.j <= l_j0+l_nj-1      ) then
             do kk=1,Pil0_recvmaxproc
                ki=Pil0_recvproc(kk)
                if (imx >= Ptopo_gindx(1,ki).and.imx <= Ptopo_gindx(2,ki).and. &
                    imy >= Ptopo_gindx(3,ki).and.imy <= Ptopo_gindx(4,ki))then
                    recv_len(kk)=recv_len(kk)+1
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil0_recv_i(Pil0_recv_adr(kk)+recv_len(kk))=ii
                    Pil0_recv_j(Pil0_recv_adr(kk)+recv_len(kk))=jj
                endif
             enddo
         endif

! check to send to who
         if (imx >= l_i0.and.imx <= l_i0+l_ni-1 .and. &
             imy >= l_j0.and.imy <= l_j0+l_nj-1      ) then
             do kk=1,Pil0_sendmaxproc
                ki=Pil0_sendproc(kk)
                if (i >= Ptopo_gindx(1,ki).and.i <= Ptopo_gindx(2,ki).and. &
                    j >= Ptopo_gindx(3,ki).and.j <= Ptopo_gindx(4,ki))then
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
         call localise(imx,imy,x_a,y_a, &
                          xg_8(1),yg_8(1),h1,h2,1,1)
         imx = min(max(imx,glb_pil_w+1),G_ni-glb_pil_e-1)
         imy = min(max(imy,glb_pil_s+1),G_nj-glb_pil_n-1)
! check to collect from who
         if (i >= l_i0.and.i <= l_i0+l_ni-1 .and. &
             j >= l_j0.and.j <= l_j0+l_nj-1      ) then
             do kk=1,Pil0_recvmaxproc
                ki=Pil0_recvproc(kk)
                if (imx >= Ptopo_gindx(1,ki).and.imx <= Ptopo_gindx(2,ki).and. &
                    imy >= Ptopo_gindx(3,ki).and.imy <= Ptopo_gindx(4,ki))then
                    recv_len(kk)=recv_len(kk)+1
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil0_recv_i(Pil0_recv_adr(kk)+recv_len(kk))=ii
                    Pil0_recv_j(Pil0_recv_adr(kk)+recv_len(kk))=jj
                endif
             enddo
         endif

! check to send to who
         if (imx >= l_i0.and.imx <= l_i0+l_ni-1 .and. &
             imy >= l_j0.and.imy <= l_j0+l_nj-1      ) then
             do kk=1,Pil0_sendmaxproc
                ki=Pil0_sendproc(kk)
                if (i >= Ptopo_gindx(1,ki).and.i <= Ptopo_gindx(2,ki).and. &
                    j >= Ptopo_gindx(3,ki).and.j <= Ptopo_gindx(4,ki))then
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
         call localise(imx,imy,x_a,y_a, &
                          xg_8(1),yg_8(1),h1,h2,1,1)
         imx = min(max(imx,glb_pil_w+1),G_ni-glb_pil_e-1)
         imy = min(max(imy,glb_pil_s+1),G_nj-glb_pil_n-1)
! check to collect from who
         if (i >= l_i0.and.i <= l_i0+l_ni-1 .and. &
             j >= l_j0.and.j <= l_j0+l_nj-1      ) then
             do kk=1,Pil0_recvmaxproc
                ki=Pil0_recvproc(kk)
                if (imx >= Ptopo_gindx(1,ki).and.imx <= Ptopo_gindx(2,ki).and. &
                    imy >= Ptopo_gindx(3,ki).and.imy <= Ptopo_gindx(4,ki))then
                    recv_len(kk)=recv_len(kk)+1
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil0_recv_i(Pil0_recv_adr(kk)+recv_len(kk))=ii
                    Pil0_recv_j(Pil0_recv_adr(kk)+recv_len(kk))=jj
                endif
             enddo
         endif

! check to send to who
         if (imx >= l_i0.and.imx <= l_i0+l_ni-1 .and. &
             imy >= l_j0.and.imy <= l_j0+l_nj-1      ) then
             do kk=1,Pil0_sendmaxproc
                ki=Pil0_sendproc(kk)
                if (i >= Ptopo_gindx(1,ki).and.i <= Ptopo_gindx(2,ki).and. &
                    j >= Ptopo_gindx(3,ki).and.j <= Ptopo_gindx(4,ki))then
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
         call localise(imx,imy,x_a,y_a, &
                          xg_8(1),yg_8(1),h1,h2,1,1)
         imx = min(max(imx,glb_pil_w+1),G_ni-glb_pil_e-1)
         imy = min(max(imy,glb_pil_s+1),G_nj-glb_pil_n-1)

! check to collect from who
         if (i >= l_i0.and.i <= l_i0+l_ni-1 .and. &
             j >= l_j0.and.j <= l_j0+l_nj-1      ) then
             do kk=1,Pil0_recvmaxproc
                ki=Pil0_recvproc(kk)
                if (imx >= Ptopo_gindx(1,ki).and.imx <= Ptopo_gindx(2,ki).and. &
                    imy >= Ptopo_gindx(3,ki).and.imy <= Ptopo_gindx(4,ki))then
                    recv_len(kk)=recv_len(kk)+1
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil0_recv_i(Pil0_recv_adr(kk)+recv_len(kk))=ii
                    Pil0_recv_j(Pil0_recv_adr(kk)+recv_len(kk))=jj
                endif
             enddo
         endif

! check to send to who
         if (imx >= l_i0.and.imx <= l_i0+l_ni-1 .and. &
             imy >= l_j0.and.imy <= l_j0+l_nj-1      ) then
             do kk=1,Pil0_sendmaxproc
                ki=Pil0_sendproc(kk)
                if (i >= Ptopo_gindx(1,ki).and.i <= Ptopo_gindx(2,ki).and. &
                    j >= Ptopo_gindx(3,ki).and.j <= Ptopo_gindx(4,ki))then
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
!        if (Ptopo_couleur == 0) then
!            kkproc = kk+Ptopo_numproc-1
!        else
!            kkproc = kk -1
!        endif
!    write(6,1000) 'Pil0_recv_len',kkproc,Pil0_recv_len(kk),Pil0_recv_adr(kk)
!   enddo
!Check send lengths to each processor

!     do ki=1,Pil0_sendmaxproc
!        kk=Pil0_sendproc(ki)
!        if (Ptopo_couleur == 0) then
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

