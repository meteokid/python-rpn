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
!***s/r yyg_initscalphy - to initialize communication pattern for cubic
!                            interpolation of scalar fields that includes
!                            the G_halos outside of the global grid.
!                            Only used for the smoothing of physics fields.


      Subroutine yyg_initscalphy()
      use tdpack
      use gem_options
      use glb_ld
      use glb_pil
      use ptopo
      use yyg_pilp
      implicit none
#include <arch_specific.hf>
!
!author
!           Abdessamad Qaddouri/ V.lee - September 2011
!  PLEASE consult Abdessamad or Vivian before modifying this routine.
!

      integer i,j,imx,imy,kk,ii,jj,ki,ksend,krecv
      !integer kkproc
      integer p_i0,p_j0,p_ni,p_nj
      integer pe_gindx(4,Ptopo_numproc)
      integer, dimension (:), pointer :: recv_len,send_len
      real*8  xx_8(1-G_halox:G_ni+G_halox,1-G_haloy:G_nj+G_haloy)
      real*8  yy_8(1-G_halox:G_ni+G_halox,1-G_haloy:G_nj+G_haloy)
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

      h1=xg_8(2)-xg_8(1)
      h2=yg_8(2)-yg_8(1)


!     Add the rest of the points outside of the global grid increasing or
!     decreasing degrees with the same delta x or y
!
      do i=1,G_ni
         xg_8(1-i)    = xg_8(1)    - (i)*h1
         xg_8(i+G_ni) = xg_8(G_ni) + (i)*h1
      end do
      do j=1,G_nj
         yg_8(1-j)    = yg_8(1)    - (j)*h2
         yg_8(j+G_nj) = yg_8(G_nj) + (j)*h2
      end do

!     do i=1-G_halox,G_ni+G_halox
!        print *,'xg_8(',i,')=',xg_8(i)
!     enddo
!     do j=1-G_haloy,G_nj+G_haloy
!        print *,'yg_8(',j,')=',yg_8(j)
!     enddo

      do j=1-G_haloy,G_nj+G_haloy
      do i=1-G_halox,G_ni+G_halox
         xx_8(i,j)=xg_8(i)
      enddo
      enddo
      do j=1-G_haloy,G_nj+G_haloy
      do i=1-G_halox,G_ni+G_halox
         yy_8(i,j)=yg_8(j)
      enddo
      enddo

! And allocate temp vectors needed for counting for each processor
!
      p_i0=l_i0 - west*G_halox
      p_j0=l_j0 - south*G_haloy
      p_ni=l_ni + east*G_halox
      p_nj=l_nj + north*G_haloy
!     print *,'p_i0=',p_i0,' p_j0=',p_j0,' p_ni=',p_ni,' p_nj=',p_nj

      do kk=1,Ptopo_numproc
         pe_gindx(1,kk)=Ptopo_gindx(1,kk) !west
         pe_gindx(2,kk)=Ptopo_gindx(2,kk) !east
         pe_gindx(3,kk)=Ptopo_gindx(3,kk) !south
         pe_gindx(4,kk)=Ptopo_gindx(4,kk) !north
         if (pe_gindx(1,kk) == 1) pe_gindx(1,kk)=1-G_halox
         if (pe_gindx(2,kk) == G_ni) pe_gindx(2,kk)=G_ni+G_halox
         if (pe_gindx(3,kk) == 1) pe_gindx(3,kk)=1-G_haloy
         if (pe_gindx(4,kk) == G_nj) pe_gindx(4,kk)=G_nj+G_halox
!        print *,'kk=',kk,' pe_gindx=',pe_gindx(1,kk),pe_gindx(2,kk),pe_gindx(3,kk),pe_gindx(4,kk)
      enddo
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

      do j=1-G_haloy, G_nj+G_haloy
      do i=1-G_halox,Glb_pil_w
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx,imy,x_a,y_a, &
                          xg_8(1),yg_8(1),h1,h2,1,1)
         imx = min(max(imx-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy = min(max(imy-1,glb_pil_s+1),G_nj-glb_pil_n-3)
! check to collect from who
         if (i >= p_i0.and.i <= l_i0+p_ni-1 .and. &
             j >= p_j0.and.j <= l_j0+p_nj-1      ) then
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
                if (i >= pe_gindx(1,kk).and.i <= pe_gindx(2,kk).and. &
                    j >= pe_gindx(3,kk).and.j <= pe_gindx(4,kk))then
                    send_len(kk)=send_len(kk)+1
                endif
             enddo
         endif
      enddo
      enddo
!
!
! East section
      do j=1-G_haloy, G_nj+G_haloy
      do i=G_ni-Glb_pil_e+1,G_ni+G_halox
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx,imy,x_a,y_a, &
                          xg_8(1),yg_8(1),h1,h2,1,1)
         imx = min(max(imx-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy = min(max(imy-1,glb_pil_s+1),G_nj-glb_pil_n-3)
! check to collect from who
         if (i >= p_i0.and.i <= l_i0+p_ni-1 .and. &
             j >= p_j0.and.j <= l_j0+p_nj-1      ) then
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
                if (i >= pe_gindx(1,kk).and.i <= pe_gindx(2,kk).and. &
                    j >= pe_gindx(3,kk).and.j <= pe_gindx(4,kk))then
                    send_len(kk)=send_len(kk)+1
                endif
             enddo
         endif
      enddo
      enddo
!
! South section
      do j=1-G_haloy,Glb_pil_s
      do i=1+Glb_pil_w,G_ni-Glb_pil_e

         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx,imy,x_a,y_a, &
                          xg_8(1),yg_8(1),h1,h2,1,1)
         imx = min(max(imx-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy = min(max(imy-1,glb_pil_s+1),G_nj-glb_pil_n-3)
! check to collect from who
         if (i >= p_i0.and.i <= l_i0+p_ni-1 .and. &
             j >= p_j0.and.j <= l_j0+p_nj-1      ) then
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
                if (i >= pe_gindx(1,kk).and.i <= pe_gindx(2,kk).and. &
                    j >= pe_gindx(3,kk).and.j <= pe_gindx(4,kk))then
                    send_len(kk)=send_len(kk)+1
                endif
             enddo
         endif
      enddo
      enddo
!
! North section
      do j=G_nj-Glb_pil_n+1,G_nj+G_haloy
      do i=1+Glb_pil_w,G_ni-Glb_pil_e

         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx,imy,x_a,y_a, &
                          xg_8(1),yg_8(1),h1,h2,1,1)
         imx = min(max(imx-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy = min(max(imy-1,glb_pil_s+1),G_nj-glb_pil_n-3)

! check to collect from who
         if (i >= p_i0.and.i <= l_i0+p_ni-1 .and. &
             j >= p_j0.and.j <= l_j0+p_nj-1      ) then
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
                if (i >= pe_gindx(1,kk).and.i <= pe_gindx(2,kk).and. &
                    j >= pe_gindx(3,kk).and.j <= pe_gindx(4,kk))then
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
     Pilp_send_all=0
     Pilp_recv_all=0
     Pilp_sendmaxproc=0
     Pilp_recvmaxproc=0

     do kk=1,Ptopo_numproc
        Pilp_send_all=send_len(kk)+Pilp_send_all
        Pilp_recv_all=recv_len(kk)+Pilp_recv_all

        if (send_len(kk) > 0) Pilp_sendmaxproc=Pilp_sendmaxproc+1
        if (recv_len(kk) > 0) Pilp_recvmaxproc=Pilp_recvmaxproc+1
     enddo
!
!     print *,'Allocate common vectors'
      allocate (Pilp_recvproc(Pilp_recvmaxproc))
      allocate (Pilp_recv_len(Pilp_recvmaxproc))
      allocate (Pilp_recv_adr(Pilp_recvmaxproc))

      allocate (Pilp_sendproc(Pilp_sendmaxproc))
      allocate (Pilp_send_len(Pilp_sendmaxproc))
      allocate (Pilp_send_adr(Pilp_sendmaxproc))
      Pilp_recv_len(:) = 0
      Pilp_send_len(:) = 0
      Pilp_recv_adr(:) = 0
      Pilp_send_adr(:) = 0

!    print*,'Pilp_sendmaxproc=',Pilp_sendmaxproc,'recvmaxproc=',Pilp_recvmaxproc

     ksend=0
     krecv=0
     Pilp_send_all=0
     Pilp_recv_all=0
!
! Fill the lengths and addresses for selected processors to communicate
!
     do kk=1,Ptopo_numproc
        if (send_len(kk) > 0) then
            ksend=ksend+1
            Pilp_sendproc(ksend)=kk
            Pilp_send_len(ksend)=send_len(kk)

            Pilp_send_adr(ksend)= Pilp_send_all
            Pilp_send_all= Pilp_send_all + Pilp_send_len(ksend)
        endif
        if (recv_len(kk) > 0) then
            krecv=krecv+1
            Pilp_recvproc(krecv)=kk
            Pilp_recv_len(krecv)=recv_len(kk)

            Pilp_recv_adr(krecv)= Pilp_recv_all
            Pilp_recv_all= Pilp_recv_all + Pilp_recv_len(krecv)
        endif

     enddo
!    print *,'krecv=',krecv,'Pilp_recvmaxproc=',Pilp_recvmaxproc
!    print *,'ksend=',ksend,'Pilp_sendmaxproc=',Pilp_sendmaxproc

!     print *,'Summary of SCALBC comm procs'
!     do kk=1,Pilp_recvmaxproc
!       print *,'From proc:',Pilp_recvproc(kk),'Pilp_recv_len',Pilp_recv_len(kk),'adr',Pilp_recv_adr(kk)
!     enddo
!     do kk=1,Pilp_sendmaxproc
!       print *,'To proc:',Pilp_sendproc(kk),'Pilp_send_len',Pilp_send_len(kk),'adr',Pilp_send_adr(kk)
!     enddo

!
! Now allocate the vectors needed for sending and receiving each processor
!
      if (Pilp_recv_all > 0) then
          allocate (Pilp_recv_i(Pilp_recv_all))
          allocate (Pilp_recv_j(Pilp_recv_all))
          Pilp_recv_i(:) = 0
          Pilp_recv_j(:) = 0
      endif

      if (Pilp_send_all > 0) then
          allocate (Pilp_send_imx(Pilp_send_all))
          allocate (Pilp_send_imy(Pilp_send_all))
          allocate (Pilp_send_xxr(Pilp_send_all))
          allocate (Pilp_send_yyr(Pilp_send_all))
          allocate (Pilp_send_s1(Pilp_send_all))
          allocate (Pilp_send_s2(Pilp_send_all))
          allocate (Pilp_send_s3(Pilp_send_all))
          allocate (Pilp_send_s4(Pilp_send_all))
          Pilp_send_imx(:) = 0
          Pilp_send_imy(:) = 0
          Pilp_send_xxr(:) = 0.0
          Pilp_send_yyr(:) = 0.0
          Pilp_send_s1(:) = 0.0
          Pilp_send_s2(:) = 0.0
          Pilp_send_s3(:) = 0.0
          Pilp_send_s4(:) = 0.0
      endif
!

      recv_len(:)=0
      send_len(:)=0
!
! SECOND PASS is to initialize the vectors with information for communication
!
! WEST section

      do j=1-G_haloy, G_nj+G_haloy
      do i=1-G_halox,Glb_pil_w
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx,imy,x_a,y_a, &
                          xg_8(1),yg_8(1),h1,h2,1,1)
         imx = min(max(imx-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy = min(max(imy-1,glb_pil_s+1),G_nj-glb_pil_n-3)
! check to collect from who
         if (i >= p_i0.and.i <= l_i0+p_ni-1 .and. &
             j >= p_j0.and.j <= l_j0+p_nj-1      ) then
             do kk=1,Pilp_recvmaxproc
                ki=Pilp_recvproc(kk)
                if (imx >= Ptopo_gindx(1,ki).and.imx <= Ptopo_gindx(2,ki).and. &
                    imy >= Ptopo_gindx(3,ki).and.imy <= Ptopo_gindx(4,ki))then
                    recv_len(kk)=recv_len(kk)+1
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pilp_recv_i(Pilp_recv_adr(kk)+recv_len(kk))=ii
                    Pilp_recv_j(Pilp_recv_adr(kk)+recv_len(kk))=jj
                endif
             enddo
         endif

! check to send to who
         if (imx >= l_i0.and.imx <= l_i0+l_ni-1 .and. &
             imy >= l_j0.and.imy <= l_j0+l_nj-1      ) then
             do kk=1,Pilp_sendmaxproc
                ki=Pilp_sendproc(kk)
                if (i >= pe_gindx(1,ki).and.i <= pe_gindx(2,ki).and. &
                    j >= pe_gindx(3,ki).and.j <= pe_gindx(4,ki))then
                    send_len(kk)=send_len(kk)+1
                    Pilp_send_imx(Pilp_send_adr(kk)+send_len(kk))=imx-l_i0+1
                    Pilp_send_imy(Pilp_send_adr(kk)+send_len(kk))=imy-l_j0+1
                    Pilp_send_xxr(Pilp_send_adr(kk)+send_len(kk))=x_a
                    Pilp_send_yyr(Pilp_send_adr(kk)+send_len(kk))=y_a
                    Pilp_send_s1(Pilp_send_adr(kk)+send_len(kk))=s(1,1)
                    Pilp_send_s2(Pilp_send_adr(kk)+send_len(kk))=s(1,2)
                    Pilp_send_s3(Pilp_send_adr(kk)+send_len(kk))=s(2,1)
                    Pilp_send_s4(Pilp_send_adr(kk)+send_len(kk))=s(2,2)
                endif
             enddo
         endif
      enddo
      enddo
!
!
! East section
      do j=1-G_haloy, G_nj+G_haloy
      do i=G_ni-Glb_pil_e+1,G_ni+G_halox
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx,imy,x_a,y_a, &
                          xg_8(1),yg_8(1),h1,h2,1,1)
         imx = min(max(imx-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy = min(max(imy-1,glb_pil_s+1),G_nj-glb_pil_n-3)
! check to collect from who
         if (i >= p_i0.and.i <= l_i0+p_ni-1 .and. &
             j >= p_j0.and.j <= l_j0+p_nj-1      ) then
             do kk=1,Pilp_recvmaxproc
                ki=Pilp_recvproc(kk)
                if (imx >= Ptopo_gindx(1,ki).and.imx <= Ptopo_gindx(2,ki).and. &
                    imy >= Ptopo_gindx(3,ki).and.imy <= Ptopo_gindx(4,ki))then
                    recv_len(kk)=recv_len(kk)+1
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pilp_recv_i(Pilp_recv_adr(kk)+recv_len(kk))=ii
                    Pilp_recv_j(Pilp_recv_adr(kk)+recv_len(kk))=jj
                endif
             enddo
         endif

! check to send to who
         if (imx >= l_i0.and.imx <= l_i0+l_ni-1 .and. &
             imy >= l_j0.and.imy <= l_j0+l_nj-1      ) then
             do kk=1,Pilp_sendmaxproc
                ki=Pilp_sendproc(kk)
                if (i >= pe_gindx(1,ki).and.i <= pe_gindx(2,ki).and. &
                    j >= pe_gindx(3,ki).and.j <= pe_gindx(4,ki))then
                    send_len(kk)=send_len(kk)+1
                    Pilp_send_imx(Pilp_send_adr(kk)+send_len(kk))=imx-l_i0+1
                    Pilp_send_imy(Pilp_send_adr(kk)+send_len(kk))=imy-l_j0+1
                    Pilp_send_xxr(Pilp_send_adr(kk)+send_len(kk))=x_a
                    Pilp_send_yyr(Pilp_send_adr(kk)+send_len(kk))=y_a
                    Pilp_send_s1(Pilp_send_adr(kk)+send_len(kk))=s(1,1)
                    Pilp_send_s2(Pilp_send_adr(kk)+send_len(kk))=s(1,2)
                    Pilp_send_s3(Pilp_send_adr(kk)+send_len(kk))=s(2,1)
                    Pilp_send_s4(Pilp_send_adr(kk)+send_len(kk))=s(2,2)
                endif
             enddo
         endif
      enddo
      enddo
!
! South section
      do j=1-G_haloy,Glb_pil_s
      do i=1+Glb_pil_w,G_ni-Glb_pil_e

         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx,imy,x_a,y_a, &
                          xg_8(1),yg_8(1),h1,h2,1,1)
         imx = min(max(imx-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy = min(max(imy-1,glb_pil_s+1),G_nj-glb_pil_n-3)
! check to collect from who
         if (i >= p_i0.and.i <= l_i0+p_ni-1 .and. &
             j >= p_j0.and.j <= l_j0+p_nj-1      ) then
             do kk=1,Pilp_recvmaxproc
                ki=Pilp_recvproc(kk)
                if (imx >= Ptopo_gindx(1,ki).and.imx <= Ptopo_gindx(2,ki).and. &
                    imy >= Ptopo_gindx(3,ki).and.imy <= Ptopo_gindx(4,ki))then
                    recv_len(kk)=recv_len(kk)+1
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pilp_recv_i(Pilp_recv_adr(kk)+recv_len(kk))=ii
                    Pilp_recv_j(Pilp_recv_adr(kk)+recv_len(kk))=jj
                endif
             enddo
         endif

! check to send to who
         if (imx >= l_i0.and.imx <= l_i0+l_ni-1 .and. &
             imy >= l_j0.and.imy <= l_j0+l_nj-1      ) then
             do kk=1,Pilp_sendmaxproc
                ki=Pilp_sendproc(kk)
                if (i >= pe_gindx(1,ki).and.i <= pe_gindx(2,ki).and. &
                    j >= pe_gindx(3,ki).and.j <= pe_gindx(4,ki))then
                    send_len(kk)=send_len(kk)+1
                    Pilp_send_imx(Pilp_send_adr(kk)+send_len(kk))=imx-l_i0+1
                    Pilp_send_imy(Pilp_send_adr(kk)+send_len(kk))=imy-l_j0+1
                    Pilp_send_xxr(Pilp_send_adr(kk)+send_len(kk))=x_a
                    Pilp_send_yyr(Pilp_send_adr(kk)+send_len(kk))=y_a
                    Pilp_send_s1(Pilp_send_adr(kk)+send_len(kk))=s(1,1)
                    Pilp_send_s2(Pilp_send_adr(kk)+send_len(kk))=s(1,2)
                    Pilp_send_s3(Pilp_send_adr(kk)+send_len(kk))=s(2,1)
                    Pilp_send_s4(Pilp_send_adr(kk)+send_len(kk))=s(2,2)
                endif
             enddo
         endif
      enddo
      enddo
!
! North section
      do j=G_nj-Glb_pil_n+1,G_nj+G_haloy
      do i=1+Glb_pil_w,G_ni-Glb_pil_e

         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx,imy,x_a,y_a, &
                          xg_8(1),yg_8(1),h1,h2,1,1)
         imx = min(max(imx-1,glb_pil_w+1),G_ni-glb_pil_e-3)
         imy = min(max(imy-1,glb_pil_s+1),G_nj-glb_pil_n-3)

! check to collect from who
         if (i >= p_i0.and.i <= l_i0+p_ni-1 .and. &
             j >= p_j0.and.j <= l_j0+p_nj-1      ) then
             do kk=1,Pilp_recvmaxproc
                ki=Pilp_recvproc(kk)
                if (imx >= Ptopo_gindx(1,ki).and.imx <= Ptopo_gindx(2,ki).and. &
                    imy >= Ptopo_gindx(3,ki).and.imy <= Ptopo_gindx(4,ki))then
                    recv_len(kk)=recv_len(kk)+1
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pilp_recv_i(Pilp_recv_adr(kk)+recv_len(kk))=ii
                    Pilp_recv_j(Pilp_recv_adr(kk)+recv_len(kk))=jj
                endif
             enddo
         endif

! check to send to who
         if (imx >= l_i0.and.imx <= l_i0+l_ni-1 .and. &
             imy >= l_j0.and.imy <= l_j0+l_nj-1      ) then
             do kk=1,Pilp_sendmaxproc
                ki=Pilp_sendproc(kk)
                if (i >= pe_gindx(1,ki).and.i <= pe_gindx(2,ki).and. &
                    j >= pe_gindx(3,ki).and.j <= pe_gindx(4,ki))then
                    send_len(kk)=send_len(kk)+1
                    Pilp_send_imx(Pilp_send_adr(kk)+send_len(kk))=imx-l_i0+1
                    Pilp_send_imy(Pilp_send_adr(kk)+send_len(kk))=imy-l_j0+1
                    Pilp_send_xxr(Pilp_send_adr(kk)+send_len(kk))=x_a
                    Pilp_send_yyr(Pilp_send_adr(kk)+send_len(kk))=y_a
                    Pilp_send_s1(Pilp_send_adr(kk)+send_len(kk))=s(1,1)
                    Pilp_send_s2(Pilp_send_adr(kk)+send_len(kk))=s(1,2)
                    Pilp_send_s3(Pilp_send_adr(kk)+send_len(kk))=s(2,1)
                    Pilp_send_s4(Pilp_send_adr(kk)+send_len(kk))=s(2,2)
                endif
             enddo
         endif
      enddo
      enddo
!Check receive lengths from each processor
!     do ki=1,Pilp_recvmaxproc
!        kk=Pilp_recvproc(ki)
!        if (Ptopo_couleur == 0) then
!            kkproc = kk+Ptopo_numproc-1
!        else
!            kkproc = kk -1
!        endif
!    write(6,1000) 'Pilp_recv_len',kkproc,Pilp_recv_len(kk),Pilp_recv_adr(kk)
!   enddo
!Check send lengths to each processor

!     do ki=1,Pilp_sendmaxproc
!        kk=Pilp_sendproc(ki)
!        if (Ptopo_couleur == 0) then
!            kkproc = kk+Ptopo_numproc-1
!        else
!            kkproc = kk -1
!        endif
! write(6,1000) 'Pilp_send_len',kkproc,Pilp_send_len(kk),Pilp_send_adr(kk)
!     enddo
      deallocate (recv_len,send_len)

 1000 format(a15,i3,'=',i5,'bytes, addr=',i5)
 1001 format(a15,i3,'=',i4,'bytes   i:', i3,' j:',i3)


!
      return
      end

