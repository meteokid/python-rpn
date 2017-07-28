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
!***s/r yyg_initblenbc2 - to initialize communication pattern for cubic
!                            blending of scalar fields
!
      Subroutine yyg_initblenbc2()
       use tdpack
      use glb_ld
      use glb_pil
      use ptopo
      implicit none
#include <arch_specific.hf>
!
!author
!           Abdessamad Qaddouri/ V.lee - September 2011
!  PLEASE consult Abdessamad or Vivian before modifying this routine.
!
#include "yyg_bln.cdk"

      integer err,Ndim,i,j,k,imx,imy,kk,ii,jj,ki,ksend,krecv
      integer kkproc,xmin,xmax,ymin,ymax
      integer, dimension (:), pointer :: recv_len,send_len
      real*8  xx_8(G_ni,G_nj),yy_8(G_ni,G_nj)
      real*8  xg_8(1-G_ni:2*G_ni),yg_8(1-G_nj:2*G_nj)
      real*8  t,p,s(2,2),h1,h2
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
      xmin=1-G_ni
      xmax=2*G_ni
      ymin=1-G_nj
      ymax=2*G_nj

!Delta xg, yg is not identical between xg(i) and xg(i+1)
!h1, h2 used in this routine is ok as it is a close estimate for
!creating YY pattern exchange and it works on the global tile
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

      do j=1+glb_pil_s, G_nj-glb_pil_n
      do i=1+glb_pil_w, G_ni-glb_pil_e
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise_blend(imx,imy,x_a,y_a, &
                          xg_8,yg_8,xmin,xmax,ymin,ymax,h1,h2)

! check if this point can be found in the other grid
!        if (imx.ge.1+glb_pil_w .and. imx.le.G_ni-glb_pil_e .and. &
!            imy.ge.1+glb_pil_s .and. imy.le.G_nj-glb_pil_n) then
! It is important to do this check before min-max
!   (Imx,Imy )could be zero or negatif or 1<(Imx,Imy )<(G_ni,G_nj)
         if (imx.gt.1+glb_pil_w .and. imx.lt.G_ni-glb_pil_e .and. &
             imy.gt.1+glb_pil_s .and. imy.lt.G_nj-glb_pil_n) then
             imx = min(max(imx-1,glb_pil_w+1),G_ni-glb_pil_e-3)
             imy = min(max(imy-1,glb_pil_s+1),G_nj-glb_pil_n-3)

! check to collect from who
             if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
                 j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
                 do kk=1,Ptopo_numproc
                    if (imx.ge.Ptopo_gindx(1,kk).and. &
                        imx.le.Ptopo_gindx(2,kk).and. &
                        imy.ge.Ptopo_gindx(3,kk).and. &
                        imy.le.Ptopo_gindx(4,kk)) then
                        recv_len(kk)=recv_len(kk)+1
                    endif
                 enddo
             endif

! check to send to who
             if (imx.ge.l_i0.and.imx.le.l_i0+l_ni-1 .and. &
                 imy.ge.l_j0.and.imy.le.l_j0+l_nj-1      ) then
                 do kk=1,Ptopo_numproc
                    if (i  .ge.Ptopo_gindx(1,kk).and. &
                        i  .le.Ptopo_gindx(2,kk).and. &
                        j  .ge.Ptopo_gindx(3,kk).and. &
                        j  .le.Ptopo_gindx(4,kk)     )then
                        send_len(kk)=send_len(kk)+1
                    endif
                 enddo
             endif
         endif
      enddo
      enddo
!
!
! Obtain sum of elements to send and receive for each processor
! and the total memory needed to store and receive for each processor
!
      Bln_send_all=0
      Bln_recv_all=0
      Bln_sendmaxproc=0
      Bln_recvmaxproc=0

      do kk=1,Ptopo_numproc
         Bln_send_all=send_len(kk)+Bln_send_all
         Bln_recv_all=recv_len(kk)+Bln_recv_all
         if (send_len(kk).gt.0) Bln_sendmaxproc=Bln_sendmaxproc+1
         if (recv_len(kk).gt.0) Bln_recvmaxproc=Bln_recvmaxproc+1
      enddo
!
!     print *,'Allocate common vectors'
      allocate (Bln_recvproc(Bln_recvmaxproc))
      allocate (Bln_recv_len(Bln_recvmaxproc))
      allocate (Bln_recv_adr(Bln_recvmaxproc))

      allocate (Bln_sendproc(Bln_sendmaxproc))
      allocate (Bln_send_len(Bln_sendmaxproc))
      allocate (Bln_send_adr(Bln_sendmaxproc))
      Bln_recv_len(:) = 0
      Bln_send_len(:) = 0
      Bln_recv_adr(:) = 0
      Bln_send_adr(:) = 0

!     print*,'Bln_sendmaxproc=',Bln_sendmaxproc,'recvmaxproc=',Bln_recvmaxproc

      ksend=0
      krecv=0
      Bln_send_all=0
      Bln_recv_all=0
!
! Fill the lengths and addresses for selected processors to communicate
!
      do kk=1,Ptopo_numproc
         if (send_len(kk).gt.0) then
             ksend=ksend+1
             Bln_sendproc(ksend)=kk
             Bln_send_len(ksend)=send_len(kk)

             Bln_send_adr(ksend)= Bln_send_all
             Bln_send_all= Bln_send_all + Bln_send_len(ksend)
         endif
         if (recv_len(kk).gt.0) then
             krecv=krecv+1
             Bln_recvproc(krecv)=kk
             Bln_recv_len(krecv)=recv_len(kk)

             Bln_recv_adr(krecv)= Bln_recv_all
             Bln_recv_all= Bln_recv_all + Bln_recv_len(krecv)
         endif

      enddo
!     print *,'krecv=',krecv,'Bln_recvmaxproc=',Bln_recvmaxproc
!     print *,'ksend=',ksend,'Bln_sendmaxproc=',Bln_sendmaxproc

!     print *,'Summary of BLN comm procs'
!     do kk=1,Bln_recvmaxproc
!       print *,'From proc:',Bln_recvproc(kk),'Bln_recv_len',Bln_recv_len(kk),'adr',Bln_recv_adr(kk)
!     enddo
!     do kk=1,Bln_sendmaxproc
!       print *,'To proc:',Bln_sendproc(kk),'Bln_send_len',Bln_send_len(kk),'adr',Bln_send_adr(kk)
!     enddo

!
! Now allocate the vectors needed for sending and receiving each processor
!
      if (Bln_recv_all.gt.0) then
          allocate (Bln_recv_i(Bln_recv_all))
          allocate (Bln_recv_j(Bln_recv_all))
          Bln_recv_i(:) = 0
          Bln_recv_j(:) = 0
      endif

      if (Bln_send_all.gt.0) then
          allocate (Bln_send_imx(Bln_send_all))
          allocate (Bln_send_imy(Bln_send_all))
          allocate (Bln_send_xxr(Bln_send_all))
          allocate (Bln_send_yyr(Bln_send_all))
          allocate (Bln_send_s1(Bln_send_all))
          allocate (Bln_send_s2(Bln_send_all))
          allocate (Bln_send_s3(Bln_send_all))
          allocate (Bln_send_s4(Bln_send_all))
          Bln_send_imx(:) = 0
          Bln_send_imy(:) = 0
          Bln_send_xxr(:) = 0.0
          Bln_send_yyr(:) = 0.0
          Bln_send_s1(:) = 0.0
          Bln_send_s2(:) = 0.0
          Bln_send_s3(:) = 0.0
          Bln_send_s4(:) = 0.0
      endif
!

      recv_len(:)=0
      send_len(:)=0
!
! SECOND PASS is to initialize the vectors with information for communication
!

      do j=1+glb_pil_s, G_nj-glb_pil_n
      do i=1+glb_pil_w, G_ni-glb_pil_e
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise_blend(imx,imy,x_a,y_a, &
                          xg_8,yg_8,xmin,xmax,ymin,ymax,h1,h2)

! check if this point can be found in the other grid
!        if (imx.ge.1+glb_pil_w .and. imx.le.G_ni-glb_pil_e .and. &
!            imy.ge.1+glb_pil_s .and. imy.le.G_nj-glb_pil_n) then
! It is important to do this check before min-max
!   (Imx,Imy )could be zero or negatif or 1<(Imx,Imy )<(G_ni,G_nj)

         if (imx.gt.1+glb_pil_w .and. imx.lt.G_ni-glb_pil_e .and. &
             imy.gt.1+glb_pil_s .and. imy.lt.G_nj-glb_pil_n) then
             imx = min(max(imx-1,glb_pil_w+1),G_ni-glb_pil_e-3)
             imy = min(max(imy-1,glb_pil_s+1),G_nj-glb_pil_n-3)

! check to collect from who
             if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
                 j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
                 do kk=1,Bln_recvmaxproc
                    ki=Bln_recvproc(kk)
                    if (imx.ge.Ptopo_gindx(1,ki).and. &
                        imx.le.Ptopo_gindx(2,ki).and. &
                        imy.ge.Ptopo_gindx(3,ki).and. &
                        imy.le.Ptopo_gindx(4,ki) )then
                        recv_len(kk)=recv_len(kk)+1
                        ii=i-l_i0+1
                        jj=j-l_j0+1
                        Bln_recv_i(Bln_recv_adr(kk)+recv_len(kk))=ii
                        Bln_recv_j(Bln_recv_adr(kk)+recv_len(kk))=jj
                    endif
                 enddo
             endif

! check to send to who
             if (imx.ge.l_i0.and.imx.le.l_i0+l_ni-1 .and. &
                 imy.ge.l_j0.and.imy.le.l_j0+l_nj-1      ) then
                 do kk=1,Bln_sendmaxproc
                    ki=Bln_sendproc(kk)
                    if (i  .ge.Ptopo_gindx(1,ki).and. &
                        i  .le.Ptopo_gindx(2,ki).and. &
                        j  .ge.Ptopo_gindx(3,ki).and. &
                        j  .le.Ptopo_gindx(4,ki))then
                        send_len(kk)=send_len(kk)+1
                        Bln_send_imx(Bln_send_adr(kk)+send_len(kk))=imx-l_i0+1
                        Bln_send_imy(Bln_send_adr(kk)+send_len(kk))=imy-l_j0+1
                        Bln_send_xxr(Bln_send_adr(kk)+send_len(kk))=x_a
                        Bln_send_yyr(Bln_send_adr(kk)+send_len(kk))=y_a
                        Bln_send_s1(Bln_send_adr(kk)+send_len(kk))=s(1,1)
                        Bln_send_s2(Bln_send_adr(kk)+send_len(kk))=s(1,2)
                        Bln_send_s3(Bln_send_adr(kk)+send_len(kk))=s(2,1)
                        Bln_send_s4(Bln_send_adr(kk)+send_len(kk))=s(2,2)
                    endif
                 enddo
             endif
         endif
      enddo
      enddo
!
!
!Check receive lengths from each processor
!     do ki=1,Bln_recvmaxproc
!        kk=Bln_recvproc(ki)
!        if (Ptopo_couleur.eq.0) then
!            kkproc = kk+Ptopo_numproc-1
!        else
!            kkproc = kk -1
!        endif
!        write(6,1000) 'Bln_recv_len',kkproc,Bln_recv_len(kk),Bln_recv_adr(kk)
!     enddo
!Check send lengths to each processor

!     do ki=1,Bln_sendmaxproc
!        kk=Bln_sendproc(ki)
!        if (Ptopo_couleur.eq.0) then
!            kkproc = kk+Ptopo_numproc-1
!        else
!            kkproc = kk -1
!        endif
!        write(6,1000) 'Bln_send_len',kkproc,Bln_send_len(kk),Bln_send_adr(kk)
!     enddo
      deallocate (recv_len,send_len)

 1000 format(a15,i3,'=',i5,'bytes, addr=',i5)
 1001 format(a15,i3,'=',i4,'bytes   i:', i3,' j:',i3)


!
      return
      end

