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
!       -vectorial interpolation (use U,V of other grid to find V value here)
!


      Subroutine yyg_initvecbc2()
      use tdpack
      implicit none
#include <arch_specific.hf>
!
!author
!     Abdessamad Qaddouri/V.Lee - October 2009
!  PLEASE consult Abdessamad or Vivian before modifying this routine.
!revision
!  v4.8  V.Lee - Correction for limiting range in point found on other grid
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
      integer, dimension (:), pointer :: recv_len,send_len
      real*8  xx_8(G_ni,G_njv),yy_8(G_ni,G_njv)
      real*8  xg_8(1-G_ni:2*G_ni),yg_8(1-G_nj:2*G_nj)
      real*8  xgu_8(1-G_ni:2*G_ni-1),ygv_8(1-G_nj:2*G_nj-1)
      real*8  t,p,s(2,2),h1,h2
      real*8  x_d,y_d,x_a,y_a   
      real*8 TWO_8
      parameter( TWO_8   = 2.0d0 )
!
!
!     Localise could get point way outside of the actual grid in search
!     So extend all global arrays: xg_8,yg_8, xgu_8,ygv_8
!

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

       do i=1-G_ni,2*G_ni-1
       xgu_8(i)=0.5D0 *(xg_8(i+1)+xg_8(i))
       enddo
       do j=1-G_nj,2*G_nj-1
       ygv_8(j)= 0.5D0*(yg_8(j+1)+yg_8(j))
       enddo
!
      do j=1,G_njv
      do i=1,G_ni
         xx_8(i,j)=xg_8(i)
         yy_8(i,j)=ygv_8(j)
      enddo
      enddo

!Delta xg, yg is not identical between xg(i) and xg(i+1)
!h1, h2 used in this routine is ok as it is a close estimate for
!creating YY pattern exchange and it works on the global tile

      h1=xg_8(2)-xg_8(1)
      h2=yg_8(2)-yg_8(1)
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
                          xgu_8(1),yg_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          xg_8(1),ygv_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_nj -glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_ni -glb_pil_e-3)
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
                    recv_len(kk)=recv_len(kk)+1
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
                    send_len(kk)=send_len(kk)+1
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
                          xgu_8(1),yg_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          xg_8(1),ygv_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_nj -glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_ni -glb_pil_e-3)
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
                    recv_len(kk)=recv_len(kk)+1
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
                    send_len(kk)=send_len(kk)+1
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
                          xgu_8(1),yg_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          xg_8(1),ygv_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_nj -glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_ni -glb_pil_e-3)
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
                    recv_len(kk)=recv_len(kk)+1
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
                    send_len(kk)=send_len(kk)+1
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
                          xgu_8(1),yg_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          xg_8(1),ygv_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_nj -glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_ni -glb_pil_e-3)
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
                    recv_len(kk)=recv_len(kk)+1
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
     Pil_vsend_all=0
     Pil_vrecv_all=0
     Pil_vsendmaxproc=0
     Pil_vrecvmaxproc=0

     do kk=1,Ptopo_numproc
        Pil_vsend_all=send_len(kk)+Pil_vsend_all
        Pil_vrecv_all=recv_len(kk)+Pil_vrecv_all

        if (send_len(kk).gt.0) Pil_vsendmaxproc=Pil_vsendmaxproc+1
        if (recv_len(kk).gt.0) Pil_vrecvmaxproc=Pil_vrecvmaxproc+1
     enddo
!
!     print *,'Allocate common vectors'
      allocate (Pil_vrecvproc(Pil_vrecvmaxproc))
      allocate (Pil_vrecv_len(Pil_vrecvmaxproc))
      allocate (Pil_vrecv_adr(Pil_vrecvmaxproc))

      allocate (Pil_vsendproc(Pil_vsendmaxproc))
      allocate (Pil_vsend_len(Pil_vsendmaxproc))
      allocate (Pil_vsend_adr(Pil_vsendmaxproc))
      Pil_vrecv_len(:) = 0
      Pil_vsend_len(:) = 0
      Pil_vrecv_adr(:) = 0
      Pil_vsend_adr(:) = 0

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

            Pil_vsend_adr(ksend)= Pil_vsend_all
            Pil_vsend_all= Pil_vsend_all + Pil_vsend_len(ksend)
        endif
        if (recv_len(kk).gt.0) then
            krecv=krecv+1
            Pil_vrecvproc(krecv)=kk
            Pil_vrecv_len(krecv)=recv_len(kk)

            Pil_vrecv_adr(krecv)= Pil_vrecv_all
            Pil_vrecv_all= Pil_vrecv_all + Pil_vrecv_len(krecv)
        endif

     enddo
!    print *,'krecv=',krecv,'Pil_vrecvmaxproc=',Pil_vrecvmaxproc
!    print *,'ksend=',ksend,'Pil_vsendmaxproc=',Pil_vsendmaxproc

!     print *,'Summary of comm procs'
!     do kk=1,Pil_vrecvmaxproc
!  print *,'From proc:',Pil_vrecvproc(kk),'Pil_vrecv_len',Pil_vrecv_len(kk),'adr',Pil_vrecv_adr(kk)
!     enddo
!     do kk=1,Pil_vsendmaxproc
!       print *,'To proc:',Pil_vsendproc(kk),'Pil_vsend_len',Pil_vsend_len(kk),'adr',Pil_vsend_adr(kk)
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

      recv_len(:)=0
      send_len(:)=0
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
                          xgu_8(1),yg_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          xg_8(1),ygv_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_nj -glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_ni -glb_pil_e-3)
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
                    recv_len(kk)=recv_len(kk)+1
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil_vrecv_i(Pil_vrecv_adr(kk)+recv_len(kk))=ii
                    Pil_vrecv_j(Pil_vrecv_adr(kk)+recv_len(kk))=jj
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
                    send_len(kk)=send_len(kk)+1
                    Pil_vsend_imx1(Pil_vsend_adr(kk)+send_len(kk))=imx1-l_i0+1
                    Pil_vsend_imy1(Pil_vsend_adr(kk)+send_len(kk))=imy1-l_j0+1
                    Pil_vsend_imx2(Pil_vsend_adr(kk)+send_len(kk))=imx2-l_i0+1
                    Pil_vsend_imy2(Pil_vsend_adr(kk)+send_len(kk))=imy2-l_j0+1
                    Pil_vsend_xxr(Pil_vsend_adr(kk)+send_len(kk))=x_a
                    Pil_vsend_yyr(Pil_vsend_adr(kk)+send_len(kk))=y_a
                    Pil_vsend_s1(Pil_vsend_adr(kk)+send_len(kk))=s(2,1)
                    Pil_vsend_s2(Pil_vsend_adr(kk)+send_len(kk))=s(2,2)
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
                          xgu_8(1),yg_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          xg_8(1),ygv_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_nj -glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_ni -glb_pil_e-3)
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
                    recv_len(kk)=recv_len(kk)+1
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil_vrecv_i(Pil_vrecv_adr(kk)+recv_len(kk))=ii
                    Pil_vrecv_j(Pil_vrecv_adr(kk)+recv_len(kk))=jj
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
                    send_len(kk)=send_len(kk)+1
                    Pil_vsend_imx1(Pil_vsend_adr(kk)+send_len(kk))=imx1-l_i0+1
                    Pil_vsend_imy1(Pil_vsend_adr(kk)+send_len(kk))=imy1-l_j0+1
                    Pil_vsend_imx2(Pil_vsend_adr(kk)+send_len(kk))=imx2-l_i0+1
                    Pil_vsend_imy2(Pil_vsend_adr(kk)+send_len(kk))=imy2-l_j0+1
                    Pil_vsend_xxr(Pil_vsend_adr(kk)+send_len(kk))=x_a
                    Pil_vsend_yyr(Pil_vsend_adr(kk)+send_len(kk))=y_a
                    Pil_vsend_s1(Pil_vsend_adr(kk)+send_len(kk))=s(2,1)
                    Pil_vsend_s2(Pil_vsend_adr(kk)+send_len(kk))=s(2,2)
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
                          xgu_8(1),yg_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a,  &
                          xg_8(1),ygv_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_nj -glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_ni -glb_pil_e-3)
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
                    recv_len(kk)=recv_len(kk)+1
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil_vrecv_i(Pil_vrecv_adr(kk)+recv_len(kk))=ii
                    Pil_vrecv_j(Pil_vrecv_adr(kk)+recv_len(kk))=jj
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
                    send_len(kk)=send_len(kk)+1
                    Pil_vsend_imx1(Pil_vsend_adr(kk)+send_len(kk))=imx1-l_i0+1
                    Pil_vsend_imy1(Pil_vsend_adr(kk)+send_len(kk))=imy1-l_j0+1
                    Pil_vsend_imx2(Pil_vsend_adr(kk)+send_len(kk))=imx2-l_i0+1
                    Pil_vsend_imy2(Pil_vsend_adr(kk)+send_len(kk))=imy2-l_j0+1
                    Pil_vsend_xxr(Pil_vsend_adr(kk)+send_len(kk))=x_a
                    Pil_vsend_yyr(Pil_vsend_adr(kk)+send_len(kk))=y_a
                    Pil_vsend_s1(Pil_vsend_adr(kk)+send_len(kk))=s(2,1)
                    Pil_vsend_s2(Pil_vsend_adr(kk)+send_len(kk))=s(2,2)
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
                          xgu_8(1),yg_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          xg_8(1),ygv_8(1),h1,h2,1,1)
         imx1 = min(max(imx1-1,glb_pil_w+1),G_niu-glb_pil_e-3)
         imy1 = min(max(imy1-1,glb_pil_s+1),G_nj -glb_pil_n-3)
         imx2 = min(max(imx2-1,glb_pil_w+1),G_ni -glb_pil_e-3)
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
                    recv_len(kk)=recv_len(kk)+1
                    ii=i-l_i0+1
                    jj=j-l_j0+1
                    Pil_vrecv_i(Pil_vrecv_adr(kk)+recv_len(kk))=ii
                    Pil_vrecv_j(Pil_vrecv_adr(kk)+recv_len(kk))=jj
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
                    send_len(kk)=send_len(kk)+1
                    Pil_vsend_imx1(Pil_vsend_adr(kk)+send_len(kk))=imx1-l_i0+1
                    Pil_vsend_imy1(Pil_vsend_adr(kk)+send_len(kk))=imy1-l_j0+1
                    Pil_vsend_imx2(Pil_vsend_adr(kk)+send_len(kk))=imx2-l_i0+1
                    Pil_vsend_imy2(Pil_vsend_adr(kk)+send_len(kk))=imy2-l_j0+1
                    Pil_vsend_xxr(Pil_vsend_adr(kk)+send_len(kk))=x_a
                    Pil_vsend_yyr(Pil_vsend_adr(kk)+send_len(kk))=y_a
                    Pil_vsend_s1(Pil_vsend_adr(kk)+send_len(kk))=s(2,1)
                    Pil_vsend_s2(Pil_vsend_adr(kk)+send_len(kk))=s(2,2)
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
!    write(6,1000) 'Pil_vrecv_len',kkproc,Pil_vrecv_len(kk),Pil_vrecv_adr(kk)
!   enddo

!Check send lengths to each processor

!     do ki=1,Pil_vsendmaxproc
!        kk=Pil_vsendproc(ki)
!        if (Ptopo_couleur.eq.0) then
!            kkproc = kk+Ptopo_numproc-1
!        else
!            kkproc = kk -1
!        endif
! write(6,1000) 'Pil_vsend_len',kkproc,Pil_vsend_len(kk),Pil_vsend_adr(kk)
!     enddo
      deallocate (recv_len,send_len)

 1000 format(a15,i3,'=',i5,'bytes, addr=',i5)
 1001 format(a15,i3,'=',i4,'bytes   i:', i3,' j:',i3)
       
!
      return
      end

