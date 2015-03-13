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
!***s/r yyg_initblenv - to initialize communication pattern for V field 
!
      Subroutine yyg_initblenv()
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
#include "yyg_blnv.cdk"

      integer err,Ndim,i,j,k,kk,ii,jj,ki,ksend,krecv
      integer imx1,imx2
      integer imy1,imy2
      integer kkproc,adr
      integer, dimension (:), pointer :: recv_len,send_len
      real*8  xx_8(G_ni,G_njv),yy_8(G_ni,G_njv)
      real*8  xgu_8(1-G_ni:2*G_ni-1),ygv_8(1-G_nj:2*G_nj-1)
      real*8  t,p,s(2,2),h1,h2
      real*8  x_d,y_d,x_a,y_a   
!
!     Get global xgu,ygv,xx,yy
       do i=1-G_ni,2*G_ni-1
       xgu_8(i)=0.5D0 *(G_xg_8(i+1)+G_xg_8(i))
       enddo
       do j=1-G_nj,2*G_nj-1
       ygv_8(j)= 0.5D0*(G_yg_8(j+1)+G_yg_8(j))
       enddo
!      do i=1,G_ni-1
!      xgu_8(i)=0.5D0 *(G_xg_8(i+1)+G_xg_8(i))
!      enddo
!      do j=1,G_njv
!      ygv_8(j)= 0.5D0*(G_yg_8(j+1)+G_yg_8(j))
!      enddo
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
      do j=1+glb_pil_s, G_njv-glb_pil_n
      do i=1+glb_pil_w, G_ni-glb_pil_e
!        V vector
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx1,imy1,x_a,y_a, &
                          G_xg_8(1),ygv_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          xgu_8(1),G_yg_8(1),h1,h2,1,1)


! check if this point can be found in the other grid
! It is important to do this check before min-max
!   (Imx,Imy )could be zero or negatif or 1<(Imx,Imy )<(G_ni,G_nj)

         if (imx1.gt.1+glb_pil_w .and. imx1.lt.G_ni-glb_pil_e .and. &
             imy1.gt.1+glb_pil_s .and. imy1.lt.G_njv-glb_pil_n  .and. &
             imx2.gt.1+glb_pil_w .and. imx2.lt.G_ni-glb_pil_e .and. &
             imy2.gt.1+glb_pil_s .and. imy2.lt.G_njv-glb_pil_n) then

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
!
! Obtain sum of elements to send and receive for each processor
! and the total memory needed to store and receive for each processor
!
     Bln_vsend_all=0
     Bln_vrecv_all=0
     Bln_vsendmaxproc=0
     Bln_vrecvmaxproc=0

     do kk=1,Ptopo_numproc
        Bln_vsend_all=send_len(kk)+Bln_vsend_all
        Bln_vrecv_all=recv_len(kk)+Bln_vrecv_all

        if (send_len(kk).gt.0) Bln_vsendmaxproc=Bln_vsendmaxproc+1
        if (recv_len(kk).gt.0) Bln_vrecvmaxproc=Bln_vrecvmaxproc+1
     enddo
!
!     print *,'Allocate common vectors'
      allocate (Bln_vrecvproc(Bln_vrecvmaxproc))
      allocate (Bln_vrecv_len(Bln_vrecvmaxproc))
      allocate (Bln_vrecv_adr(Bln_vrecvmaxproc))

      allocate (Bln_vsendproc(Bln_vsendmaxproc))
      allocate (Bln_vsend_len(Bln_vsendmaxproc))
      allocate (Bln_vsend_adr(Bln_vsendmaxproc))
      Bln_vrecv_len(:) = 0
      Bln_vsend_len(:) = 0
      Bln_vrecv_adr(:) = 0
      Bln_vsend_adr(:) = 0

!    print*,'Bln_vsendmaxproc=',Bln_vsendmaxproc,'recvmaxproc=',Bln_vrecvmaxproc

     ksend=0
     krecv=0
     Bln_vsend_all=0
     Bln_vrecv_all=0
!
! Fill the lengths and addresses for selected processors to communicate
!
     do kk=1,Ptopo_numproc
        if (send_len(kk).gt.0) then
            ksend=ksend+1
            Bln_vsendproc(ksend)=kk
            Bln_vsend_len(ksend)=send_len(kk)

            Bln_vsend_adr(ksend)= Bln_vsend_all
            Bln_vsend_all= Bln_vsend_all + Bln_vsend_len(ksend)
        endif
        if (recv_len(kk).gt.0) then
            krecv=krecv+1
            Bln_vrecvproc(krecv)=kk
            Bln_vrecv_len(krecv)=recv_len(kk)

            Bln_vrecv_adr(krecv)= Bln_vrecv_all
            Bln_vrecv_all= Bln_vrecv_all + Bln_vrecv_len(krecv)
        endif

     enddo
!    print *,'krecv=',krecv,'Bln_vrecvmaxproc=',Bln_vrecvmaxproc
!    print *,'ksend=',ksend,'Bln_vsendmaxproc=',Bln_vsendmaxproc

!     print *,'Summary of comm procs'
!     do kk=1,Bln_vrecvmaxproc
!  print *,'From proc:',Bln_vrecvproc(kk),'Bln_vrecv_len',Bln_vrecvw_len(kk),Bln_vrecve_len(kk),Bln_vrecvs_len(kk),Bln_vrecvn_len(kk),'adr',Bln_vrecvw_adr(kk),Bln_vrecve_adr(kk),Bln_vrecvs_adr(kk),Bln_vrecvn_adr(kk)
!     enddo
!     do kk=1,Bln_vsendmaxproc
!       print *,'To proc:',Bln_vsendproc(kk),'Bln_vsend_len',Bln_vsendw_len(kk),Bln_vsende_len(kk),Bln_vsends_len(kk),Bln_vsendn_len(kk),'adr',Bln_vsendw_adr(kk),Bln_vsende_adr(kk),Bln_vsends_adr(kk),Bln_vsendn_adr(kk)
!     enddo

!
! Now allocate the vectors needed for sending and receiving each processor
!
      if (Bln_vrecv_all.gt.0) then
          allocate (Bln_vrecv_i(Bln_vrecv_all))
          allocate (Bln_vrecv_j(Bln_vrecv_all))
          Bln_vrecv_i(:) = 0
          Bln_vrecv_j(:) = 0
      endif

      if (Bln_vsend_all.gt.0) then
          allocate (Bln_vsend_imx1(Bln_vsend_all))
          allocate (Bln_vsend_imy1(Bln_vsend_all))
          allocate (Bln_vsend_imx2(Bln_vsend_all))
          allocate (Bln_vsend_imy2(Bln_vsend_all))
          allocate (Bln_vsend_xxr(Bln_vsend_all))
          allocate (Bln_vsend_yyr(Bln_vsend_all))
          allocate (Bln_vsend_s1(Bln_vsend_all))
          allocate (Bln_vsend_s2(Bln_vsend_all))
          Bln_vsend_imx1(:) = 0
          Bln_vsend_imy1(:) = 0
          Bln_vsend_imx2(:) = 0
          Bln_vsend_imy2(:) = 0
          Bln_vsend_xxr(:) = 0.0
          Bln_vsend_yyr(:) = 0.0
          Bln_vsend_s1(:) = 0.0
          Bln_vsend_s2(:) = 0.0
      endif
!

      recv_len(:)=0
      send_len(:)=0
!
! SECOND PASS is to initialize the vectors with information for communication
!
! WEST section
!
      do j=1+glb_pil_s, G_njv-glb_pil_n
      do i=1+glb_pil_w, G_ni-glb_pil_e
!        V vector
         x_d=xx_8(i,j)-acos(-1.D0)
         y_d=yy_8(i,j)
         call smat(s,x_a,y_a,x_d,y_d)
         x_a=x_a+(acos(-1.D0))
         call localise(imx1,imy1,x_a,y_a, &
                          G_xg_8(1),ygv_8(1),h1,h2,1,1)
         call localise(imx2,imy2,x_a,y_a, &
                          xgu_8(1),G_yg_8(1),h1,h2,1,1)


! check if this point can be found in the other grid
! It is important to do this check before min-max
!   (Imx,Imy )could be zero or negatif or 1<(Imx,Imy )<(G_ni,G_nj)

         if (imx1.gt.1+glb_pil_w .and. imx1.lt.G_ni-glb_pil_e .and. &
             imy1.gt.1+glb_pil_s .and. imy1.lt.G_njv-glb_pil_n  .and. &
             imx2.gt.1+glb_pil_w .and. imx2.lt.G_ni-glb_pil_e .and. &
             imy2.gt.1+glb_pil_s .and. imy2.lt.G_njv-glb_pil_n) then

             imx1 = min(max(imx1-1,glb_pil_w+1),G_ni-glb_pil_e-3)
             imy1 = min(max(imy1-1,glb_pil_s+1),G_njv-glb_pil_n-3)
             imx2 = min(max(imx2-1,glb_pil_w+1),G_ni-glb_pil_e-3)
             imy2 = min(max(imy2-1,glb_pil_s+1),G_njv-glb_pil_n-3)

!

! check to collect from who
             if (i  .ge.l_i0.and.i  .le.l_i0+l_ni-1 .and. &
                 j  .ge.l_j0.and.j  .le.l_j0+l_nj-1      ) then
                 do kk=1,Bln_vrecvmaxproc
                    ki=Bln_vrecvproc(kk)
                    if (max(imx1,imx2).ge.Ptopo_gindx(1,ki).and. &
                        max(imx1,imx2).le.Ptopo_gindx(2,ki).and. &
                        max(imy1,imy2).ge.Ptopo_gindx(3,ki).and. &
                        max(imy1,imy2).le.Ptopo_gindx(4,ki) ) then
                        recv_len(kk)=recv_len(kk)+1
                        adr=Bln_vrecv_adr(kk)+recv_len(kk)
                        ii=i-l_i0+1
                        jj=j-l_j0+1
                        Bln_vrecv_i(adr)=ii
                        Bln_vrecv_j(adr)=jj
                    endif
                 enddo       
             endif

! check to send to who
             if (max(imx1,imx2).ge.l_i0.and.         &
                 max(imx1,imx2).le.l_i0+l_ni-1 .and. &
                 max(imy1,imy2).ge.l_j0.and.         &
                 max(imy1,imy2).le.l_j0+l_nj-1) then
                 do kk=1,Bln_vsendmaxproc
                    ki=Bln_vsendproc(kk)
                    if (i  .ge.Ptopo_gindx(1,ki).and. &
                        i  .le.Ptopo_gindx(2,ki).and. &
                        j  .ge.Ptopo_gindx(3,ki).and. &
                        j  .le.Ptopo_gindx(4,ki)     )then
                        send_len(kk)=send_len(kk)+1
                        adr=Bln_vsend_adr(kk)+send_len(kk)
                        Bln_vsend_imx1(adr)=imx1-l_i0+1
                        Bln_vsend_imy1(adr)=imy1-l_j0+1
                        Bln_vsend_imx2(adr)=imx2-l_i0+1
                        Bln_vsend_imy2(adr)=imy2-l_j0+1
                        Bln_vsend_xxr(adr)=x_a
                        Bln_vsend_yyr(adr)=y_a
                        Bln_vsend_s1(adr)=s(2,2)
                        Bln_vsend_s2(adr)=s(2,1)
                    endif
                 enddo       
             endif
         endif
      enddo   
      enddo   
!
!
!Check receive lengths from each processor
!     do ki=1,Bln_vrecvmaxproc
!        kk=Bln_vrecvproc(ki)
!        if (Ptopo_couleur.eq.0) then
!            kkproc = kk+Ptopo_numproc-1
!        else
!            kkproc = kk -1
!        endif
!    write(6,1000) 'Bln_vrecvw_len',kkproc,Bln_vrecvw_len(kk),Bln_vrecvw_adr(kk)
!    write(6,1000) 'Bln_vrecve_len',kkproc,Bln_vrecve_len(kk),Bln_vrecve_adr(kk)
!    write(6,1000) 'Bln_vrecvs_len',kkproc,Bln_vrecvs_len(kk),Bln_vrecvs_adr(kk)
!    write(6,1000) 'Bln_vrecvn_len',kkproc,Bln_vrecvn_len(kk),Bln_vrecvn_adr(kk)
!   enddo

!Check send lengths to each processor

!     do ki=1,Bln_vsendmaxproc
!        kk=Bln_vsendproc(ki)
!        if (Ptopo_couleur.eq.0) then
!            kkproc = kk+Ptopo_numproc-1
!        else
!            kkproc = kk -1
!        endif
! write(6,1000) 'Bln_vsendw_len',kkproc,Bln_vsendw_len(kk),Bln_vsendw_adr(kk)
! write(6,1000) 'Bln_vsende_len',kkproc,Bln_vsende_len(kk),Bln_vsende_adr(kk)
! write(6,1000) 'Bln_vsends_len',kkproc,Bln_vsends_len(kk),Bln_vsends_adr(kk)
! write(6,1000) 'Bln_vsendn_len',kkproc,Bln_vsendn_len(kk),Bln_vsendn_adr(kk)
!     enddo
      deallocate (recv_len,send_len)

 1000 format(a15,i3,'=',i5,'bytes, addr=',i5)
 1001 format(a15,i3,'=',i4,'bytes   i:', i3,' j:',i3)
       
!
      return
      end

