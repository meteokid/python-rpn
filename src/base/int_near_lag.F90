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
!**s/r int_near_lag - to do YY nearest interpolation for scalar
!
       subroutine int_near_lag3(FF,F,Imx,Imy,Geomgx,Geomgy,Minx,Maxx,Miny,Maxy,Nk,Xi,Yi,NLEN)

       implicit none
#include <arch_specific.hf>

       integer Minx,Maxx,Miny,Maxy,Nk,NLEN
       integer Imx(NLEN),Imy(NLEN)
       real FF(NLEN*Nk)
       real*8 F(Minx:Maxx,Miny:Maxy,Nk),geomgx(Minx:Maxx),geomgy(Miny:Maxy)
       real*8 Xi(NLEN),Yi(NLEN)

!
!author
!           Abdessamad Qaddouri - October 2009
!
       integer i,j,k,Im,Jm
       real*8  FF_8,betax,betax1,betay,betay1,h1,h2
!
       h1=Geomgx(2)-Geomgx(1)
       h2=Geomgy(2)-Geomgy(1)

!$omp parallel private (i,j,k,im,jm,       &
!$omp      FF_8,betax,betax1,betay,betay1) &
!$omp shared (h1,h2)
!$omp do
   Do 200 i=1,NLEN
       Im = imx(i)
       Jm = imy(i)

       betax= ( Xi(i) - Geomgx(Im))/h1
       betax1= (1.0-betax)
       if (betax.le.betax1) then
           betax=0.0d0
           betax1=1.0d0
       else
           betax=1.0d0
           betax1=0.0d0
       endif

       betay=(Yi(i) - Geomgy(Jm))/h2
       betay1=1.0-betay
       if (betay.le.betay1) then
           betay=0.0d0
           betay1=1.0d0
       else
           betay=1.0d0
           betay1=0.0d0
       endif
   Do 300 k=1,Nk
       FF_8= betay1*(betax1*F(Im,Jm,k)+betax*F(Im+1,Jm,k))+ &
           betay*(betax1*F(Im,Jm+1,k)+betax*F(Im+1,Jm+1,k))
       FF((i-1)*Nk+k)=real(FF_8)
   300 continue
   200 continue
!$omp enddo
!$omp end parallel

      return
      end
       SUBROUTINe int_near_lag2(FF,F,Imx,Imy,Geomgx,Geomgy,Minx,Maxx,Miny,Maxy,Xi,Yi) 

       implicit none
#include <arch_specific.hf>
!
!author
!           Abdessamad Qaddouri - October 2009
!
       integer Ni,Nj,Imx,Imy,Minx,Maxx,Miny,Maxy
       integer Im, Jm
       real*8 F(Minx:Maxx,Miny:Maxy),geomgx(Minx:Maxx),geomgy(Miny:Maxy)
       real*8 FF,Xi,Yi,h1,h2
       real*8 betax,betax1,betay,betay1
!
       Im = imx
       Jm = imy
       h1=Geomgx(2)-Geomgx(1)
       h2=Geomgy(2)-Geomgy(1)
	   betax= ( Xi-Geomgx(Im))/h1
           betax1= (1.0-betax)
           if (betax.le.betax1) then
             betax=0.0d0
             betax1=1.0d0
           else
             betax=1.0d0
             betax1=0.0d0
           endif
           betay=(Yi-Geomgy(Jm))/h2
           betay1=1.0-betay
           if (betay.le.betay1) then
             betay=0.0d0
             betay1=1.0d0
           else
             betay=1.0d0
             betay1=0.0d0
           endif
           FF= betay1*(betax1*F(Im,Jm)+betax*F(Im+1,Jm))+ &
                    betay*(betax1*F(Im,Jm+1)+betax*F(Im+1,Jm+1))
      return
      end 
       SUBROUTINe int_near_lag(FF,F,Imx,Imy,Ni,Nj,Nx,Ny,Xi, &
                                            Yi,x,y,h1,h2) 

       implicit none
!
!author
!           Abdessamad Qaddouri - October 2009
!
       integer Ni,Nj,Imx(Ni,Nj),Imy(Ni,Nj),Nx,Ny
       integer k,i,j,Mx(Ni,Nj),My(Ni,Nj)
       real*8  W1,W2,W3,W4,X1,XX,X2,X3,X4 
       integer Im, Jm
       real*8 YY,y1,y2,y3,y4,FF(Ni,Nj),Fi(Ni*Nj)
       real*8 F(Nx,Ny),x(*),y(*)
       real*8 Xi(Ni,Nj),Yi(Ni,Nj),h1,h2
       real*8 betax,betax1,betay,betay1
!
       Do j=1, Nj
	Do i = 1, Ni
	   Im = imx(i,j)
	   Jm = imy(i,j)
	   betax= ( Xi(i,j)-x(Im))/h1
           betax1= (1.0-betax)
           if (betax.le.betax1) then
             betax=0.0d0
             betax1=1.0d0
           else
             betax=1.0d0
             betax1=0.0d0
           endif
           betay=(Yi(i,j)-y(Jm))/h2
           betay1=1.0-betay
           if (betay.le.betay1) then
             betay=0.0d0
             betay1=1.0d0
           else
             betay=1.0d0
             betay1=0.0d0
           endif
           FF(i,j)= betay1*(betax1*F(Im,Jm)+betax*F(Im+1,Jm))+ &
                    betay*(betax1*F(Im,Jm+1)+betax*F(Im+1,Jm+1))

	Enddo
      Enddo

      return
      end 
