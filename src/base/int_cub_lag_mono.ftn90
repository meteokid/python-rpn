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
!**s/r int_cub_lag_mono - to do YY cubic interpolation for fields (local PE)


       Subroutine int_cub_lag_mono(FF,F,Imx,Imy,geomgx,geomgy,Minx,Maxx,Miny,Maxy,Xi,Yi) 

       implicit none
#include <arch_specific.hf>
!
!author   
!           Abdessamad Qaddouri - October 2009
!revision   V.Lee - Aug 2011 (to replace int_cub_lag,int_cub_lagu,int_cub_lagv)
!
#include "glb_ld.cdk"

       integer Imx,Imy,Nx,Ny,Minx,Maxx,Miny,Maxy
       integer k,i,j,Mx,My
       real*8  W1,W2,W3,W4,X1,XX,X2,X3,X4 
       integer Im, Jm
       real*8 YY,y1,y2,y3,y4,FF,Fi
       real*8 F(Minx:Maxx,Miny:Maxy),fx1,fx2,fx3,fx4
       real*8 Xi,Yi,geomgx(Minx:Maxx),geomgy(Miny:Maxy)
       real*8 prmax,prmin

!
        Im = iMx
        Jm = iMy
	   X1  = Geomgx(im)
	   X2  = Geomgx(im+1) - X1
	   X3  = Geomgx(im+2) - X1
	   X4  = Geomgx(im+3) - X1
	   XX  = Xi   - X1
!
	   W1  = (XX-X2)*(XX-X3)*(XX-X4)/(-X2*X3*X4)
	   W2  =  XX    *(XX-X3)*(XX-X4)/(X2*(X2-X3)*(X2-X4))
	   W3  =  XX    *(XX-X2)*(XX-X4)/(X3*(X3-X2)*(X3-X4))
	   W4  =  XX    *(XX-X2)*(XX-X3)/(X4*(X4-X2)*(X4-X3))

          prmax=max(F(Im,Jm),F(Im + 1,Jm ),F( Im + 2, Jm ),F( Im + 3, jm ),&
           F(Im ,Jm +1),F(Im +1,jm +1),F(Im +2,Jm +1 ),F( Im + 3, Jm + 1 ),&
           F(Im,Jm+2),F(Im+1,jm+2),F(Im+2,Jm+2),F( Im + 3, Jm + 2 ),&
           F(Im,jm+3),F(Im+1,jm+3),F(Im +2,Jm+3),F( Im + 3, Jm + 3 ))
          prmin=min(F(Im,Jm),F(Im + 1,Jm ),F( Im + 2, Jm ),F( Im + 3, jm ),&
           F(Im ,Jm +1),F(Im +1,jm +1),F(Im +2,Jm +1 ),F( Im + 3, Jm + 1 ),&
           F(Im,Jm+2),F(Im+1,jm+2),F(Im+2,Jm+2),F( Im + 3, Jm + 2 ),&
           F(Im,jm+3),F(Im+1,jm+3),F(Im +2,Jm+3),F( Im + 3, Jm + 3 ))
!           
          Fx1 = W1*F(Im ,Jm)+W2*F(Im + 1,Jm )+ W3* F( Im + 2, Jm )+ &
            W4 * F( Im + 3, jm )
	  Fx2 =W1*F(Im ,Jm +1)+W2*F(Im +1,jm +1)+W3*F(Im +2,Jm +1 )+&
            W4 * F( Im + 3, Jm + 1 )
	  Fx3 =W1*F(Im,Jm+2)+W2*F(Im+1,jm+2)+W3*F(Im+2,Jm+2)+       &
            W4 * F( Im + 3, Jm + 2 )
	  Fx4 =W1*F(Im,jm+3)+W2*F(Im+1,jm+3)+W3*F(Im +2,Jm+3)+      &
            W4 * F( Im + 3, Jm + 3 )
!           
	   Y1  = Geomgy(Jm)
	   Y2  = Geomgy(Jm+1) - Y1
	   Y3  = Geomgy(Jm+2) - Y1
	   Y4  = Geomgy(Jm+3) - Y1
	   YY  = Yi   - Y1
!
           W1  = (yy-y2)*(yy-y3)*(yy-y4)/(-y2*y3*y4)
           W2  =  yy    *(yy-y3)*(yy-y4)/(y2*(y2-y3)*(y2-y4))
           W3  =  yy    *(yy-y2)*(yy-y4)/(y3*(y3-y2)*(y3-y4))
           W4  =  yy    *(yy-y2)*(yy-y3)/(y4*(y4-y2)*(y4-y3))

           FF= W1*fx1+W2*fx2+W3*fx3+W4*fx4
           FF= max(prmin,min(prmax,FF))

        return
        end 
