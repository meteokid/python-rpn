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

real*8 function yyg_weight (x,y,dx,dy,np)
   Implicit none

   Real*8  :: x,y,dx,dy
   integer :: np

!author
!     Author Abdessamad Qaddouri -- Summer 2014
!
!revision
! v4_70 - Qaddouri A.     - initial version
!
!Object
!    Based on a Draft from Zerroukat (2013) - Evaluates weight
!    for each cell, so that the overlap is computed once

   real*8 :: pi, xmin, xmax, ymin, ymax, xb1, xb2
   real*8 :: t1x, t2x, t1y, dcell, xi, yi, di, dp, df, d

!
!     ---------------------------------------------------------------
!
   pi   = acos(-1.d0)
   xmin = -3.d0*pi/4.d0 ;  xmax = 3.d0*pi/4.d0
   ymin = -pi/4.d0       ;  ymax = pi/4.d0
   xb1  = -0.5d0*pi       ;  xb2  = 0.5d0*pi

   t1x = (x-xmin)*(xmax-x)
   t2x = (x-xb1)*(xb2-x)
   t1y = (y-ymin)*(ymax-y)

   if ( t1x < 0.d0 .or. t1y < 0.d0 ) then
      yyg_weight = 0.0
   elseif ( t2x > 0.d0 .and. t1y > 0.d0 ) then
      yyg_weight = 1.d0
   else
      dcell = 0.5d0*dsqrt(dx**2 + dy**2)

      call inter_curve_boundary_yy (x, y, xi, yi, np)


      di = sqrt( xi**2 + yi**2 )
      dp = sqrt(  x**2 +  y**2 )
      df = dp - di
      d  = min(max(-dcell,df),dcell)
      yyg_weight = 0.5d0*(1.d0 - (d/dcell))
   endif
!
!     ---------------------------------------------------------------
!
return
end

!----------------------------------------------------------------

subroutine inter_curve_boundary_yy (x,y,xi,yi,np)
    implicit none
    real*8 :: x,y,xi,yi

    real*8 :: tol, pi, xmin, ymin, xb
    real*8 :: xc, yc, s1, s2, x1, test, dxs
    real*8 :: xp1, xp2, xr1, yr1, xr2, yr2
    integer :: np, i
!
!     ---------------------------------------------------------------
!
    tol = 1.0d-16
    pi  = acos(-1.d0)
    xmin = -3.d0*pi/4.d0
    ymin = -pi/4.d0
    xb  = -0.5d0*pi

    xc = x
    yc = y

    if ( x > 0.d0 ) xc = - x
    if ( y > 0.d0 ) yc = - y

    If ( abs(xc) < tol ) then
        xi = xc
        yi = ymin
    Else

        s1 = yc / xc
        s2 = ymin/(xb-xmin)

        x1 = s2*xmin/(s2-s1)
        if (x1 > xb ) then
           xi = ymin/s1
           yi = ymin
        else
            test = -1.d0
            dxs  = -xb/(np-1)
            i = 1
            Do while (test < 0.d0 )
               xp1 = (i-1)*dxs + xb
               xp2 =   (i)*dxs + xb
               xr1 = atan2(sin(ymin),-cos(ymin)*cos(xp1))
               yr1 = asin(cos(ymin)*sin(xp1))
               xr2 = atan2(sin(ymin),-cos(ymin)*cos(xp2))
               yr2 = asin(cos(ymin)*sin(xp2))
               s2 = (yr1-yr2)/(xr1-xr2)
               xi = (s2*xr2-yr2)/(s2-s1)
               yi = s1*xi
               test=(xi-xr1)*(xr2-xi)
               i = i+1
            Enddo
         endif
    Endif

    if ( x > 0.d0 ) xi = - xi
    if ( y > 0.d0 ) yi = - yi
!
!     ---------------------------------------------------------------
!
return
end
