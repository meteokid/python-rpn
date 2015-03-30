!--------------------------------------------------------------------------
! This is free software, you can use/redistribute/modify it under the terms of
! the EC-RPN License v2 or any later version found (if not provided) at:
! - http://collaboration.cmc.ec.gc.ca/science/rpn.comm/license.html
! - EC-RPN License, 2121 TransCanada, suite 500, Dorval (Qc), CANADA, H9P 1J3
! - service.rpn@ec.gc.ca
! It is distributed WITHOUT ANY WARRANTY of FITNESS FOR ANY PARTICULAR PURPOSE.
!-------------------------------------------------------------------------- 
!==== int_macros2.hf [bgn] =========================================
! Copyright: MSC-RPN COMM Group Licence/Disclaimer version 2
! http://www.cmc.ec.gc.ca/rpn/modcom/licence.html
! @author Stephane Chamberland <stephane.chamberland@ec.gc.ca>
! @created 2004
! @lastmodified 2009
!-------------------------------------------------------------------
!  naming conventions:
!  integer :: _NI,_NI    !- Array dimensions
!  integer :: _IMIN,_IMAX !- Array index min and max
!  integer :: _I,_J      !- Integer position value
!  real    :: _X,_Y      !- Real position value
!  real    :: _V         !- Real values
!  real    :: _A1D       !- 1D Array
!  real    :: _A2D       !- 2D Array
!
!  var suffix 
!  - 0 or 1    : value at point before or after target dest (e.g X0,X1)
!  - no suffix : value at target point (e.g. V = fn(X))
!  macro suffix
!  - _48 : input in real(4) output in real(8)
!  - _8  : input in real(8) output in real(8)
!  - no suffix: input and output are real(4)
!===================================================================
!- Linear Interpolation of a 1D Array at pos X
!TODO: may want to optimize
!- Linear Interpolation of a 2D Array along X at pos X,J
!- Linear Interpolation of a 2D Array along Y at pos I,Y
!- Bilinear Interpolation of a 2D Array (A2D) at pos X,Y
!- First order approx of pos (X) of a value (V) in a monoticaly ordered vector/1D-Array (A1D)
!==== int_macros2.hf [end]===========================================
!/**
function get_corners_xy(xc,yc,x,y,ni,nj) result(F_istat)
   implicit none
   
   
   integer , parameter :: NB_CORNERS = 4
   integer :: ni,nj 
   real,dimension(ni,nj)   :: x,y   
   real,dimension(NB_CORNERS,ni,nj) :: xc,yc 
   
   integer :: F_istat
   
   
   
   integer, parameter :: SW=1,NW=2,NE=3,SE=4,DIM_I=1,DIM_J=2
   integer, parameter :: DIJ(2,NB_CORNERS) = reshape( (/ &
        -1,-1, & 
        -1, 1, & 
         1, 1, & 
         1,-1  & 
       /) , (/2,NB_CORNERS/))
   integer :: i,j,corner
   real    :: rdi,rdj,riCorner,rjCorner
   
   F_istat = 0
   
   do corner=1,NB_CORNERS
      rdi = 0.5*real(DIJ(DIM_I,corner))
      rdj = 0.5*real(DIJ(DIM_J,corner))
      do j=1,nj
         rjCorner = real(j) + rdj
         do i=1,ni
            riCorner = real(i) + rdi
            xc(corner,i,j) = real(          ((          ((dble(x(     max((1),min(floor(riCorner),(ni)-1)),     max((1),min(floor(r&
     &jCorner),(nj)-1))))) + ((dble(riCorner))-(dble(     max((1),min(floor(riCorner),(ni)-1)))))*     (((dble(x(     max((1),min(f&
     &loor(riCorner),(ni)-1))+1,     max((1),min(floor(rjCorner),(nj)-1)))))-(dble(x(     max((1),min(floor(riCorner),(ni)-1)),    &
     & max((1),min(floor(rjCorner),(nj)-1)))))) / ((dble(     max((1),min(floor(riCorner),(ni)-1))+1))-(dble(     max((1),min(floor&
     &(riCorner),(ni)-1)))))))) + ((dble(rjCorner))-(dble(     max((1),min(floor(rjCorner),(nj)-1)))))*     (((          ((dble(x( &
     &    max((1),min(floor(riCorner),(ni)-1)),     max((1),min(floor(rjCorner),(nj)-1))+1))) + ((dble(riCorner))-(dble(     max((1&
     &),min(floor(riCorner),(ni)-1)))))*     (((dble(x(     max((1),min(floor(riCorner),(ni)-1))+1,     max((1),min(floor(rjCorner)&
     &,(nj)-1))+1)))-(dble(x(     max((1),min(floor(riCorner),(ni)-1)),     max((1),min(floor(rjCorner),(nj)-1))+1)))) / ((dble(   &
     &  max((1),min(floor(riCorner),(ni)-1))+1))-(dble(     max((1),min(floor(riCorner),(ni)-1))))))))-(          ((dble(x(     max&
     &((1),min(floor(riCorner),(ni)-1)),     max((1),min(floor(rjCorner),(nj)-1))))) + ((dble(riCorner))-(dble(     max((1),min(flo&
     &or(riCorner),(ni)-1)))))*     (((dble(x(     max((1),min(floor(riCorner),(ni)-1))+1,     max((1),min(floor(rjCorner),(nj)-1))&
     &)))-(dble(x(     max((1),min(floor(riCorner),(ni)-1)),     max((1),min(floor(rjCorner),(nj)-1)))))) / ((dble(     max((1),min&
     &(floor(riCorner),(ni)-1))+1))-(dble(     max((1),min(floor(riCorner),(ni)-1))))))))) / ((dble(     max((1),min(floor(rjCorner&
     &),(nj)-1))+1))-(dble(     max((1),min(floor(rjCorner),(nj)-1))))))))
            yc(corner,i,j) = real(          ((          ((dble(y(     max((1),min(floor(riCorner),(ni)-1)),     max((1),min(floor(r&
     &jCorner),(nj)-1))))) + ((dble(riCorner))-(dble(     max((1),min(floor(riCorner),(ni)-1)))))*     (((dble(y(     max((1),min(f&
     &loor(riCorner),(ni)-1))+1,     max((1),min(floor(rjCorner),(nj)-1)))))-(dble(y(     max((1),min(floor(riCorner),(ni)-1)),    &
     & max((1),min(floor(rjCorner),(nj)-1)))))) / ((dble(     max((1),min(floor(riCorner),(ni)-1))+1))-(dble(     max((1),min(floor&
     &(riCorner),(ni)-1)))))))) + ((dble(rjCorner))-(dble(     max((1),min(floor(rjCorner),(nj)-1)))))*     (((          ((dble(y( &
     &    max((1),min(floor(riCorner),(ni)-1)),     max((1),min(floor(rjCorner),(nj)-1))+1))) + ((dble(riCorner))-(dble(     max((1&
     &),min(floor(riCorner),(ni)-1)))))*     (((dble(y(     max((1),min(floor(riCorner),(ni)-1))+1,     max((1),min(floor(rjCorner)&
     &,(nj)-1))+1)))-(dble(y(     max((1),min(floor(riCorner),(ni)-1)),     max((1),min(floor(rjCorner),(nj)-1))+1)))) / ((dble(   &
     &  max((1),min(floor(riCorner),(ni)-1))+1))-(dble(     max((1),min(floor(riCorner),(ni)-1))))))))-(          ((dble(y(     max&
     &((1),min(floor(riCorner),(ni)-1)),     max((1),min(floor(rjCorner),(nj)-1))))) + ((dble(riCorner))-(dble(     max((1),min(flo&
     &or(riCorner),(ni)-1)))))*     (((dble(y(     max((1),min(floor(riCorner),(ni)-1))+1,     max((1),min(floor(rjCorner),(nj)-1))&
     &)))-(dble(y(     max((1),min(floor(riCorner),(ni)-1)),     max((1),min(floor(rjCorner),(nj)-1)))))) / ((dble(     max((1),min&
     &(floor(riCorner),(ni)-1))+1))-(dble(     max((1),min(floor(riCorner),(ni)-1))))))))) / ((dble(     max((1),min(floor(rjCorner&
     &),(nj)-1))+1))-(dble(     max((1),min(floor(rjCorner),(nj)-1))))))))
         enddo
      enddo
   enddo
   
   return
end function get_corners_xy
!/**
function get_sidescenter_xy(xc,yc,x,y,ni,nj) result(F_istat)
   implicit none
   
   
   integer :: ni,nj 
   real,dimension(ni,nj)   :: x,y   
   real,dimension(4,ni,nj) :: xc,yc 
   
   integer :: F_istat
   
   
   
   integer :: i,j
   
   F_istat = 0
   do j=1,nj
      do i=1,ni
         
         if (j==1) then
            xc(1,i,j) = x(i,j) - real(0.5d0*(dble(x(i,j+1))-dble(x(i,j))))
            yc(1,i,j) = y(i,j) - real(0.5d0*(dble(y(i,j+1))-dble(y(i,j))))
         else
            xc(1,i,j) = real(0.5d0*(dble(x(i,j-1))+dble(x(i,j))))
            yc(1,i,j) = real(0.5d0*(dble(y(i,j-1))+dble(y(i,j))))
         endif
         
         if (i==1) then
            xc(2,i,j) = x(i,j) - real(0.5d0*(dble(x(i+1,j))-dble(x(i,j))))
            yc(2,i,j) = y(i,j) - real(0.5d0*(dble(y(i+1,j))-dble(y(i,j))))
         else
            xc(2,i,j) = real(0.5d0*(dble(x(i-1,j))+dble(x(i,j))))
            yc(2,i,j) = real(0.5d0*(dble(y(i-1,j))+dble(y(i,j))))
         endif
         
         if (j==nj) then
            xc(3,i,j) = x(i,j) + real(0.5d0*(dble(x(i,j))-dble(x(i,j-1))))
            yc(3,i,j) = y(i,j) + real(0.5d0*(dble(y(i,j))-dble(y(i,j-1))))
         else
            xc(3,i,j) = real(0.5d0*(dble(x(i,j+1))+dble(x(i,j))))
            yc(3,i,j) = real(0.5d0*(dble(y(i,j+1))+dble(y(i,j))))
         endif
         
         if (i==ni) then
            xc(4,i,j) = x(i,j) + real(0.5d0*(dble(x(i,j))-dble(x(i-1,j))))
            yc(4,i,j) = y(i,j) + real(0.5d0*(dble(y(i,j))-dble(y(i-1,j))))
         else
            xc(4,i,j) = real(0.5d0*(dble(x(i+1,j))+dble(x(i,j))))
            yc(4,i,j) = real(0.5d0*(dble(y(i+1,j))+dble(y(i,j))))
         endif
      enddo
   enddo
   
   return
end function get_sidescenter_xy
