!--------------------------------------------------------------------------
! This is free software, you can use/redistribute/modify it under the terms of
! the EC-RPN License v2 or any later version found (if not provided) at:
! - http://collaboration.cmc.ec.gc.ca/science/rpn.comm/license.html
! - EC-RPN License, 2121 TransCanada, suite 500, Dorval (Qc), CANADA, H9P 1J3
! - service.rpn@ec.gc.ca
! It is distributed WITHOUT ANY WARRANTY of FITNESS FOR ANY PARTICULAR PURPOSE.
!-------------------------------------------------------------------------- 
!/**
subroutine test_get_corners_xy0(ni,nj)
   implicit none
   
   
   integer,intent(in) :: ni,nj
   
   
   
   integer, parameter :: SW=1,NW=2,NE=3,SE=4,DIM_I=1,DIM_J=2
   integer , parameter :: NB_CORNERS = 4
   real,dimension(ni,nj)   :: x,y
   real,dimension(NB_CORNERS,ni,nj) :: xc,yc
   integer :: istat
   integer :: i,j
   integer, external :: get_corners_xy
   
   do j=0,nj-1
      do i=0,ni-1
         x(i+1,j+1) = float(i)
         y(i+1,j+1) = float(j)
      enddo
   enddo
   istat = get_corners_xy(xc,yc,x,y,ni,nj)
   istat = 0
   do j=1,nj
      do i=1,ni
         if (.not. (&
              xc(NW,i,j) == x(i,j)-0.5 .and. xc(SW,i,j) == x(i,j)-0.5 .and. &
              xc(NE,i,j) == x(i,j)+0.5 .and. xc(SE,i,j) == x(i,j)+0.5)) then
            istat = 1
            print '(A,2I3)','XC',i,j
            print '(F4.1,4X,F4.1)',xc(NW,i,j),xc(NE,i,j)
            print '(4X,F4.1)',x(i,j)
            print '(F4.1,4X,F4.1)',xc(SW,i,j),xc(SE,i,j)
         endif
         if (.not. (&
              yc(NW,i,j) == y(i,j)+0.5 .and. yc(NE,i,j) == y(i,j)+0.5 .and. &
              yc(SW,i,j) == y(i,j)-0.5 .and. yc(SE,i,j) == y(i,j)-0.5)) then
            istat = 1
            print '(A,2I3)','YC',i,j
            print '(F4.1,4X,F4.1)',yc(NW,i,j),yc(NE,i,j)
            print '(4X,F4.1)',y(i,j)
            print '(F4.1,4X,F4.1)',yc(SW,i,j),yc(SE,i,j)
         endif
      enddo
   enddo
   if (istat == 0) then
      print *,'test_get_corners_xy: OK'
   else
      print *,'test_get_corners_xy: FAILED'
   endif
   
   return
end subroutine test_get_corners_xy0
program test_get_corners_xy
   implicit none
   
   
   
   
   
   call test_get_corners_xy0(3,3)
   
end program
