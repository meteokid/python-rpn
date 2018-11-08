program test_fstluk
   implicit none
   integer,external :: fnom, fstouv, fstinf, fstluk, fstfrm, fclos
   character(len=128) :: filename
   integer :: iunit,ni,nj,nk,key,istat
   real, allocatable :: data(:,:,:)
   iunit = 0
   filename = '/cnfs/ops/production/gridpt/dbase/prog/gsloce/2015070706_042'
   istat = fnom(iunit,trim(filename),'RND+OLD+R/O',0)
   istat = fstouv(iunit,'RND')
   key = fstinf(iunit,ni,nj,nk,-1,' ', -1,-1,-1,'P@','TM')
   allocate(data(ni,nj,nk))
   istat = fstluk(data,key,ni,nj,nk)
   istat = fstfrm(iunit)
   istat = fclos(iunit)
   print *,data(1,1,1)
   deallocate(data)
end program 
