!/* libdescrip - Vertical grid descriptor library for FORTRAN programming
! * Copyright (C) 2016  Direction du developpement des previsions nationales
! *                     Centre meteorologique canadien
! *
! * This library is free software; you can redistribute it and/or
! * modify it under the terms of the GNU Lesser General Public
! * License as published by the Free Software Foundation,
! * version 2.1 of the License.
! *
! * This library is distributed in the hope that it will be useful,
! * but WITHOUT ANY WARRANTY; without even the implied warranty of
! * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
! * Lesser General Public License for more details.
! *
! * You should have received a copy of the GNU Lesser General Public
! * License along with this library; if not, write to the
! * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
! * Boston, MA 02111-1307, USA.
program tests
  use vGrid_Descriptors, only: vgrid_descriptor,vgd_new,vgd_put,vgd_get,vgd_print,vgd_levels, &
       vgd_write,VGD_LEN_NAME,VGD_OK
  implicit none

  ! Variable declarations
  type(vgrid_descriptor) :: d
  integer, parameter :: STDERR=0
  integer :: status,fstkey,ni,nj,nk,ip1,ip2,iun=0,oun=0,tun=69
  real, dimension(:,:,:), pointer :: lev
  character(len=VGD_LEN_NAME) :: vname

  ! External functions
  integer, external :: fstinf,fnom,fstouv,fstfrm,fclos

  nullify(lev)

   ! Open file and get FST key for field of interest (in this case TT)
   status = fnom(iun,'../tests/data_Linux/dm_5002_from_model_run','STD',0)
   status = fstouv(iun,'RND')
   status = fnom(oun,'test_output.fst','STD',0)
   status = fstouv(oun,'RND')
   fstkey = fstinf(iun,ni,nj,nk,-1,' ',-1,-1,-1,' ','TT')

   ! Open associated text file to get ip1/2 values of descriptors
   open(unit=tun,file='../tests/data_Linux/dm_5002_ips.txt',status='OLD')
   read(tun,*) ip1,ip2
   close(tun)

  ! Construct a new set of 3D coordinate descriptors
  status =vgd_new(d,unit=iun,format='fst',ip1=ip1,ip2=ip2)
  if (status /= VGD_OK) write(STDERR,*) 'WARNING - error during construction'

  ! Get information about the coordinate
  status = vgd_get(d,key='NAME - descriptor name',value=vname)
  if (status /= VGD_OK) write(STDERR,*) 'WARNING - error during get operation'

  ! Change an element of the structure
  status = vgd_put(d,key='NAME - descriptor name',value='VCRD')
  if (status /= VGD_OK) write(STDERR,*) 'WARNING - error during put operation'

  ! Print information about the instance
  status = vgd_print(d)
  if (status /= VGD_OK) write(STDERR,*) 'WARNING - error during print'

  ! Get physical levelling information
  status = vgd_levels(d,unit=iun,fstkeys=(/fstkey/),levels=lev)
  if (status /= VGD_OK) write(STDERR,*) 'WARNING - error during level calculation'

  ! Write descriptors to an output file
  status = vgd_write(d,unit=oun,format='fst')
  if (status /= VGD_OK) write(STDERR,*) 'WARNING - error during write'

  ! Close files
  status = fstfrm(iun)
  status = fclos(iun)
  status = fstfrm(oun)
  status = fclos(oun)

end program tests
