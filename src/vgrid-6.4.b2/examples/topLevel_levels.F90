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
 program calculate_levels
   use vGrid_Descriptors, only: vgd_levels,VGD_OK
   implicit none
   !
   ! Variable declarations
   integer :: status,count,ni,nj,nk,iun=0
   integer, dimension(1000) :: keys
   real, dimension(:,:,:), pointer :: lev
   !
   ! External functions
   integer, external :: fstinl,fnom,fstouv,fstfrm,fclos
   !
   
   nullify(lev)

   ! Open file and get FST key for field of interest (in this case TT)
   status = fnom(iun,'../tests/data_Linux/dm_5002_from_model_run_ig4_ip1_link','STD',0)
   status = fstouv(iun,'RND')
   status = fstinl(iun,ni,nj,nk,-1,' ',-1,-1,-1,' ','TT',keys,count,size(keys))
   !
   ! Calculate level information for this field
   status = vgd_levels(unit=iun,fstkeys=keys(1:count),levels=lev)
   if (status /= VGD_OK) write(0,*) 'WARNING - error during level calculation'
   !
   ! Generate a bit of output
   print*, 'shape of returned lev: ',shape(lev)
   print*, 'profile of pressures at (20,10) ',lev(20,10,:)

   ! Close the input file and stop
   status = fstfrm(iun)
   status = fclos(iun)
 end program calculate_levels
