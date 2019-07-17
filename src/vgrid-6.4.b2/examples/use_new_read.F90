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
program use_new_read
   use vGrid_Descriptors, only: vgrid_descriptor,vgd_new,vgd_print,VGD_OK

   implicit none
   
   integer :: status,fnom,fstouv,iun=10
   type(vgrid_descriptor) :: vgd

   status = fnom(iun,'../tests/data_Linux/dm_5002_from_model_run','RND',0)
   status = fstouv(iun,'RND')

   status = vgd_new(vgd,iun,'fst')
   
   if (status /= VGD_OK)then
      print*,'ERROR'
      call exit(1)
   endif

   status = vgd_print(vgd)

end program use_new_read
