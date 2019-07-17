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
 program levels_with_ref
   use vGrid_Descriptors, only: vgrid_descriptor,vgd_new,vgd_levels,VGD_OK
   !
   implicit none
   !
   integer :: status,count,iun=0,key,k
   integer, dimension(1000) :: keys
   integer, dimension(:), pointer :: ip1s
   real, dimension(:,:,:), pointer :: pressure
   real, dimension(:,:), pointer :: p0
   type(vgrid_descriptor) :: vgd
   ! For fstprm
   integer :: ig1,ig2,ig3,ig4,dateo,deet,npas,datyp,nbits
   integer :: ni,nj,nk
   integer :: ip1,ip2,ip3,swa,lng,dltf,ubc,extra1,extra2,extra3
   character(len=1) :: grtyp
   character(len=2) :: typvar
   character(len=4) :: nomvar
   character(len=12) :: etiket
   !
   ! External functions
   integer, external :: fstinl,fnom,fstouv,fstfrm,fclos,fstinf,fstluk,fstprm
   !
   !===========================================================================
   !
   nullify(ip1s,pressure,p0)
   !
   !===========================================================================
   ! Open file and get FST key for field of interest (in this case TT)
   ! Get its ip1s 
   !
   status = fnom(iun,"../tests/data_Linux/dm_5001_from_model_run",'STD',0)
   status = fstouv(iun,'RND')
   status = fstinl(iun,ni,nj,nk,-1,' ',-1,-1,-1,' ','TT',keys,count,size(keys))
   !
   ! Allocate space for 3D pressure, P0 and ip1s
   !
   allocate(pressure(ni,nj,count),p0(ni,nj),ip1s(count),stat=status)
   !
   ! Read P0
   !
   key = fstinf(iun,ni,nj,nk,-1,' ',-1,-1,-1,' ','P0')
   status = fstluk(p0,key,ni,nj,nk)
   !
   ! Get ip1 list in ip1s
   !
   do k=1,count
      status = fstprm(keys(k),dateo,deet,npas,ni,nj,nk,nbits,datyp,ip1,&
           ip2,ip3,typvar,nomvar,etiket,grtyp,ig1,ig2,ig3, &
           ig4,swa,lng,dltf,ubc,extra1,extra2,extra3)
      ip1s(k) = ip1
   end do
   !
   ! ==============================================================================
   !
   ! Get vertical structure information of file associated to iun in variable vgd
   !
   status = vgd_new(vgd,iun,'fst')
   if (status /= VGD_OK)then
      print*,'Error with vgd_new'
      call exit(1)
   endif
   !
   ! Get pressure levels for TT variable
   !
   status = vgd_levels(vgd,ip1s,pressure,p0)
   if (status /= VGD_OK)then
      print*,'Error with vgd_levels'
      call exit(1)
   endif
   !
   print*,'pressure(1,1,1:count)=',pressure(1,1,1:count)
   print*,'p0(1,1)=',p0(1,1)
   !
   ! Close the input file and stop
   status = fstfrm(iun)
   status = fclos(iun)
end program levels_with_ref
