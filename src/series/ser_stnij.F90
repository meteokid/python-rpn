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

!/@*
subroutine ser_stnij3(phy_lcl_gid,drv_glb_gid,phy_lcl_ni,phy_lcl_nj)
   implicit none
#include <arch_specific.hf>
   !@object: compute i,j from lat,lon
   !@params
   integer,intent(in) :: phy_lcl_gid, drv_glb_gid, phy_lcl_ni, phy_lcl_nj
   !@author: V. Lee - May 2000
   !@revisions
   ! v4_50 - Desgagne M.   - Major revision
   !*@/
   include "series.cdk"
   integer,external :: gdxyfll
   integer :: i,k,il,jl,ig,jg,ni,nj,err,pos(1)
   real, dimension(:), allocatable :: xl, yl, xg, yg, lat, lon
   type(station), allocatable :: stn_tmp(:)
   !---------------------------------------------------------------
   if (xst_nstat < 1) return

   if (xst_unout > 0) write(xst_unout,1001)
1001 format('PROCESSING TIME-SERIES GRID POINTS (S/R SER_stnij)', &
          /,'=============================================+===')

   allocate(stn_tmp(xst_nstat), &
        xl(xst_nstat), yl(xst_nstat), &
        xg(xst_nstat), yg(xst_nstat), &
        lat(xst_nstat), lon(xst_nstat))

   !# Get grid points i,j
   do k= 1, xst_nstat
      lat(k) = xst_stn(k)%lat
      lon(k) = xst_stn(k)%lon
   enddo

   err = gdxyfll(phy_lcl_gid, xl, yl, lat, lon, xst_nstat)
   err = gdxyfll(drv_glb_gid, xg, yg, lat, lon, xst_nstat)
   do k= 1,xst_nstat
      xst_stn(k)%i = nint(xg(k))
      xst_stn(k)%j = nint(yg(k))         
      il = nint(xl(k))
      jl = nint(yl(k))
      if (il > 0 .and. il <= phy_lcl_ni .and. jl > 0 .and. jl <= phy_lcl_nj) then
         !# Note: i,j are not folded values, ser_init converts them
         xst_stn(k)%stcori = il
         xst_stn(k)%stcorj = jl
      else
         xst_stn(k)%stcori = STN_MISSING
         xst_stn(k)%stcorj = STN_MISSING     
      endif
   enddo

   !# Put the stations in increasing order of index in a list
   ni= maxval(xst_stn(1:xst_nstat)%i)
   nj= maxval(xst_stn(1:xst_nstat)%j)
   do k=1,xst_nstat
      xst_stn(k)%index = xst_stn(k)%i+(xst_stn(k)%j-1)*ni
   enddo
   i = 1
   do k=1,xst_nstat
      pos = minloc(xst_stn(1:xst_nstat)%index)
      stn_tmp(i) = xst_stn(pos(1))
      xst_stn(pos(1))%index = ni*nj+1
      i = i+1
   enddo
   xst_stn(1:xst_nstat) = stn_tmp(1:xst_nstat)

   deallocate(stn_tmp, xl, yl, xg, yg, lat, lon, stat=err)

   if (xst_unout.gt.0) then
      write(xst_unout,910)
      do k = 1,xst_nstat
         write(xst_unout,912) k,xst_stn(k)%name,xst_stn(k)%i,&
              xst_stn(k)%j,xst_stn(k)%lat,xst_stn(k)%lon
      enddo
      write(xst_unout,901)
   endif

 901  format(' __________________________________________________________________')
 910  format(' __________________________________________________________________', &
           /,'  Reordered grid points with ACTUAL lat-lon values and short names', &
           /,' __________________________________________________________________', &
           /,'    N    |        NAME        |   I    |   J    |  LAT   |  LON   |' &
           /,' __________________________________________________________________')
 912  format(1x,I5,'    ',a18,'   ',I5,'    ',I5,'    ',f8.3,' ',f8.3,' ')
   !---------------------------------------------------------------
   return
end subroutine ser_stnij3
