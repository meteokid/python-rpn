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
subroutine ser_init5 ( phy_lcl_gid, drv_glb_gid        , &
                       phydim_ni, phydim_nj, phydim_nk , &
                       p_runlgt, phy_lcl_ni, phy_lcl_nj, &
                       moyhr, delt, master_pe )
   implicit none
#include <arch_specific.hf>

   integer,intent(in) :: phy_lcl_gid, drv_glb_gid, phydim_ni, phydim_nj, &
                         phydim_nk, p_runlgt, phy_lcl_ni, phy_lcl_nj   , &
                         moyhr, master_pe
   real   ,intent(in) :: delt

   !@description
   !      This routine initializes the physics variables
   !      related to time series extraction: variable names to
   !      extract (profil:3D, surface:2D), grid point indicies where
   !      to extract from, number or vertical levels...
   !
   !      It also performs memory allocation for buffers based
   !      on: the number of 2D and 3D variables, the number of
   !      vertical level and the number of grid point where to
   !      extract from.
   !*@/
#include <WhiteBoard.hf>
#include <msg.h>
   include "series.cdk"

   integer, external :: msg_getUnit, serdim
   integer :: err(13), pnmxsrf, i, j, xst_nstatl
   integer :: iold, jold, inew, jnew, nelem, p_nmp
   !---------------------------------------------------------------
   err = 0
   xst_master_pe = master_pe
   xst_unout     = msg_getUnit(MSG_INFO)
   
   mxsrf=0 ; mxprf=0   ; mxstt=0   ; mxnvo=0
   nstat=0 ; nsurf=0   ; nprof=0   ; initok=.false.
   srwri=0 ; tsver=100 ; tsmoyhr=0 ; series_paused=.false.

   if (p_serg_srsus_l) then
      call ser_stnij3(phy_lcl_gid,drv_glb_gid,phy_lcl_ni,phy_lcl_nj)
   else
      xst_nstat = 0
      return
   endif

   !# Keep only locally available stations
   xst_nstatl = 0
   do i = 1, xst_nstat
      if ( (xst_stn(i)%stcori .ne. STN_MISSING) .and. &
           (xst_stn(i)%stcorj .ne. STN_MISSING) ) then
         xst_nstatl = xst_nstatl + 1
         xst_stn(xst_nstatl)%lclsta = i
         xst_stn(xst_nstatl)%stcori = xst_stn(i)%stcori
         xst_stn(xst_nstatl)%stcorj = xst_stn(i)%stcorj
      endif
   end do

   !# Add Mandatory P0 for profiles
   if (p_serg_srprf > 0) then
      p_serg_srsrf = p_serg_srsrf + 1
      p_serg_srsrf_s(p_serg_srsrf) = 'p0'
   endif

   ! extracted variables at each station is written to disk
   ! once every "p_serg_srwri" time step.
   ! serallc does memory allocation for buffer containing one
   ! timestep information to be written on the disk.

   pnmxsrf = max(cnsrgeo, p_serg_srsrf)
   mxstt = xst_nstat
   mxsrf = pnmxsrf
   mxprf = p_serg_srprf
   mxnvo = phydim_nk

   ! xst_dimsers = max(1, xst_nstat * pnmxsrf  )
   ! xst_dimserp = max(1, mxstt * mxprf * mxnvo)

   xst_dimsers = max(1,serdim (xst_nstat,pnmxsrf,1))
   xst_dimserp = max(1,serdim (xst_nstat,p_serg_srprf,phydim_nk))

   allocate(xstb_sers(xst_nstat,pnmxsrf), &
        xstb_sersx(xst_nstat,pnmxsrf), &
        xstb_serp (phydim_nk,xst_nstat,p_serg_srprf), &
        xstb_serpx(phydim_nk,xst_nstat,p_serg_srprf), &
        lastout_surf(xst_nstat,pnmxsrf), lastout_prof(xst_nstat,pnmxsrf))

   allocate(jstat(mxstt),ijstat(mxstt,2),statnum(mxstt),&
        jstat_g(mxstt),istat_g(mxstt),kam(phydim_nj),&
        name(mxstt*stn_string_length/4))

   sers => xstb_sers
   serp => xstb_serp
   ninjnk(1) = phydim_ni
   ninjnk(2) = phydim_nj
   ninjnk(3) = phydim_nk
   nstat   = 0
   nstat_g = 0
   nsurf   = 0
   nprof   = 0
   ijstat  = 0
   jstat   = 0
   istat_g = 0
   jstat_g = 0
   statnum = 0
   name    = ''
   surface = '        '
   profils = '        '
   sers    = 0.
   serp    = 0.

   lastout_surf = -1
   lastout_prof = -1

   !# Initializes number of vertical levels
   call serset('ISTAT',xst_stn(:)%i,xst_nstat,err(2))
   do j= 1, phydim_nj 
      call sersetm('KA', j, phydim_nk)
   end do

   !# Convert to folded bus index
   if (P_runlgt > 0) then
      do i = 1, xst_nstat
         iold = xst_stn(i)%stcori
         jold = xst_stn(i)%stcorj

         nelem = (jold-1)*phy_lcl_ni + iold
         jnew = nelem/phydim_ni
         if (phydim_ni*jnew < nelem) then
            jnew = jnew+1
         endif
         inew = nelem-(jnew-1)*phydim_ni

         xst_stn(i)%stcori = inew
         xst_stn(i)%stcorj = jnew
      end do
   endif

   !# Initialize station identities
   call serset  ('ISTAT'  ,xst_stn(:)%stcori,xst_nstatl,err(3))
   call serset  ('JSTAT'  ,xst_stn(:)%stcorj,xst_nstatl,err(4))
   call serset  ('STATNUM',xst_stn(:)%lclsta,xst_nstatl,err(5))
   call sersetc ('NAME'   ,xst_stn(:)%name,xst_nstat,err(6))
   call serset  ('ISTAT_G',xst_stn(:)%i   ,xst_nstat,err(7))
   call serset  ('JSTAT_G',xst_stn(:)%j   ,xst_nstat,err(8))      

   !# Initializes name of SURFACE type variables (2D variables)
   call sersetc('SURFACE', P_serg_srsrf_s, P_serg_srsrf, err(9))

   !# Initializes name of PROFILE type variables (3D variables)
   call sersetc('PROFILS', P_serg_srprf_s, P_serg_srprf, err(10))

   !# Initializes the frequency of extraction
   call serset('SERINT', P_serg_srwri, 1, err(11))

   !# Initializes model and time series output time step
   call serset('TSMOYHR', moyhr, 1, err(12))
   call serset('SRWRI', int(P_serg_srwri*delt), 1, err(13))

   !# Initializes buffers to zero
   call serdbu()

   if (minval(err) >= 0) then
      initok=.true.
   else
      write(*,*) 'ser_init error',err
   endif
   !---------------------------------------------------------------
   return
end subroutine ser_init5
