!-------------------------------------- LICENCE BEGIN ------------------------------------
!Environment Canada - Atmospheric Science and Technology License/Disclaimer, 
!                     version 3; Last Modified: May 7, 2008.
!This is free but copyrighted software; you can use/redistribute/modify it under the terms 
!of the Environment Canada - Atmospheric Science and Technology License/Disclaimer 
!version 3 or (at your option) any later version that should be found at: 
!http://collaboration.cmc.ec.gc.ca/science/rpn.comm/license.html 
!
!This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
!without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
!See the above mentioned License/Disclaimer for more details.
!You should have received a copy of the License/Disclaimer along with this software; 
!if not, you can write to: EC-RPN COMM Group, 2121 TransCanada, suite 500, Dorval (Quebec), 
!CANADA, H9P 1J3; or send e-mail to service.rpn@ec.gc.ca
!-------------------------------------- LICENCE END --------------------------------------
!/@*
subroutine ser_write2(F_date, F_etiket, F_ig, F_dgrw,&
                      F_nhybm, F_nhybt,&
                      F_rgas,F_grav,F_satues_L, F_satuco_L, F_init_L ,F_wr_L)
      use vGrid_Descriptors, only: vgrid_descriptor,vgd_get,VGD_OK,VGD_ERROR
      use vgrid_wb, only: vgrid_wb_get
  implicit none
#include <arch_specific.hf>

  !@objective Generate a time series binary dump file.

  !@arguments
  integer, dimension(14), intent(in) :: F_date          !date array
  character(len=*), intent(in) :: F_etiket              !forecast label
  integer, intent(in) :: F_nhybm,F_nhybt          !number of momentum levels
  integer F_ig(4)
  real, intent(in) :: F_dgrw                            !east-west grid orientation
  real, intent(in) :: F_rgas                  !dry gas constant
  real, intent(in) :: F_grav                  !gravity
  logical, intent(in) :: F_satues_L                     !include saturation wrt ice for post-processing
  logical, intent(in) :: F_satuco_L                     !include saturation wrt ice
  logical, intent(in) :: F_init_L                       !write heading
  logical, intent(in) :: F_wr_L                         !write time series values

  !@author Ron McTaggart-Cowan, 2009-04

  !@revisions
  !  2009-04,  Ron McTaggart-Cowan: update from serwrit3.ftn

  !@description
  !  See SERDBU for more information about the functionality
  !  of this subprogram.  This subprogram is a member of the RPN
  !  Physics package interface, and has an explicit interface
  !  specified in itf_physics_f.h for inclusion by external 
  !  packages that call the Physics.
  !*@/

#include "series.cdk"

  type(vgrid_descriptor) :: vcoord
  integer :: k,l,m,err,phydim_nk
  integer, dimension(:), pointer :: ip1t
  real*8, dimension(:,:,:), pointer :: vtbl
!
!     ---------------------------------------------------------------
!
  ! Return if not initialized or writing
  if ( nstat <= 0 .and. .not.F_wr_L) return
  if (.not. initok) return
  phydim_nk=F_nhybm-1
  ! Write header on request
  if (F_init_L .and. F_wr_L) then

     nullify(ip1t,vtbl)
     err = vgrid_wb_get('ref-t',vcoord,ip1t)
     deallocate(ip1t); nullify(ip1t)
     err= vgd_get(vcoord,'VTBL - vgrid_descriptor table',vtbl)

     write(noutser) nstat_g, nsurf, nprof, phydim_nk
     !print *,'ser_write1: nstat_g=',nstat_g,'nsurf=',nsurf,'nprof=',nprof,'phydim_nk=',phydim_nk
     write(noutser) size(vtbl,1),size(vtbl,2),size(vtbl,3)
     !print *,'ser_write2: size of vtbl()', size(vtbl,1),size(vtbl,2),size(vtbl,3)
     write(noutser) vtbl
     !print *,'ser_write3: write out vtbl'

     write(noutser) nstat_g                          , &
          (name(l),istat_g(l),jstat_g(l),l=1,nstat_g), &
          nsurf,(surface(m,2),m=1,nsurf)             , &
          nprof,(profils(m,2),m=1,nprof)             , &
          (F_date(k),k=1,14),F_etiket,(F_ig(k),k=1,4), &
          F_dgrw,F_rgas,F_grav,F_satues_L,F_satuco_L,tsmoyhr,srwri
     !print *,'ser_write4: write out rest of stuff surface(1)=',surface(1,2)
     write(6,*) ' ---ENTETE DE SERIES ECRITE SUR ',noutser

  endif

  ! Write time series

  if (mod(kount,serint) == 0) then

     if (F_wr_L) then
        !print *,'ser_write5: write out data of sers(1,1)=',sers(1,1)
        write(noutser) heure, ((sers(l,m),l=1,nstat_g),m=1,nsurf), &
            (((serp(k,l,m),k=1,phydim_nk),l=1,nstat_g),m=1,nprof)
     endif
     !print *,'ser_write5: and then zap data to 0.'

     surface(1:nsurf,2) = ''
     do l=1,nstat
        sers(statnum(l),1:nsurf) = 0.
     enddo
     profils(1:nprof,2) = ''
     do l=1,nstat
        serp(1:phydim_nk,statnum(l),1:nprof) = 0.
     enddo

  endif
!
!     ---------------------------------------------------------------
!
  return
end subroutine ser_write2

