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

module phy_input_mod
   use phy_typedef, only: phymeta, PHY_NONE, NPATH_DEFAULT, BPATH_DEFAULT
   use iso_c_binding
   use vGrid_Descriptors
   use vgrid_wb
   use input_mod
   use ezgrid_mod
   use statfld_dm_mod
   use phy_options
   use phygetmetaplus_mod, only: phymetaplus, phygetmetaplus
   private
   public :: phy_input

#include <arch_specific.hf>
#include <rmnlib_basics.hf>
#include <clib_interface_mu.hf>
#include <gmm.hf>
#include <msg.h>
   include "rpn_comm.inc"
   include "phyinput.cdk"
   include "phygrd.cdk"

   integer, external :: phyent2per, phyfillbus, phyfold2

   integer, parameter :: STAT_PRECISION = 8
   logical, parameter :: FSTOPC_SET = .false.
   integer, parameter :: NVARMAX = 80
   integer, parameter :: MUST_INIT = 1

contains

   !/@*
   function phy_input(pre_fold_opr_clbk,F_step,F_incfg_S,F_basedir_S,F_geoname_S) result(F_istat)
      implicit none
      !@objective 
      !@arguments
      integer,external :: pre_fold_opr_clbk
      integer,intent(in) :: F_step
      character(len=*) :: F_incfg_S    !- physics_input_table path
      character(len=*) :: F_basedir_S  !- base path for input data file
      character(len=*) :: F_geoname_S  !- name of geophys file
      !@return
      integer :: F_istat
      !@author Michel Desgagne - Spring 2011
      !@revision
      !  2011-06 Stephane Chamberland 
      !  2015-09 Stephane Chamberland: use phy_getmeta
      !*@/

      logical, save :: is_init_L = .false.
      integer, save :: inputid = -1
      integer, save :: nbvar = 0

      integer :: ivar, istat, tmidx, nread
      integer :: dateo, idt, ismandatory
      real, pointer, dimension(:,:)   :: refp0, pw_p0
      real, pointer, dimension(:,:,:) :: data, data2
      character(len=4) :: inname_S, inname2_S
      character(len=32) :: varname_S, varname2_S, readlist_S(PHYINREAD_MAX), horiz_interp_S, vgrid_S
      character(len=512) :: dummylist_S(10)
      type(vgrid_descriptor) :: vgridm, vgridt
      type(gmm_metadata) :: mymeta
      type(phymeta) :: meta1, meta2
      type(phymetaplus) :: meta1plus, meta2plus
      integer, pointer :: ip1list(:),ip1list2(:)
      real :: vmin, vmax
      ! ---------------------------------------------------------------------
      F_istat = RMN_ERR
      if (phy_init_ctrl == PHY_NONE) then
         F_istat = PHY_NONE
         return
      else if (phy_init_ctrl /= PHY_CTRL_INI_OK) then
         call msg(MSG_ERROR,'(phy_input) Physics not properly initialized.')
         return
      endif


      istat = fstopc('MSGLVL','INFORM',FSTOPC_SET)

      ! Retrieve input from the model dynamics into the dynamics bus
      istat = phyfillbus(F_step)
      if (.not.RMN_IS_OK(istat)) then
         call msg(MSG_ERROR,'(phy_input) problem filling buses')
         return
      endif

      istat = rpn_comm_bloc(ninblocx,ninblocy)

      dateo = date(14)
      idt   = nint(delt)

      nullify(pw_p0)
      istat = gmm_get('PW_P0:M',pw_p0)

      F_istat = RMN_OK
      IF_INIT: if (.not.is_init_L) then
         is_init_L = .true.
         if (any((/phydim_ni,phydim_nj/) == 0)) then
            call msg(MSG_ERROR,'(phy_input) problem getting bus size')
            F_istat = RMN_ERR
         endif
         inputid = input_new(dateo,idt,F_incfg_S)
         istat = inputid
         if (.not.RMN_IS_OK(inputid)) then
            call msg(MSG_ERROR,'(phy_input) problem initializing physic input')
         else
            call phyinputdiag(inputid)
            nbvar = input_nbvar(inputid)
            istat = min(input_set_basedir(inputid,F_basedir_S),istat)
            istat = min(input_set_filename(inputid,'geop',F_geoname_S,.false.,INPUT_FILES_GEOP),istat)
         endif
         F_istat = min(istat,F_istat)
         nullify(ip1list)
         istat = vgrid_wb_get('ref-m',vgridm,ip1list)
         ip1list2 => ip1list
         if (size(ip1list) > phydim_nk) then
            !#TODO: check: should we keep phydim_nk instead of phydim_nk+1 (diag level)
            ip1list2(phydim_nk) = ip1list2(phydim_nk+1)
            ip1list2 => ip1list(1:phydim_nk)
         endif
         istat = vgrid_wb_put('phy-m',vgridm,ip1list2,'PHYREFP0:M', &
              F_overwrite_L=.true.)
         nullify(ip1list, ip1list2)
         istat = vgrid_wb_get('ref-t',vgridt,ip1list)
         ip1list2 => ip1list
         if (size(ip1list) > phydim_nk) then
            !#TODO: check: should we keep phydim_nk instead of phydim_nk+1 (diag level)
            ip1list2(phydim_nk) = ip1list2(phydim_nk+1)
            ip1list2 => ip1list(1:phydim_nk)
         endif
         istat = vgrid_wb_put('phy-t',vgridt,ip1list2,'PHYREFP0:M', &
              F_overwrite_L=.true.)
         mymeta = GMM_NULL_METADATA
         mymeta%l(1) = gmm_layout(1,phy_lcl_ni,0,0,phy_lcl_ni)
         mymeta%l(2) = gmm_layout(1,phy_lcl_nj,0,0,phy_lcl_nj)
         nullify(refp0)
         istat = gmm_create('PHYREFP0:M',refp0,mymeta)
         call collect_error(F_istat)
      endif IF_INIT
      if (.not.RMN_IS_OK(F_istat)) return

      nullify(refp0)
      istat = gmm_get('PHYREFP0:M',refp0)

      if (associated(refp0) .and. associated(pw_p0)) then
         refp0(:,:) = pw_p0(phy_lcl_i0:phy_lcl_in,phy_lcl_j0:phy_lcl_jn)
      endif

      phyinread_n = 0
      phyinread_dateo = dateo
      phyinread_dt = idt
      phyinread_step = F_step
      nread = 0
      readlist_S(:) = ' '

      call priv_ozone(F_step)

      call chm_load_emissions(F_basedir_S,dateo,idt,phy_lcl_ni,phydim_ni,F_step)

      tmidx = -1
      F_istat = RMN_OK

      istat = input_setgridid(inputid,phy_lclcore_gid)
!!$      if (F_step == 0) BUSENT3d = 0. !# Now done in phydebu
      VARLOOP: do ivar=1,nbvar
         istat = input_isvarstep(inputid,ivar,F_step)
         if (.not.RMN_IS_OK(istat)) then
            cycle VARLOOP !var not requested at this step
         endif
         istat = input_meta(inputid,ivar,inname_S,inname2_S,dummylist_S,horiz_interp_S,F_mandatory=ismandatory, F_vmin=vmin, F_vmax=vmax)
         if (.not.RMN_IS_OK(istat)) then
            call msg(MSG_ERROR,'(phy_input) problem getting input varname')
            cycle VARLOOP
         endif

         istat = phygetmetaplus(meta1plus,inname_S, F_npath='IOV', &
              F_bpath='EDPV', F_quiet=.true., F_shortmatch=.false.)
         meta1 = meta1plus%meta
         if (RMN_IS_OK(istat) .and. inname2_S /= ' ') then
            istat = phygetmetaplus(meta2plus,inname2_S, F_npath='IOV', &
                 F_bpath='EDPV', F_quiet=.true., F_shortmatch=.false.)
            meta2 = meta2plus%meta
            varname2_S = meta2%vname
         else
            varname2_S = ' '
         endif
         if (.not.RMN_IS_OK(istat)) then
            call msg(MSG_INFO,'(phy_input) ignoring var, not declared in bus: '//trim(inname_S)//' : '//trim(inname2_S))
            cycle VARLOOP !# var not needed
         endif

         varname_S  = meta1%vname
         if (ismandatory == -1) ismandatory = meta1%init

         vgrid_S = 'phy-m' !#SLB
         if (meta1%stag == 1) vgrid_S = 'phy-t' !#SLC / SLS
         nullify(data,data2)
         istat = input_get(inputid,ivar,F_step,phy_lcl_gid,vgrid_S,data,data2)

         if (.not.(RMN_IS_OK(istat).and. &
              associated(data).and.&
              (inname2_S == ' ' .or. associated(data2)))) then
            if (associated(data)) deallocate(data,stat=istat)
            if (associated(data2)) deallocate(data2,stat=istat)
            if (ismandatory == 0) then
               call msg(MSG_WARNING,'(phy_input) missing optional var: '//trim(inname_S)//' : '//trim(varname_S)//' ('//trim(inname2_S)//' : '//trim(varname2_S)//')')
               cycle VARLOOP
            endif
            call msg(MSG_ERROR,'(phy_input) missing var: '//trim(inname_S)//' : '//trim(varname_S))
            F_istat = RMN_ERR
            cycle VARLOOP
         endif

         if (inname_S == 'tm') tmidx = meta1plus%index

         vmin = max(vmin, meta1%vmin)
         vmax = min(vmax, meta1%vmax)
         F_istat = min(priv_fold(F_step,varname_S,inname_S,meta1%bus,data,readlist_S,nread,horiz_interp_S,meta1%n(3),vmin,vmax,pre_fold_opr_clbk),F_istat)
         if (inname2_S /= ' ') then
            F_istat = min(priv_fold(F_step,varname2_S,inname2_S,meta2%bus,data2,readlist_S,nread,horiz_interp_S,meta2%n(3),vmin,vmax,pre_fold_opr_clbk),F_istat)
         endif

         if (associated(data)) deallocate(data,stat=istat)
         if (associated(data2)) deallocate(data2,stat=istat)

      enddo VARLOOP

      F_istat = min(priv_checklist(readlist_S,nread,F_step),F_istat)

      if ((RMN_IS_OK(F_istat)).and.(nread > 0)) then
         call msg(MSG_INFO,'(phy_input) All needed var were found')
      endif

      F_istat = min(phyent2per(readlist_S,nread,F_step),F_istat)

      phyinread_n = nread
      if (nread > 0) phyinread_list_S(1:nread) = readlist_S(1:nread)

      ! ---------------------------------------------------------------------
      return
   end function phy_input


   !/@*
   subroutine priv_ozone(my_step)
      implicit none
      !@objective check for daily update to climatological ozone
      integer,intent(in) :: my_step
      !*@/
      integer :: istat,aujour,curdate,curdd,curmo,part1,part2,ppjour
      real(8) :: hours
      ! ---------------------------------------------------------------------
      if (intozot .and. my_step > 1) then
         aujour = 1 
         ppjour = nint(86400./delt)
         if (ppjour > 0) aujour = mod(my_step, ppjour)

         if(aujour == 1)then
            hours = my_step/(3600./dble(delt))
            call incdatr(curdate,date(14), hours)
            istat = newdate(curdate,part1,part2,RMN_DATE_STAMP2PRINT)
            if (istat == 0) then
               curdd = mod(part1,100)
               curmo = mod(part1/100,100)
               call intozon(curdd, curmo, RMN_STDOUT)
            endif
         endif
      endif
      ! ---------------------------------------------------------------------
      return
   end subroutine priv_ozone


   !/@*
   function priv_fold(my_step,my_varname_S,my_inname_S,my_bus_S,my_data,my_readlist_S,my_nread,my_horiz_interp_S,my_nk,my_vmin,my_vmax,pre_fold_opr_clbk) result(my_istat)
      implicit none
      character(len=*) :: my_varname_S,my_inname_S,my_bus_S,my_readlist_S(:),my_horiz_interp_S
      real, dimension(:,:,:), pointer :: my_data
      integer :: my_step,my_nread,my_nk,my_istat
      real    :: my_vmin, my_vmax
      integer,external :: pre_fold_opr_clbk
      !*@/
      character(len=64) :: msgFormat_S,msg_S
      integer :: minxyz(3),maxxyz(3),msgLevelMin,msgUnit,istat,k
      logical :: canWrite_L
      ! ---------------------------------------------------------------------
      minxyz = lbound(my_data)
      maxxyz = ubound(my_data)
      call physimple_transforms3d2(my_varname_S,my_data,maxxyz(1),maxxyz(2),maxxyz(3))
      my_istat = pre_fold_opr_clbk(my_data,my_varname_S,my_horiz_interp_S,minxyz(1),maxxyz(1),minxyz(2),maxxyz(2),minxyz(3),maxxyz(3))

!!$      print *,'(phy_input) 0 :'//trim(my_varname_S),my_vmin,my_vmax,minval(my_data),maxval(my_data)
      my_data = min(max(my_vmin, my_data), my_vmax)
!!$      print *,'(phy_input) 1 :'//trim(my_varname_S),my_vmin,my_vmax,minval(my_data),maxval(my_data)

      call msg_getInfo(canWrite_L,msgLevelMin,msgUnit,msgFormat_S)
      if (MSG_INFOPLUS >= msgLevelMin) then
         do k=lbound(my_data,3),ubound(my_data,3)
            write(msg_S,'(a,i4.4)') trim(my_inname_S)//' => '//trim(my_varname_S)//' ',k
            call statfld_dm(my_data(:,:,k:k),msg_S,my_step,'phy_input',STAT_PRECISION)
         enddo
      endif
      if (maxxyz(3) == minxyz(3) .and. my_nk > 1) then
         !# Put read data into diag level
         maxxyz(3) = my_nk
         minxyz(3) = maxxyz(3)
      endif
      !#TODO: re-use metadata from above, use phyfoldmeta
      my_istat = min(phyfold2(my_data,trim(my_varname_S),trim(my_bus_S),minxyz,maxxyz), my_istat)
      if (RMN_IS_OK(my_istat)) then
         my_nread = min(my_nread + 1,size(my_readlist_S))
         my_readlist_S(my_nread) = my_varname_S
         istat = clib_tolower(my_readlist_S(my_nread))
      endif
      ! ---------------------------------------------------------------------
      return
   end function priv_fold


   !/@*
   function priv_checklist(F_readlist_S,F_nread,F_step) result(F_istat)
      implicit none
      !@objective Check if all needed var are read
      integer,intent(in) :: F_nread,F_step
      character(len=*) :: F_readlist_S(:)
      integer :: F_istat
      !*@/
      logical,parameter:: NOSHORTMATCH_L = .false.
      integer :: nvars,ivar,istat
      type(phymetaplus), pointer :: metalist(:)
      ! ---------------------------------------------------------------------
      F_istat = RMN_OK
      if (F_step /= 0) return

      nullify(metalist)
      nvars = phygetmetaplus(metalist, F_name=' ', F_npath='V', F_bpath='E', &
           F_maxmeta=NVARMAX, F_quiet=.true., F_shortmatch=NOSHORTMATCH_L)

      do ivar = 1,nvars
         istat = clib_tolower(metalist(ivar)%meta%vname)
      end do

      do ivar=1,F_nread
         istat = clib_tolower(F_readlist_S(ivar))
      enddo

      do ivar = 1,nvars
         if (metalist(ivar)%meta%init == MUST_INIT) then
            if (F_nread == 0 .or. &
                 .not.any(F_readlist_S(1:F_nread) == metalist(ivar)%meta%vname)) then
               F_istat = RMN_ERR
               call msg(MSG_ERROR,'(phy_input) Missing mandatory var (physics_input_table missing entry?): '//trim(metalist(ivar)%meta%vname))
            endif
         endif
      end do

      deallocate(metalist,stat=istat)
      ! ---------------------------------------------------------------------
      return
   end function priv_checklist


end module phy_input_mod
