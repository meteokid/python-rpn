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

module phy_init_mod
   use phy_typedef, only: PHY_NONE, NPATH_DEFAULT, BPATH_DEFAULT
   use phy_options
   use timestr_mod
   use cnv_options
   private
   public :: phy_init

   interface phy_init
      module procedure phy_init_2grids
      module procedure phy_init_4grids
   end interface phy_init

#include <arch_specific.hf>
#include <WhiteBoard.hf>
#include <rmnlib_basics.hf>
#include <msg.h>
   include "phygrd.cdk"
   include "tables.cdk"
   include "surface.cdk"

contains

   !/@*
   function phy_init_2grids(F_path_S, F_dateo, F_dt, F_phygrid_S, &
        F_lclcore_S, F_nk, F_std_pres) result(F_istat)
      implicit none

      character(len=*), intent(in) :: F_path_S
      character(len=*), intent(in) :: F_phygrid_S,F_lclcore_S
      integer, intent(in) :: F_dateo
      integer, intent(in) :: F_nk
      real,    intent(in) :: F_dt, F_std_pres(F_nk)
      integer :: F_istat !Return status

      !@authors Desgagne, Chamberland, McTaggart-Cowan, Spacek -- Spring 2014

      !@revision

      !*@/

      F_istat = phy_init_4grids(F_path_S, F_dateo, F_dt, F_phygrid_S , &
           F_lclcore_S, 'NULL' ,'NULL', F_nk, F_std_pres)

      return
   end function phy_init_2grids


   !/@*
   function phy_init_4grids(F_path_S, F_dateo, F_dt, F_phygrid_S , &
        F_lclcore_S, F_drv_glb_S, F_drv_lcl_S, F_nk, F_std_pres) result(F_istat)
      use hgrid_wb, only: hgrid_wb_get
      implicit none
      
      character(len=*), intent(in) :: F_path_S
      character(len=*), intent(in) :: F_phygrid_S,F_lclcore_S,F_drv_glb_S, F_drv_lcl_S
      integer, intent(in) :: F_dateo
      integer, intent(in) :: F_nk
      real,    intent(in) :: F_dt, F_std_pres(F_nk)
      integer :: F_istat !Return status

      !@authors Desgagne, Chamberland, McTaggart-Cowan, Spacek -- Spring 2014

      !@revision

      !*@/
      integer, external :: msg_getUnit, phydebu2, sfc_init, itf_cpl_init

      logical :: print_L
      integer :: bidon,unout,options
      integer :: ier,p_ni,p_nj,master_pe
      ! --------------------------------------------------------------------

      F_istat = -1

      if (phy_init_ctrl /= PHY_CTRL_NML_OK) then
         if (phy_init_ctrl == PHY_NONE .or. &
             phy_init_ctrl == PHY_CTRL_INI_OK) then
            F_istat = PHY_NONE
         else
            call msg(MSG_ERROR, '(phy_init) Must call phy_nml first!')
         endif
         return
      endif
      phy_init_ctrl = PHY_ERROR

      unout   = msg_getUnit(MSG_INFO)
      print_L = (unout > 0)

      ier = newdate(F_dateo,date, bidon,RMN_DATE_STAMP2OLD)
      if (.not.RMN_IS_OK(ier)) then
         call msg(MSG_ERROR,'(phy_init) Problem converting datao')
         return
      endif

      delt   = F_dt
      ier    = timestr2step(kntrad, kntrad_S, dble(delt))
      kntrad = max(1,kntrad)

      ier = wb_get('itf_phy/DYNOUT', dynout)
      if (.not.WB_IS_OK(ier)) dynout = .false.

      ier = wb_get('itf_phy/VSTAG', vstag)
      if (.not.WB_IS_OK(ier)) vstag = .false.

      ier = wb_get('itf_phy/TLIFT', Tlift)
      if (.not.WB_IS_OK(ier)) Tlift = 0

      ! Store local core grid information
      ier = hgrid_wb_get(F_lclcore_S, phy_lclcore_gid)
      if (.not.RMN_IS_OK(ier)) then
         call msg (MSG_ERROR,'(phy_init) Unable to retrieve grid info for '&
              //trim(F_lclcore_S))
         return
      endif

      ! Establish physics grid information
      ier = hgrid_wb_get(F_phygrid_S, phy_lcl_gid, F_i0=phy_lcl_i0, &
           F_j0=phy_lcl_j0, F_lni=phy_lcl_ni, F_lnj=phy_lcl_nj)
      if (.not.RMN_IS_OK(ier)) then
         call msg (MSG_ERROR,'(phy_init) Unable to retrieve grid info for '&
              //trim(F_phygrid_S))
         return
      endif

      phy_lcl_in = phy_lcl_i0 + phy_lcl_ni - 1
      phy_lcl_jn = phy_lcl_j0 + phy_lcl_nj - 1

      if (F_drv_glb_S /= 'NULL') then
         ier = hgrid_wb_get(F_drv_glb_S, drv_glb_gid, &
              F_lni=drv_glb_ni, F_lnj=drv_glb_nj)
         if (.not.RMN_IS_OK(ier)) then
            call msg (MSG_ERROR,'(phy_init) Unable to retrieve grid info for '&
                 //trim(F_drv_glb_S))
            return
         endif
      else
         drv_glb_gid = -99
      endif

      if (F_drv_lcl_S /= 'NULL') then
         ier = hgrid_wb_get(F_drv_lcl_S, drv_lcl_gid, F_i0=drv_lcl_i0, &
              F_j0=drv_lcl_j0, F_lni=drv_lcl_ni, F_lnj=drv_lcl_nj)
         if (.not.RMN_IS_OK(ier)) then
            call msg (MSG_ERROR,'(phy_init) Unable to retrieve grid info for '&
                 //trim(F_drv_lcl_S))
            return
         endif
         drv_lcl_in = drv_lcl_i0 + drv_lcl_ni - 1
         drv_lcl_jn = drv_lcl_j0 + drv_lcl_nj - 1
      else
         drv_lcl_gid = -99
      endif

      if (p_runlgt <= 0) p_runlgt = phy_lcl_ni
      p_runlgt = min(phy_lcl_ni*phy_lcl_nj,max(1,p_runlgt))
      p_ni = p_runlgt
      p_nj = phy_lcl_ni*phy_lcl_nj/p_ni
      if (p_ni*p_nj < phy_lcl_ni*phy_lcl_nj) p_nj = p_nj + 1

      phydim_ni = p_ni
      phydim_nj = p_nj
      phydim_nk = F_nk

      ier = phydebu2(p_ni, p_nj, F_nk, F_path_S)

      if ((moyhr>0 .or. acchr>0) .and. dynout) then
         call msg(MSG_ERROR,'(phy_init) cannot use MOYHR nor ACCHR with sortie_p(average/accum)')
         return
      endif

      if (.not.RMN_IS_OK(ier)) then
         call msg(MSG_ERROR,'(phy_init) Problem in phydebu')
         return
      endif

      allocate(std_p_prof(F_nk))
      std_p_prof = F_std_pres

      options = WB_REWRITE_NONE+WB_IS_LOCAL
      ier = WB_OK
      ier = min(wb_put('phy/date'    ,date     , options),ier)
      ier = min(wb_put('phy/climat'  ,climat   , options),ier)
      ier = min(wb_put('phy/convec'  ,convec   , options),ier)
      ier = min(wb_put('phy/delt'    ,delt     , options),ier)
      ier = min(wb_put('phy/fluvert' ,fluvert  , options),ier)
      ier = min(wb_put('phy/radslope',radslope , options),ier)
      ier = min(wb_put('phy/radia'   ,radia    , options),ier)
      ier = min(wb_put('phy/test_phy',test_phy , options),ier)
      if (.not.WB_IS_OK(ier)) then
         call msg(MSG_ERROR,'(phy_init) Problem with WB_put')
         return
      endif

      ier = sfc_init(p_ni, p_nj, F_nk)
      if (.not.RMN_IS_OK(ier)) then
         call msg(MSG_ERROR,'(phy_init) Problem in sfc_debu')
         return
      endif

      ier = WB_OK
      ier = min(wb_get('sfc/as'          ,as          ),ier)
      ier = min(wb_get('sfc/beta'        ,beta        ),ier)
      ier = min(wb_get('sfc/ci'          ,ci          ),ier)
      ier = min(wb_get('sfc/critlac'     ,critlac     ),ier)
      ier = min(wb_get('sfc/critmask'    ,critmask    ),ier)
      ier = min(wb_get('sfc/critsnow'    ,critsnow    ),ier)
      ier = min(wb_get('sfc/drylaps'     ,drylaps     ),ier)
      ier = min(wb_get('sfc/impflx'      ,impflx      ),ier)
      ier = min(wb_get('sfc/indx_soil'   ,indx_soil   ),ier)
      ier = min(wb_get('sfc/indx_glacier',indx_glacier),ier)
      ier = min(wb_get('sfc/indx_water'  ,indx_water  ),ier)
      ier = min(wb_get('sfc/indx_ice'    ,indx_ice    ),ier)
      ier = min(wb_get('sfc/indx_urb'    ,indx_urb    ),ier)
      ier = min(wb_get('sfc/indx_agrege' ,indx_agrege ),ier)
      ier = min(wb_get('sfc/leadfrac'    ,leadfrac    ),ier)
      ier = min(wb_get('sfc/n0rib'       ,n0rib       ),ier)
      ier = min(wb_get('sfc/tdiaglim'    ,tdiaglim    ),ier)
      ier = min(wb_get('sfc/vamin'       ,vamin       ),ier)
      ier = min(wb_get('sfc/veg_rs_mult' ,veg_rs_mult ),ier)
      ier = min(wb_get('sfc/z0dir'       ,z0dir       ),ier)
      ier = min(wb_get('sfc/zt'          ,zt          ),ier)
      ier = min(wb_get('sfc/zta'         ,zta         ),ier)
      ier = min(wb_get('sfc/zu'          ,zu          ),ier)
      ier = min(wb_get('sfc/zua'         ,zua         ),ier)
      if (.not.WB_IS_OK(ier)) then
         call msg(MSG_ERROR,'(phy_init) Problem with WB_get #2')
         return
      endif

      ier = WB_OK
      ier = min(wb_put('phy/zu',zu,options),ier)
      ier = min(wb_put('phy/zt',zt,options),ier)

      if (.not.WB_IS_OK(ier)) then
         call msg(MSG_ERROR,'(phy_init) Problem with WB_put #2')
         return
      endif

      if (print_L) then
         call printbus('E')
         call printbus('D')
         call printbus('P')
         call printbus('V')
      endif

      call mapping2drivergrid()

      ier = wb_get('model/outout/pe_master', master_pe)

      call ser_init5(phy_lcl_gid, drv_glb_gid, phydim_ni, phydim_nj, phydim_nk,&
           p_runlgt, phy_lcl_ni, phy_lcl_nj, moyhr, delt, master_pe)

      phy_init_ctrl = PHY_CTRL_INI_OK
      F_istat = itf_cpl_init(F_path_S, print_L, unout, F_dateo, F_dt)

      ! --------------------------------------------------------------------
      return
   end function phy_init_4grids

end module phy_init_mod
