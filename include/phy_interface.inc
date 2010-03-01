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
!/**
  !@objective Define interface for the RPN Physics package

  !@author

  !@revisions

  !@description
  !  This include file contains explicit interfaces for all entry
  !  points to the RPN Physics package.
  !**/

      interface phy_serwrit4
         subroutine serwrit4_priv (F_date,F_etiket,F_nk,F_ig,F_dgrw,		&
     &        F_hybm,F_hybt,F_am_8,F_bm_8,F_at_8,F_bt_8,F_rcoef,                &
     &        F_ptop_8,F_pref_8,F_vcode,F_rgas_8,F_grav_8,F_satues_L,		&
     &        F_satuco_L,F_init_L,F_wr_L,F_step1out_L)
            integer, dimension(:), intent(in) :: F_date
            character(len=*), intent(in) :: F_etiket
            integer, intent(in) :: F_nk
            integer, dimension(:), intent(in) :: F_ig
            real, intent(in) :: F_dgrw
            real, dimension(:), intent(in) :: F_hybm
            real, dimension(:), intent(in) :: F_hybt
            real(kind=8), dimension(:), intent(in) :: F_am_8
            real(kind=8), dimension(:), intent(in) :: F_bm_8
            real(kind=8), dimension(:), intent(in) :: F_at_8
            real(kind=8), dimension(:), intent(in) :: F_bt_8
            real, dimension(:), intent(in) :: F_rcoef
            real(kind=8), intent(in) :: F_ptop_8
            real(kind=8), intent(in) :: F_pref_8
            integer, intent(in) :: F_vcode
            real(kind=8), intent(in) :: F_rgas_8
            real(kind=8), intent(in) :: F_grav_8
            logical, intent(in) :: F_satues_L
            logical, intent(in) :: F_satuco_L
            logical, intent(in) :: F_init_L
            logical, intent(in) :: F_wr_L
            logical, intent(in) :: F_step1out_L
	 end subroutine serwrit4_priv
      end interface
