!_______________________________________________________________________________________________
!                                                                                               !
! This module does NOT contains the Predicted Particle Property (P3) bulk microphysics scheme.  !
! The option STCOND = MP_P3 is not available in this version of RPNPHY.  This dummy module is   !
! here as a placeholder, replacing an obsolete version of P3 which was removed.  The updated    !
! version of the scheme, and the option to run GEM with MP_P3, is available in PHY_6.0          !
!_______________________________________________________________________________________________!

 MODULE MP_P3

 contains


 SUBROUTINE p3_init(lookup_file_1,lookup_file_2)
! Passed arguments:
 character*(*), intent(in) :: lookup_file_1    !lookup table for main processes
 character*(*), intent(in) :: lookup_file_2    !lookup table for ice category interactions

  print*
  print*, '***     ABORT     *** '
  print*
  print*, 'The option STCOND = MP_P3 is not availabe in this version of GEM/RPNPHY.'
  print*, 'P3 is available in RPNPHY_6.0 (for GEM_4.8.3 and GEM_5.0).  '
  print*, 'To obtain the necessary code to use P3 in GEM_4.8-LTS,       '
  print*, 'please contact jason.milbrandt@canada.ca'
  print*
  print*, '***     ABORT     *** '
  print*
  stop

 END SUBROUTINE p3_init

 SUBROUTINE mp_p3_wrapper_gem(qvap,temp,dt,ww,psfc,sigma,kount,trnch,ni,nk,prt_liq,      &
                              prt_sol,diag_Zet,diag_Zec,diag_effc,diag_effi,diag_vmi,    &
                              diag_di,diag_rhopo,qc,qr,nr,n_iceCat,                      &
                              qitot_1,qirim_1,nitot_1,birim_1,                           &
                              qitot_2,qirim_2,nitot_2,birim_2,                           &
                              qitot_3,qirim_3,nitot_3,birim_3,                           &
                              qitot_4,qirim_4,nitot_4,birim_4)

!------------------------------------------------------------------------------------------!
! This wrapper subroutine is the main GEM interface with the P3 microphysics scheme.  It   !
! prepares some necessary fields (converts temperature to potential temperature, etc.),    !
! passes 2D slabs (i,k) to the main microphysics subroutine ('P3_MAIN') -- which updates   !
! the prognostic variables (hydrometeor variables, temperature, and water vapor) and       !
! computes various diagnostics fields (precipitation rates, reflectivity, etc.) -- and     !
! finally converts the updated potential temperature to temperature.                       !
!------------------------------------------------------------------------------------------!

 implicit none

!----- input/ouput arguments:  ----------------------------------------------------------!

 integer, intent(in)                    :: ni                    ! number of columns in slab           -
 integer, intent(in)                    :: nk                    ! number of vertical levels           -
 integer, intent(in)                    :: n_iceCat              ! number of ice categories            -
 integer, intent(in)                    :: kount                 ! time step counter                   -
 integer, intent(in)                    :: trnch                 ! number of slice                     -
 real, intent(in)                       :: dt                    ! model time step                     s
 real, intent(inout), dimension(ni,nk)  :: qc                    ! cloud mixing ratio, mass            kg kg-1
 real, intent(inout), dimension(ni,nk)  :: qr                    ! rain  mixing ratio, mass            kg kg-1
 real, intent(inout), dimension(ni,nk)  :: nr                    ! rain  mixing ratio, number          #  kg-1
 real, intent(inout), dimension(ni,nk)  :: qitot_1               ! ice   mixing ratio, mass (total)    kg kg-1
 real, intent(inout), dimension(ni,nk)  :: qirim_1               ! ice   mixing ratio, mass (rime)     kg kg-1
 real, intent(inout), dimension(ni,nk)  :: nitot_1               ! ice   mixing ratio, number          #  kg-1
 real, intent(inout), dimension(ni,nk)  :: birim_1               ! ice   mixing ratio, volume          m3 kg-1

 real, intent(inout), dimension(ni,nk), optional  :: qitot_2     ! ice   mixing ratio, mass (total)    kg kg-1
 real, intent(inout), dimension(ni,nk), optional  :: qirim_2     ! ice   mixing ratio, mass (rime)     kg kg-1
 real, intent(inout), dimension(ni,nk), optional  :: nitot_2     ! ice   mixing ratio, number          #  kg-1
 real, intent(inout), dimension(ni,nk), optional  :: birim_2     ! ice   mixing ratio, volume          m3 kg-1

 real, intent(inout), dimension(ni,nk), optional  :: qitot_3     ! ice   mixing ratio, mass (total)    kg kg-1
 real, intent(inout), dimension(ni,nk), optional  :: qirim_3     ! ice   mixing ratio, mass (rime)     kg kg-1
 real, intent(inout), dimension(ni,nk), optional  :: nitot_3     ! ice   mixing ratio, number          #  kg-1
 real, intent(inout), dimension(ni,nk), optional  :: birim_3     ! ice   mixing ratio, volume          m3 kg-1

 real, intent(inout), dimension(ni,nk), optional  :: qitot_4     ! ice   mixing ratio, mass (total)    kg kg-1
 real, intent(inout), dimension(ni,nk), optional  :: qirim_4     ! ice   mixing ratio, mass (rime)     kg kg-1
 real, intent(inout), dimension(ni,nk), optional  :: nitot_4     ! ice   mixing ratio, number          #  kg-1
 real, intent(inout), dimension(ni,nk), optional  :: birim_4     ! ice   mixing ratio, volume          m3 kg-1

 real, intent(inout), dimension(ni,nk)  :: qvap                  ! vapor  mixing ratio, mass           kg kg-1
 real, intent(inout), dimension(ni,nk)  :: temp                  ! temperature                         K
 real, intent(in),    dimension(ni)     :: psfc                  ! surface air pressure                Pa
 real, intent(in),    dimension(ni,nk)  :: sigma                 ! sigma = p(k,:)/psfc(:)
 real, intent(in),    dimension(ni,nk)  :: ww                    ! vertical motion                     m s-1
 real, intent(out),   dimension(ni)     :: prt_liq               ! precipitation rate, liquid          m s-1
 real, intent(out),   dimension(ni)     :: prt_sol               ! precipitation rate, solid           m s-1
 real, intent(out),   dimension(ni,nk)  :: diag_Zet              ! equivalent reflectivity, 3D         dBZ
 real, intent(out),   dimension(ni)     :: diag_Zec              ! equivalent reflectivity, col-max    dBZ
 real, intent(out),   dimension(ni,nk)  :: diag_effc             ! effective radius, cloud             m
 real, intent(out),   dimension(ni,nk)  :: diag_effi             ! effective radius, ice               m
 real, intent(out),   dimension(ni,nk)  :: diag_di               ! mean-mass diameter, ice             m
 real, intent(out),   dimension(ni,nk)  :: diag_vmi              ! fall speed (mass-weighted), ice     m s-1
 real, intent(out),   dimension(ni,nk)  :: diag_rhopo            ! bulk density, ice                   kg m-3

 end SUBROUTINE mp_p3_wrapper_gem


 END MODULE MP_P3
