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
module canonical
   implicit none
   public
   save

!______________________________________________________________________
!                                                                      |
! GMM variables for Canonical cases Williamson and DCMIP               |
!______________________________________________________________________|
!                    |                                                 |
! NAME               | DESCRIPTION                                     |
!--------------------|-------------------------------------------------|
! pth                | Perturbation real potential temperature         |
! thbase             | Basic potential temperature                     |
! dtv                | Perturbation: Virtual temperature - Tstar       |
!----------------------------------------------------------------------|
! cly                | cl + 2*cl2                                      |
! acl                | Average column integrated of cl                 |
! acl2               | Average column integrated of cl2                |
! acly               | Average column integrated of cly                |
!----------------------------------------------------------------------|
! irt                | Instantaneous precipitation rate                |
! art                | Averaged precipitation rate                     |
! wrt                | Averaged precipitation rate (WORK FIELD)        |
!----------------------------------------------------------------------|
! uref               | U initial Schaer/Vertical Diffusion             |
! vref               | V initial Schaer/Vertical Diffusion             |
! wref               | W initial Schaer/Vertical Diffusion             |
! fcu                | U Factor for Rayleigh Friction Schaer           |
! fcv                | V Factor for Rayleigh Friction Schaer           |
! fcw                | W Factor for Rayleigh Friction Schaer           |
! zdref              |ZD Vertical Diffusion                            |
! qvref              |HU Vertical Diffusion                            |
! qcref              |QC Vertical Diffusion                            |
! qrref              |RW Vertical Diffusion                            |
! thref              |TH Vertical Diffusion                            |
!----------------------------------------------------------------------|
! q1ref              |Q1 tracer REFERENCE                              |
! q2ref              |Q2 tracer REFERENCE                              |
! q3ref              |Q3 tracer REFERENCE                              |
! q4ref              |Q4 tracer REFERENCE                              |
! q1err              |Q1 tracer ERROR                                  |
! q2err              |Q2 tracer ERROR                                  |
! q3err              |Q3 tracer ERROR                                  |
! q4err              |Q4 tracer ERROR                                  |
!----------------------------------------------------------------------| 
! clyref             |CLY REFERENCE                                    |
! clyerr             |CLY ERROR                                        |
!----------------------------------------------------------------------| 
!
      real, pointer, dimension (:,:,:) :: pth   => null()
      real, pointer, dimension (:,:,:) :: thbase=> null()
      real, pointer, dimension (:,:,:) :: dtv   => null()

      real, pointer, dimension (:,:,:) :: cly  => null()
      real, pointer, dimension (:,:)   :: acl  => null()
      real, pointer, dimension (:,:)   :: acl2 => null()
      real, pointer, dimension (:,:)   :: acly => null()

      real, pointer, dimension (:,:)   :: irt  => null()
      real, pointer, dimension (:,:)   :: art  => null()
      real, pointer, dimension (:,:)   :: wrt  => null()

      real, pointer, dimension (:,:,:) :: uref => null()
      real, pointer, dimension (:,:,:) :: vref => null()
      real, pointer, dimension (:,:,:) :: wref => null()
      real, pointer, dimension (:,:,:) ::zdref => null()
      real, pointer, dimension (:,:,:) ::qvref => null()
      real, pointer, dimension (:,:,:) ::qcref => null()
      real, pointer, dimension (:,:,:) ::qrref => null()
      real, pointer, dimension (:,:,:) ::thref => null()
      real, pointer, dimension (:,:,:) :: fcu  => null()
      real, pointer, dimension (:,:,:) :: fcv  => null()
      real, pointer, dimension (:,:,:) :: fcw  => null()

      real, pointer, dimension (:,:,:) :: q1ref => null()
      real, pointer, dimension (:,:,:) :: q2ref => null()
      real, pointer, dimension (:,:,:) :: q3ref => null()
      real, pointer, dimension (:,:,:) :: q4ref => null()

      real, pointer, dimension (:,:,:) :: q1err => null()
      real, pointer, dimension (:,:,:) :: q2err => null()
      real, pointer, dimension (:,:,:) :: q3err => null()
      real, pointer, dimension (:,:,:) :: q4err => null()

      real, pointer, dimension (:,:,:) ::clyref => null()
      real, pointer, dimension (:,:,:) ::clyerr => null()

      integer, parameter :: MAXNAMELENGTH = 32

      character(len=MAXNAMELENGTH) :: gmmk_pth_s, gmmk_thbase_s, gmmk_dtv_s, &
                                gmmk_cly_s, gmmk_acl_s, gmmk_acl2_s, gmmk_acly_s, &
                                gmmk_irt_s, gmmk_art_s, gmmk_wrt_s, &
                                gmmk_uref_s, gmmk_vref_s, gmmk_wref_s, gmmk_zdref_s, &
                                gmmk_qvref_s, gmmk_qcref_s, gmmk_qrref_s, gmmk_thref_s, &
                                gmmk_fcu_s, gmmk_fcv_s, gmmk_fcw_s, &
                                gmmk_q1ref_s, gmmk_q2ref_s, gmmk_q3ref_s, gmmk_q4ref_s, &
                                gmmk_clyref_s, gmmk_q1err_s, gmmk_q2err_s, &
                                gmmk_q3err_s, gmmk_q4err_s, gmmk_clyerr_s

      logical Canonical_dcmip_L, Canonical_williamson_L

end module canonical
