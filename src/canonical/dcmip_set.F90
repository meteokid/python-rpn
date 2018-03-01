!--------------------------------- LICENCE BEGIN -------------------------------
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

!**s/r dcmip_set -  Setup for parameters DCMIP 2012/2016

      subroutine dcmip_set (F_adv_L,F_unout)
      use dcst
      use step_options
      use dcmip_options
      use tdpack, only : rayt_8, omega_8
      use gem_options

      use glb_ld
      use lun
      implicit none

      logical F_adv_L
      integer F_unout

      !object
      !=======================================================================
      !   Setup for parameters DCMIP 2012/2016
      !=======================================================================


      !---------------------------------------------------------------

      real(8) Rotation

      !---------------------------------------------------------------

      if (Dcmip_case==0) return

      !Identify 3D Advection runs
      !--------------------------
      F_adv_L = Dcmip_case>=11.and.Dcmip_case<=13
      if (F_unout>0.and.F_adv_L) write (F_unout, 7000)

      if (Schm_phyms_L.and.Dcmip_case/=162) call handle_error(-1,'SET_DCMIP','SET_DCMIP: Turn OFF GEM physics')

      if (Schm_phyms_L.and.Dcmip_case==162.and.(Dcmip_prec_type/=-1.or.Dcmip_pbl_type/=-1)) &
         call handle_error(-1,'SET_DCMIP','SET_DCMIP: T162: GEM physics + DCMIP2016 physics not allowed')

      !-----------------------------------
      !Set Earth's radius reduction factor
      !-----------------------------------

        if (Dcmip_case== 20.and.Dcmip_X/=1.   ) call handle_error(-1,'SET_DCMIP','SET_DCMIP Dcmip_case= 20: Set Dcmip_X=1   ')
        if (Dcmip_case== 21.and.Dcmip_X/=500. ) call handle_error(-1,'SET_DCMIP','SET_DCMIP Dcmip_case= 21: Set Dcmip_X=500 ')
        if (Dcmip_case== 22.and.Dcmip_X/=500. ) call handle_error(-1,'SET_DCMIP','SET_DCMIP Dcmip_case= 22: Set Dcmip_X=500 ')
        if (Dcmip_case== 31.and.Dcmip_X/=125. ) call handle_error(-1,'SET_DCMIP','SET_DCMIP Dcmip_case= 31: Set Dcmip_X=125 ')
        if (Dcmip_case== 43.and.Dcmip_X/=1.   ) call handle_error(-1,'SET_DCMIP','SET_DCMIP Dcmip_case= 43: Set Dcmip_X=1   ')
        if (Dcmip_case==410.and.Dcmip_X/=1.   ) call handle_error(-1,'SET_DCMIP','SET_DCMIP Dcmip_case=410: Set Dcmip_X=1   ')
        if (Dcmip_case==411.and.Dcmip_X/=10.  ) call handle_error(-1,'SET_DCMIP','SET_DCMIP Dcmip_case=411: Set Dcmip_X=10  ')
        if (Dcmip_case==412.and.Dcmip_X/=100. ) call handle_error(-1,'SET_DCMIP','SET_DCMIP Dcmip_case=412: Set Dcmip_X=100 ')
        if (Dcmip_case==413.and.Dcmip_X/=1000.) call handle_error(-1,'SET_DCMIP','SET_DCMIP Dcmip_case=413: Set Dcmip_X=1000')
        if (Dcmip_case==161.and.Dcmip_X/=1.   ) call handle_error(-1,'SET_DCMIP','SET_DCMIP Dcmip_case=161: Set Dcmip_X=1   ')
        if (Dcmip_case==162.and.Dcmip_X/=1.   ) call handle_error(-1,'SET_DCMIP','SET_DCMIP Dcmip_case=162: Set Dcmip_X=1   ')
        if (Dcmip_case==163.and.Dcmip_X/=120. ) call handle_error(-1,'SET_DCMIP','SET_DCMIP Dcmip_case=163: Set Dcmip_X=120 ')

        !Reset Earth's radius
        !--------------------
        Dcst_rayt_8     = rayt_8/Dcmip_X ! rayt_8 = Reference Earth's Radius (m)
        Dcst_inv_rayt_8 = Dcmip_X/rayt_8

        !Reset time step
        !---------------
        Step_dt = Step_dt/Dcmip_X

        !Reset Vspng_coeftop
        !-------------------
        Vspng_coeftop = Vspng_coeftop/Dcmip_X

      !--------------------
      !Set Earth's rotation
      !--------------------

        Rotation = Dcmip_X

        !No Rotation
        !-----------
        if (Dcmip_case==163.or. &
            Dcmip_case== 20.or. &
            Dcmip_case== 21.or. &
            Dcmip_case== 22.or. &
            Dcmip_case== 31) Rotation = 0.

        !Reset Earth's angular velocity
        !------------------------------
        Dcst_omega_8 = Rotation * omega_8 ! omega_8 = Reference rotation rate of the Earth (s^-1)

      !------------------------
      !Toy Chemistry Terminator
      !------------------------
      if (Dcmip_case==161.and..NOT.Dcmip_Terminator_L) &
         call handle_error(-1,'SET_DCMIP','SET_DCMIP Dcmip_case= 161: Set Dcmip_Terminator_L')

      !-----------------
      !Rayleigh Friction
      !-----------------
      if ((Dcmip_case==21.or.Dcmip_case==22).and..NOT.Dcmip_Rayleigh_friction_L) &
         call handle_error(-1,'SET_DCMIP','SET_DCMIP Dcmip_case= 21/22: Set Dcmip_Rayleigh_friction_L')

      !------------------------------------------------------------------------------------
      !Vertical Diffusion: CAUTION: WE ASSUME COEFFICIENTS ARE ALREADY SET FOR SMALL PLANET
      !------------------------------------------------------------------------------------
      Dcmip_vrd_L = Dcmip_nuZ_wd/=0.or.Dcmip_nuZ_tr/=0.or.Dcmip_nuZ_th/=0

      !--------------------------
      !DCMIP 2016 Physics Package
      !--------------------------
      if (Dcmip_pbl_type/=-1.and.Dcmip_case==163) &
         call handle_error(-1,'SET_DCMIP', 'SET_DCMIP: DONT activate Planetary Boundary Layer when Supercell')

      !--------------------------------------------------
      !Moist=1/Dry=0 Initial conditions (case=161/41X/43)
      !--------------------------------------------------
      if (Dcmip_moist==0.and.Dcmip_case==43) &
         call handle_error(-1,'SET_DCMIP', 'SET_DCMIP: Set Dcmip_moist==1 when Dcmip_case= 43' )
      if (Dcmip_moist==1.and.Dcmip_case>=410.and.Dcmip_case<=413) &
         call handle_error(-1,'SET_DCMIP', 'SET_DCMIP: Set Dcmip_moist==0 when Dcmip_case= 41X')

      if (Lun_out>0) write (Lun_out,1000)

      if (Lun_out>0) write (Lun_out,1001) Dcmip_X,Dcst_rayt_8,Dcst_omega_8,Dcmip_prec_type,Dcmip_pbl_type,Dcmip_Terminator_L,Dcmip_vrd_L

      !---------------------------------------------------------------

      return

 1000 format(                                                                    /, &
      '!----------------------------------------------------------------------|',/, &
      '!DESCRIPTION of DCMIP_2016_PHYSICS                                     |',/, &
      '!----------------------------------------------------------------------|',/, &
      '!  prec_type         | Type of precipitation/microphysics              |',/, &
      '!                    | ------------------------------------------------|',/, &
      '!                    |  0: Large-scale precipitation (Kessler)         |',/, &
      '!                    |  1: Large-scale precipitation (Reed-Jablonowski)|',/, &
      '!                    | -1: NONE                                        |',/, &
      '!----------------------------------------------------------------------|',/, &
      '!  pbl_type          | Type of planetary boundary layer                |',/, &
      '!                    | ------------------------------------------------|',/, &
      '!                    |  0: Reed-Jablonowski Boundary layer             |',/, &
      '!                    |  1: Georges Bryan Planetary Boundary Layer      |',/, &
      '!                    | -1: NONE                                        |',/, &
      '!----------------------------------------------------------------------|')

 1001 format( &
      /,'SETUP FOR PARAMETERS DCMIP_2016: (S/R DCMIP_2016_SET)',   &
      /,'=====================================================',/, &
        ' X Scaling Factor for Small planet  = ',F7.2          ,/, &
        ' Revised radius   for Small planet  = ',E14.5         ,/, &
        ' Revised angular velocity           = ',E14.5         ,/, &
        ' Precipitation/microphysics type    = ',I2            ,/, &
        ' Planetary boundary layer type      = ',I2            ,/, &
        ' Toy Chemistry                      = ',L2            ,/, &
        ' Vertical Diffusion                 = ',L2            ,   &
      /,'=====================================================',/)
 7000 format (//'  ====================='/&
                '  ACADEMIC 3D Advection'/&
                '  ====================='//)

      end subroutine dcmip_set
