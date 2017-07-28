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

!**s/r ens_nml - read parametres for ensemble forecast
!
      integer function ens_nml (F_namelist_S, F_Grdtyp_S, F_unout)
      use ens_options
      use wb_itf_mod
      implicit none
#include <arch_specific.hf>
!
      character* (*) F_namelist_S, F_Grdtyp_S
      integer F_unout

!author Lubos Spacek - May 2005
!
!revision
! v4_12 - Spacek L.          - called from itf_ens_init
! v_4.1.3 - N. Gagnon      - add vertical envelope parameters for PTP and change name of most parameters in the NAMELIST
! v_4.2.0 - N. Gagnon      - add the parameter Ens_ptp_env_hor_f for latitudinal enveloppe that control PTP,
!           L. Spacek      - remove limitation of use when in mode LAM (only SKEB is now unavailable)
! v_4.4.0 - N. Gagnon      - remove Ens_ptp_env_hor_f not use anymore
!                          - add ens_ptp_cape and ens_ptp_tlc keys
! v_4.5.0 - N. Gagnon      - add F_ens_ptp_crit_w and F_ens_ptp_fac_reduc keys
!
!object
!  To initialize the ensemble forecast system
!  by reading the namelist ens_nml
!
!arguments
!  Name           I/O                 Description
!-----------------------------------------------------------------------
! F_namelistf_S       I             namelist path
! F_Grdtyp_S          I             grid configuration (GU required)
! F_unout             I             output unit
!-----------------------------------------------------------------------
!
#include "ens_param.cdk"

      integer, external :: fnom,wkoffit
      logical found_namelist_ok, stochphy_L
      integer i, ier,err,err_open,unf,ios
!
!--------------------------------------------------------------------
!
      ens_nml= -1
      stochphy_L  = .false.

      if ((F_namelist_S.eq.'print').or.(F_namelist_S.eq.'PRINT')) then
         ens_nml = 0
         if (F_unout.ge.0) write (F_unout,nml=ensembles)
         return
      endif

      unf = 0
      found_namelist_ok = .false.
      err = wkoffit (F_namelist_S)
      if (err.ge.-1) then
         err_open  = fnom (unf,F_namelist_S, 'SEQ+OLD', 0)
         if (err_open.eq.0) then
            read (unf, nml=ensembles, end = 1000, err=999, iostat=ios)
            found_namelist_ok = .true.
            if (F_unout.ge.0) write (F_unout,nml=ensembles)
         endif
 1000    call fclos (unf)
      endif

      if (found_namelist_ok) then

         ens_nml = 1

         if ((F_Grdtyp_S.ne.'GU'.and.F_Grdtyp_S.ne.'GY').and.Ens_skeb_conf) then
            if(F_unout.ge.0) write(F_unout,"(a,a)" ) 'Ens_nml: ','Ens_skeb only available with grid GU/GY'
            ens_nml = -1
         endif

         if ((Ens_mc_seed.lt.0))then
            if(F_unout.ge.0)write(F_unout,*)'You have to provide a positive integer as seed see Ens_mc_seed in NAMELIST'
            ens_nml = -1
          endif

            if (Ens_skeb_nlon .ne. 2*Ens_skeb_nlat)then
              if(F_unout.ge.0)write(F_unout,*)' Nlon must equal 2*nlat'
              ens_nml = -1
            endif

         if (Ens_ptp_ncha.gt.MAX2DC) then
            if(F_unout.ge.0)write(F_unout,*)'Ens_ptp_ncha must be <=9'
            ens_nml = -1
         endif

         do i=1,Ens_ptp_ncha
            if (Ens_ptp_nlon(i).ne.2*Ens_ptp_nlat(i))then
                if(F_unout.ge.0)write(F_unout,*)'Nlon2 must equal 2*nlat2'
                ens_nml = -1
            endif
         enddo

         if (ens_nml.lt.0) return

         Ens_skeb_conf  =  Ens_skeb_conf.and.Ens_conf
         Ens_skeb_l     =  Ens_skeb_trnh-Ens_skeb_trnl+1
         Ens_skeb_m     =  Ens_skeb_trnh+1
         Ens_skeb_div   =  Ens_skeb_div .and.Ens_conf
         Ens_stat       =  Ens_stat.and.Ens_conf
         Ens_ptp_l      =  Ens_ptp_trnh-Ens_ptp_trnl+1
         Ens_ptp_lmax   =  maxval(Ens_ptp_l)
         Ens_ptp_m      =  Ens_ptp_trnh+1
         Ens_ptp_mmax   =  maxval(Ens_ptp_m)

         stochphy_L  = Ens_ptp_conf.and.Ens_conf

         if(F_unout.ge.0)then
            write(F_unout,"(a,i8)" )'Ens_mc_seed   = ',Ens_mc_seed
            write(F_unout,'(a,l5)' )'Ens_skeb_conf  = ',Ens_skeb_conf
            write(F_unout,'(a,l5)' )'Ens_stat  = ',Ens_stat
            write(F_unout,'(a,l5)' )'Ens_skeb_div   = ',Ens_skeb_div
            write(F_unout,'(a,10i5)')'Ens_ptp_l     = ',Ens_ptp_l
            write(F_unout,'(a,i5)' )'Ens_ptp_lmax = ',Ens_ptp_lmax
            write(F_unout,'(a,10i5)')'Ens_ptp_m     = ',Ens_ptp_m
            write(F_unout,'(a,i5)' )'Ens_ptp_mmax = ',Ens_ptp_mmax
            write(F_unout,'(a,10i5)')'Ens_skeb_l     = ',Ens_skeb_l
            write(F_unout,'(a,10i5)')'Ens_skeb_m     = ',Ens_skeb_m
            write(F_unout,'(a,l5)' )'Ens_stochphy_L = ',stochphy_L
            write(F_unout,'(a,i5)' )'Ens_imrkv2     = ',Ens_ptp_ncha
            write(F_unout,'(a,f8.5)' )'Ens_ens_ptp_env_u = ',Ens_ptp_env_u
            write(F_unout,'(a,f8.5)' )'Ens_ens_ptp_env_b = ',Ens_ptp_env_b
            write(F_unout,'(a,f8.5)' )'Ens_ens_ptp_cape = ',Ens_ptp_cape
            write(F_unout,'(a,f8.5)' )'Ens_ens_ptp_tlc = ',Ens_ptp_tlc
            write(F_unout,'(a,f12.5)' )'Ens_ens_ptp_crit_w = ',Ens_ptp_crit_w
            write(F_unout,'(a,f8.5)' )'Ens_ens_ptp_fac_reduc = ',Ens_ptp_fac_reduc
         endif
      else
          ens_nml = 0
      endif

      ier= WB_OK
      if (stochphy_L) then
         ier= min(wb_put('ens/IMRKV2'     , Ens_ptp_ncha     , WB_REWRITE_MANY),ier)
         ier= min(wb_put('ens/STOCHPHY'   , stochphy_L       , WB_REWRITE_MANY),ier)
         ier= min(wb_put('ens/PTPENVU'    , Ens_ptp_env_u    , WB_REWRITE_MANY),ier)
         ier= min(wb_put('ens/PTPENVB'    , Ens_ptp_env_b    , WB_REWRITE_MANY),ier)
         ier= min(wb_put('ens/PTPCAPE'    , Ens_ptp_cape     , WB_REWRITE_MANY),ier)
         ier= min(wb_put('ens/PTPTLC'     , Ens_ptp_tlc      , WB_REWRITE_MANY),ier)
         ier= min(wb_put('ens/PTPCRITW'   , Ens_ptp_crit_w   , WB_REWRITE_MANY),ier)
         ier= min(wb_put('ens/PTPFACREDUC', Ens_ptp_fac_reduc, WB_REWRITE_MANY),ier)
      endif

      err=0
      if (WB_IS_ERROR(ier)) ens_nml = -1
!
!--------------------------------------------------------------------
!
      if(ios<0)then
        ier= min(wb_put('ens/STOCHPHY'   , Ens_ptp_conf       , WB_REWRITE_MANY),ier)
        ens_nml= -3
      endif
999   continue
      return
      end

