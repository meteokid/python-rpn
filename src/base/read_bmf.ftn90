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

!**s/r read_bmf - read the dynamics fields from entrance programs
!
      subroutine read_bmf ( za_8,zb_8,lvm, zat_8,zbt_8,lvt      , &
                            u_temp,lniu,lnju, v_temp,lniv,lnjv  , &
                            hu_temp,tt_temp,gz_temp,ps,topo_temp, &
                            zd_temp,w_temp,q_temp,lni,lnj,nktmp)
      implicit none
#include <arch_specific.hf>

      integer lni,lnj,lniu,lnju,lniv,lnjv,lvm,lvt,nktmp
      real*8  za_8(lvm),zb_8(lvm),zat_8(lvt),zbt_8(lvt)
      real,   dimension(lni,lnj      ) :: ps,topo_temp
      real,   dimension(lni,lnj,nktmp) :: hu_temp,gz_temp,tt_temp, &
                                          zd_temp,w_temp,q_temp
      real    u_temp(lniu,lnju,nktmp), v_temp(lniv,lnjv,nktmp)
!
!author 
!     Luc Corbeil, mai 2002
!
!revision
! v3_01 - Corbeil L.           - initial version
! v3_10 - Lee V.               - unique bmfscraps...
! v3_11 - Gravel S.            - provide for variable topography
! v3_12 - Dugas B. & Winger K. - read TD in pressure-mode rather than HU
! v3_21 - Dugas B.             - replace TD by ES in pressure mode
! v3_30 - Tanguay M.           - Modify Check topo when no interpolation 
! v3_30 - McTaggart-Cowan R.   - update implementation of variable orography
! v4_03 - Lee V.               - Adapt to using new pressure functions
! v4_04 - Plante A.            - Remove offline mode
! v4_05 - Plante A.            - Add read for w, zd, q
!

#include "bmf.cdk"
#include "dcst.cdk"
#include "lun.cdk"
#include "acq.cdk"
#include "bcsgrds.cdk"

      integer  bmf_get
      external bmf_get

      integer i,err,nerr
      integer, parameter :: maxerr = 400
      integer, dimension(maxerr) ::  error
!
!     ---------------------------------------------------------------
!
!     Read the BMF file associated to each Ptopo_myproc

      error   = -1
      ana_zd_L= .false. ; ana_w_L = .false. ; ana_q_L = .false.; ana_vt_L = .false.

!     Initialization of some switches and dimensions

      err = bmf_get ('AHAV',bmf_time1,bmf_time2,Acqi_datasp,-1,-1., &
                                                     1,2,1,1,1,1)
      Acql_prsanl   = (Acqi_datasp(2).eq.0) ! pressure
      Acql_siganl   = (Acqi_datasp(2).eq.1) ! SIGMA
      Acql_etaanl   = (Acqi_datasp(2).eq.3) ! SIGPT   (eta, rcoef=1.0)
      Acql_hybanl   = (Acqi_datasp(2).eq.4) ! HYBLG   (hyb Laprise/Girard)
      Acql_staganl  = (Acqi_datasp(2).eq.6) ! HYBSTAG (stag hybrid Girard)
      if (Acql_staganl)      Acql_hybanl  = .true.

      error(1) = bmf_get ('ZA  ',bmf_time1,bmf_time2,-1,-1.,za_8 ,1, &
                                                        lvm,1,1,1,1)
      error(2) = bmf_get ('ZB  ',bmf_time1,bmf_time2,-1,-1.,zb_8 ,1, &
                                                        lvm,1,1,1,1)
      error(3) = bmf_get ('ZAT ',bmf_time1,bmf_time2,-1,-1.,zat_8,1, &
                                                        lvt,1,1,1,1)
      error(4) = bmf_get ('ZBT ',bmf_time1,bmf_time2,-1,-1.,zbt_8,1, &
                                                        lvt,1,1,1,1)
      error(5) = bmf_get ('ME  ',bmf_time1,bmf_time2,-1,topo_temp,-1., &
                                                      1,lni,1,lnj,1,1)
      error(6) = bmf_get ('UU  ',bmf_time1,bmf_time2,-1,u_temp,-1, &
                                           1,lniu,1,lnju,1,nktmp)
      error(7) = bmf_get ('VV  ',bmf_time1,bmf_time2,-1,v_temp,-1, &
                                           1,lniv,1,lnjv,1,nktmp)

      nerr = 8
      if (Acql_prsanl) then
         error(nerr) = bmf_get('GZ  ',bmf_time1,bmf_time2,-1, &
                             gz_temp,-1.,1,lni,1,lnj,1,nktmp)
      else
         error(nerr) = bmf_get('GZ  ',bmf_time1,bmf_time2,-1, &
                        gz_temp(1,1,lvt),-1.,1,lni,1,lnj,1,1)
      endif

      if (Acql_prsanl) then
         nerr = nerr + 1
         error(nerr) = bmf_get('HU  ',bmf_time1,bmf_time2,-1,hu_temp,-1., &
                                                     1,lni,1,lnj,1,nktmp)
         nerr = nerr + 1
         error(nerr) = bmf_get('TT  ',bmf_time1,bmf_time2,-1,tt_temp,-1., &
                                                     1,lni,1,lnj,1,nktmp)
         if (error(nerr).ne.0) then
            error(nerr) = bmf_get('VT  ',bmf_time1,bmf_time2,-1,tt_temp,-1., &
                                                     1,lni,1,lnj,1,nktmp)
            ana_vt_L=.true.
         endif
      else
         if (Acql_siganl) Acqr_ptopa = 0.
         nerr = nerr + 1
         error(nerr) = bmf_get('P0  ',bmf_time1,bmf_time2,-1,ps, &
                                            -1.,1,lni,1,lnj,1,1)
         nerr = nerr + 1
         error(nerr) = bmf_get('TT  ',bmf_time1,bmf_time2,-1,tt_temp,-1., &
                                                     1,lni,1,lnj,1,nktmp)
         if (error(nerr).ne.0) then
         error(nerr) = bmf_get('VT  ',bmf_time1,bmf_time2,-1,tt_temp, &
                                             -1.,1,lni,1,lnj,1,nktmp)
         ana_vt_L=.true.
         endif
         nerr = nerr + 1
         error(nerr) = bmf_get('HU  ',bmf_time1,bmf_time2,-1,hu_temp,-1., &
                                                     1,lni,1,lnj,1,nktmp)
         if (bmf_get('ZD  ',bmf_time1,bmf_time2,-1,zd_temp,-1., &
                                           1,lni,1,lnj,1,nktmp) .eq. 0) then
            ana_zd_L= .true.
         endif

         if (bmf_get('W   ',bmf_time1,bmf_time2,-1, w_temp,-1., &
                                           1,lni,1,lnj,1,nktmp) .eq. 0) then
            ana_w_L= .true.
         endif
         if (bmf_get('Q   ',bmf_time1,bmf_time2,-1, q_temp,-1., &
                                           1,lni,1,lnj,1,nktmp) .eq. 0) then
            ana_q_L= .true.
         endif
      endif

!     Check for error in BMF_GETS above...
      err = 0
      do i=1,nerr
         if (error(i).ne.0) then
            err = -1
            if (Lun_out.gt.0) write (Lun_out,1001) i
         endif
      end do

      call handle_error(err,'read_bmf','read_bmf')

      topo_temp (:,:) = dble(topo_temp(:,:)) * Dcst_grav_8

 1001 format (/'ERROR: #',i4,' in read_bmf')
!
!     ---------------------------------------------------------------
!
      return
      end
