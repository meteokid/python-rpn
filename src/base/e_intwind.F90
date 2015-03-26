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

!**s/r e_intwind - Wind components horizontal interpolation

      subroutine e_intwind (errcode)
      implicit none
#include <arch_specific.hf>

      integer errcode

!author       M ROCH     - july 95 - from intvent
!
!revision
! v2_30 - Sandrine Edouard  - adapt for vertical hybrid coordinate
! v2_30 - L. Corbeil        - replaced ecriseq by BMF stuff, 
! v2_30                       removed vertical interpolation
! v2_31 - M. Desgagne       - removed toppu,toppv from calling 
! v2_31                       sequence and corrected date/time recording
! v3_00 - Desgagne & Lee    - Lam configuration
! v3_01 - Lee V.            - new ip1 encoding (kind=5 -- unnormalized)
! v3_30 - Lee/Desgagne      - new LAM IO , read from analysis files to
!                             produce BCS or 3DF files
! v4_30 - Tanguay & Plante  - Derive image winds from true winds when no interpolation
! v4_60 - Gravel S.         - Remove ip2, ip3 test for e_rdhint3
! v4.70 - Gaudreault S.     - Removing wind images

#include "e_fu.cdk"
#include "e_anal.cdk"
#include "e_grids.cdk"
#include "e_cdate.cdk"
#include "dcst.cdk"
#include "bmf.cdk"
#include "pilot1.cdk"
#include "e_grdc.cdk"
#include "hgc.cdk"

      integer, external :: fstinf,fstlir,fstprm,ezqkdef,ezsetopt,&
                           ezdefset,ezuvint,e_rdhint3
      logical flag_uvt1
      integer i, j, k, src_gid, key1, key2, keyu, keyv, nic, njc, ni1, nj1,  &
              nk1,nkc,err,iu,ju,iv,jv,nu,nv, istat
      integer nisu,nisv,njsu,njsv
      integer dte, det, ipas, p1, p2, p3, g1, g2, g3, g4, bit, &
              dty, swa, lng, dlf, ubc, ex1, ex2, ex3, ip2, ip3
      character*1  typ,grd
      character*4  var,var_vv
      character*12 lab

      real, dimension (:)    , allocatable :: uu,vv,uv,w1,w2
      real, dimension (:,:,:), allocatable :: uun,vvn
!
! ---------------------------------------------------------------------
!
      if (.not.Pil_bmf_L) then
         nisu = E_Grdc_ni-1
         njsu = E_Grdc_nj
         nisv = E_Grdc_ni
         njsv = E_Grdc_nj-1
         allocate (uun(nisu,njsu,lv),vvn(nisv,njsv,lv))
      else
         nisu = niu
         njsu = nju
         nisv = niv
         njsv = njv
      endif

      allocate ( uu(max(nifi,nisu,nisv)*max(njfi,njsu,njsv)), &
	         vv(max(nifi,nisu,nisv)*max(njfi,njsu,njsv)) )
      allocate ( uv( max(nifi,nisu,nisv)*max(njfi,njsu,njsv) ))

      nu = nisu*njsu
      nv = nisv*njsv

      ip2 = ip2a
      ip3 = ip3a

      err      = 0
      errcode  = -1

      if (trim(VAR_UU).eq.'UT1') then
         var_vv   = 'VT1 '
         flag_uvt1= .true.
      else
         var_vv   = 'VV  '
         flag_uvt1= .false.
      endif

      keyu= fstinf (e_fu_anal,nic,njc,nkc,datev,' ',na(1),ip2,ip3,&
                                                        ' ',var_uu)
      keyv= fstinf (e_fu_anal,nic,njc,nkc,datev,' ',na(1),ip2,ip3,&
                                                        ' ',var_vv)

      if ((keyu.lt.0).or.(keyv.lt.0)) goto 999

      if (flag_uvt1) then

         err= 0
         do k=1,lv
            keyu = e_rdhint3 (uu,dstu_gid,nisu,njsu,var_uu,na(k),-1, &
                     -1,' ',' ',.false.,.false.,'NEAREST',e_fu_anal,6)
            keyv = e_rdhint3 (vv,dstv_gid,nisv,njsv,var_vv,na(k),-1, &
                     -1,' ',' ',.false.,.false.,'NEAREST',e_fu_anal,6)
            if ((keyu.ne.0).or.(keyv.ne.0)) goto 999
            if (Pil_bmf_L) then
               call e_bmfsplitxy2 (uu,nisu,njsu,'UU  ',k,lv,pniu,0,0,0)
               call e_bmfsplitxy2 (vv,nisv,njsv,'VV  ',k,lv,pni ,0,0,0)
            else
               call e_fill_3df (uu,uun,nisu,njsu,lv,k,1.0,0.0)
               call e_fill_3df (vv,vvn,nisv,njsv,lv,k,1.0,0.0)
            endif
         end do

         if (.not.Pil_bmf_L) then
            call e_write_3df ( uun,nisu,njsu,lv,'UU  ',unf_casc)
            call e_write_3df ( vvn,nisv,njsv,lv,'VV  ',unf_casc)
         endif         
         
      else

         anal_hav(1) = 0

         err= fstprm (keyu, DTE, DET, IPAS, ni1, nj1, nk1, BIT, DTY, P1, &
                      P2, P3, TYP, VAR, LAB, GRD, G1, G2, G3, G4, SWA  , &
                      LNG, DLF, UBC, EX1, EX2, EX3)

         src_gid = ezqkdef (nic, njc, GRD, g1, g2, g3, g4, e_fu_anal)
         err     = ezsetopt ('INTERP_DEGREE', 'CUBIC')

         allocate (w1(nic*njc),w2(nic*njc))

         err= 0
         do k=1,lv

            ip2 = ip2a
            ip3 = ip3a
            keyu = fstlir (w1, e_fu_anal, iu, ju, nkc, datev, ' ',  &
                           na(k), ip2, ip3, ' ', 'UU')
            keyv = fstlir (w2, e_fu_anal, iv, jv, nkc, datev, ' ',  &
                           na(k), ip2, ip3, ' ', 'VV')
            if (keyu.lt.0 .or. iu.ne.nic  .or. ju.ne.njc ) err= -1
            if (keyv.lt.0 .or. iv.ne.nic  .or. jv.ne.njc ) err= -1

            if (err.lt.0) goto 999

            err = ezdefset ( dstu_gid, src_gid )
            err = ezuvint  ( uu,uv,w1,w2 )
            err = ezdefset ( dstv_gid, src_gid )
            err = ezuvint  ( uv,vv,w1,w2 )

            do i=1,nisu*njsu
               uu(i) = uu(i) * dcst_knams_8
            end do
            do i=1,nisv*njsv
               vv(i) = vv(i) * dcst_knams_8
            end do
            
            if (Pil_bmf_L) then
               call e_bmfsplitxy2 (uu,nisu,njsu,'UU  ',k,lv,pniu,0,0,0)
               call e_bmfsplitxy2 (vv,nisv,njsv,'VV  ',k,lv,pni ,0,0,0)
            else
               call e_fill_3df ( uu,uun,nisu,njsu,lv,k,1.0,0.0)
               call e_fill_3df ( vv,vvn,nisv,njsv,lv,k,1.0,0.0)
            endif

         end do
         
         if (.not.Pil_bmf_L) then
            call e_write_3df ( uun,nisu,njsu,lv,'UU  ',unf_casc)
            call e_write_3df ( vvn,nisv,njsv,lv,'VV  ',unf_casc)
         endif

         deallocate (w1,w2)

      endif
      
      keyu= fstinf (e_fu_anal,nic,njc,nkc,datev,' ',na(1),ip2,ip3,&
                                                        ' ','PWUU')
      keyv= fstinf (e_fu_anal,nic,njc,nkc,datev,' ',na(1),ip2,ip3,&
                                                        ' ','PWVV')
      if ((keyu.ge.0).and.(keyv.ge.0)) then
         err= 0
         do k=1,lv-1
            keyu = e_rdhint3 (uu,dstf_gid,nifi,njfi,'PWUU',na(k),-1, &
                     -1,' ',' ',.false.,.false.,'NEAREST',e_fu_anal,6)
            keyv = e_rdhint3 (vv,dstf_gid,nifi,njfi,'PWVV',na(k),-1, &
                     -1,' ',' ',.false.,.false.,'NEAREST',e_fu_anal,6)
            if ((keyu.ne.0).or.(keyv.ne.0)) goto 999
            if (Pil_bmf_L) then
               call e_bmfsplitxy2 (uu,nifi,njfi,'PWUU',k,lv-1,pni,0,0,0)
               call e_bmfsplitxy2 (vv,nifi,njfi,'PWVV',k,lv-1,pni,0,0,0)
            else
               call e_fill_3df (uu,uun,nifi,njfi,lv-1,k,1.0,0.0)
               call e_fill_3df (vv,vvn,nifi,njfi,lv-1,k,1.0,0.0)
            endif
         end do

         if (.not.Pil_bmf_L) then
            call e_write_3df ( uun,nifi,njfi,lv-1,'PWUU',unf_casc)
            call e_write_3df ( vvn,nifi,njfi,lv-1,'PWVV',unf_casc)
         endif         
      endif

      deallocate (uu,vv,uv)

      if (Pil_bmf_L) then
         call bmf_splitwrall ('AHAV',2,1,1,Bmf_time1,Bmf_time2, &
                                             0,0,40,0,anal_hav)
         call bmf_splitend
      else
         deallocate (uun,vvn)
      endif

      errcode = 0 
!
! ---------------------------------------------------------------------
 999  return
      end
