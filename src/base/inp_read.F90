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

!**s/r inp_read - Read variable F_var_S valid at Inp_datev and perform 
!                 horizontal interpolation on proper Arakawa grid

      integer function inp_read ( F_var_S, F_hgrid_S, F_dest, &
                                  F_ip1, F_nka )
      implicit none
#include <arch_specific.hf>

      character*(*)                     ,intent(IN)  :: F_var_S,F_hgrid_S
      integer                           ,intent(OUT) :: F_nka
      integer, dimension(:    ), pointer,intent(OUT) :: F_ip1
      real   , dimension(:,:,:), pointer,intent(OUT) :: F_dest

#include "glb_ld.cdk"
#include "glb_pil.cdk"
#include "dcst.cdk"
#include "geomn.cdk"
#include "hgc.cdk"
#include "inp.cdk"
#include "ptopo.cdk"
#include "tr3d.cdk"
#include "iau.cdk"
#include <rmnlib_basics.hf>

      integer, external :: RPN_COMM_shuf_ezdist, &
                           samegrid_gid, samegrid_rot, inp_is_real_wind
      character*1 typ,grd
      character*4 nomvar,var
      character*12 lab,interp_S
      character*54 vcode_S
      logical diag_lvl_L
      logical, dimension (:), allocatable :: zlist_o
      integer, parameter :: nlis = 1024
      integer i,err, nz, n1,n2,n3, nrec, liste(nlis),lislon,cnt
      ! Remove the following line by 2021
      integer ut1_is_urt1
      integer subid,nicore,njcore,datev
      integer mpx,local_nk,irest,kstart, src_gid, vcode, nkk, ip1
      integer dte, det, ipas, p1, p2, p3, g1, g2, g3, g4, bit, &
              dty, swa, lng, dlf, ubc, ex1, ex2, ex3
      integer, dimension(:  ), allocatable :: zlist
      real   , dimension(:,:), allocatable :: wk1,wk2
      real   , dimension(:  ), pointer     :: posx,posy
      real*8 add, mult
      common /bcast_i / lislon,nz
!
!---------------------------------------------------------------------
!
      F_nka= -1 ; local_nk= 0
      add= 0.d0 ; mult= 1.d0
      ! Remove the following line by 2021
      ut1_is_urt1 = -1
      if (associated(F_ip1 )) deallocate (F_ip1 )
      if (associated(F_dest)) deallocate (F_dest)
      nullify (F_ip1,F_dest)

      nomvar = F_var_S ; ip1= -1
      select case (F_var_S)
         case ('OROGRAPHY')
            if (Inp_kind == 2  ) nomvar= '@NUL'
            if (Inp_kind == 1  ) nomvar= 'GZ'
            if (Inp_kind == 5  ) nomvar= 'GZ'
            if (Inp_kind == 105) nomvar= 'FIS0'
            if ( nomvar == 'GZ' ) ip1= 93423264
            if (( nomvar == 'GZ' ) .and. (Inp_kind == 1  )) ip1= 12000
            if ( nomvar == 'GZ' ) mult= 10.d0 * Dcst_grav_8
         case ('SFCPRES')
            if (Inp_kind == 2  ) nomvar= '@NUL'
            if (Inp_kind == 1  ) nomvar= 'P0'
            if (Inp_kind == 5  ) nomvar= 'P0'
            if (Inp_kind == 105) nomvar= 'ST1'
            if ( nomvar == 'P0' ) mult= 100.d0
         case ('TEMPERATURE')
            if (Inp_kind == 2  ) nomvar= 'TT'
            if (Inp_kind == 1  ) nomvar= 'TT'
            if (Inp_kind == 5  ) nomvar= 'TT'
            if (Inp_kind == 105) nomvar= 'TT1'
            if ( nomvar == 'TT' ) add= Dcst_tcdk_8
         case ('GEOPOTENTIAL')
            if (Inp_kind == 2  ) nomvar= 'GZ'
            if (Inp_kind == 1  ) nomvar= '@NUL'
            if (Inp_kind == 5  ) nomvar= '@NUL'
            if (Inp_kind == 105) nomvar= '@NUL'
            if ( nomvar == 'GZ' ) mult= 10.d0
         case ('UU')
            mult= Dcst_knams_8
         case ('VV')
            mult= Dcst_knams_8
      end select

      datev= Inp_cmcdate
      if ( F_var_S(1:3) == 'TR/' ) then
         nomvar= F_var_S(4:)
         if (Tr3d_anydate_L) datev= -1
      endif

      if ( nomvar == '@NUL' ) return

      if (Inp_iome .ge. 0) then
         vcode= -1 ; nz= -1
         nrec= fstinl (Inp_handle, n1,n2,n3, datev,' ', &
                       ip1,-1,-1,' ', nomvar,liste,lislon,nlis)
         if (lislon == 0) goto 999

         err= fstprm (liste(1), DTE, DET, IPAS, n1, n2, n3,&
                  BIT, DTY, P1, P2, P3, TYP, VAR, LAB, GRD,&
                  G1,G2,G3,G4,SWA,LNG,DLF,UBC,EX1,EX2,EX3)

         src_gid= ezqkdef (n1, n2, GRD, g1, g2, g3, g4, Inp_handle)

         if ((trim(nomvar) == 'URT1').or.(trim(nomvar) == 'VRT1').or.&
             (trim(nomvar) == 'UT1' ).or.(trim(nomvar) == 'VT1' )) then
             err= samegrid_rot ( src_gid, &
                        Hgc_ig1ro,Hgc_ig2ro,Hgc_ig3ro,Hgc_ig4ro)
             if (err < 0) then
                lislon= 0
                goto 999
             endif
         endif

         allocate (F_ip1(lislon))
         if (lislon.gt.1) then
            call sort_ip1 (liste,F_ip1,lislon)
         else
            F_ip1(1) = p1
         endif

         nz= (lislon + Inp_npes - 1) / Inp_npes

         allocate (wk2(G_ni*G_nj,nz+1)) ! Valin???

         mpx      = mod( Inp_iome, Inp_npes )
         local_nk = lislon / Inp_npes
         irest  = lislon  - local_nk * Inp_npes
         kstart = mpx * local_nk + 1
         if ( mpx .lt. irest ) then
            local_nk   = local_nk + 1
            kstart = kstart + mpx
         else
            kstart = kstart + irest
         endif

         allocate (wk1(n1*n2,max(local_nk,1)))

         cnt= 0
         do i= kstart, kstart+local_nk-1
            cnt= cnt+1
            err= fstlir ( wk1(1,cnt), Inp_handle,n1,n2,n3,datev,&
                          LAB, F_ip1(i), P2, P3,TYP, VAR )
            ! Remove the following line by 2021
            if( ut1_is_urt1 == -1 .and. Iau_period > 0) &
                 ut1_is_urt1 = inp_is_real_wind(wk1(1,cnt),n1*n2,nomvar)
         end do

         if (local_nk.gt.0) then

            if (F_hgrid_S == 'Q') then
               posx => Geomn_longs
               posy => Geomn_latgs
            endif
            if (F_hgrid_S == 'U') then
               posx => Geomn_longu
               posy => Geomn_latgs
            endif
            if (F_hgrid_S == 'V') then
               posx => Geomn_longs
               posy => Geomn_latgv
            endif
            if (F_hgrid_S == 'F') then
               posx => Geomn_longu
               posy => Geomn_latgv
            endif

            dstf_gid = ezgdef_fmem (G_ni, G_nj, 'Z', 'E', Hgc_ig1ro, &
                                    Hgc_ig2ro, Hgc_ig3ro, Hgc_ig4ro, &
                                    posx, posy)
            interp_S= 'CUBIC'

            if ( GRD .eq. 'U' ) then
              nicore = G_ni-Glb_pil_w-Glb_pil_e
              njcore = G_nj-Glb_pil_s-Glb_pil_n
              if (n1 >= nicore .and. n2/2 >= njcore) then
                 subid= samegrid_gid ( &
                    src_gid, Hgc_ig1ro,Hgc_ig2ro,Hgc_ig3ro,Hgc_ig4ro,&
                    posx(1+Glb_pil_w), posy(1+Glb_pil_s), nicore,njcore )
              else
                 subid=-1
              endif
              if (subid >= 0) then
                 interp_S = 'NEAREST'
                 err = ezsetopt ('USE_1SUBGRID', 'YES')
                 err = ezsetival('SUBGRIDID', subid)
              endif
            endif

            err = ezdefset ( dstf_gid , src_gid )
            err = ezsetopt ('INTERP_DEGREE', interp_S)
            write(6,1001) 'Interpolating: ',trim(nomvar),', nka= ',&
               lislon,', valid: ',Inp_datev,' on ',F_hgrid_S, ' grid'
         endif
         do i=1,local_nk
            err = ezsint(wk2(1,i), wk1(1,i))
         end do
         err = ezsetopt ( 'USE_1SUBGRID', 'NO' )
         deallocate (wk1)
      else
         allocate (wk2(1,1))
      endif

 999  call rpn_comm_bcast ( lislon, 2, "MPI_INTEGER", Inp_iobcast, &
                            "grid", err )
      F_nka= lislon
      ! Remove the following line by 2021 
      call rpn_comm_allreduce ( ut1_is_urt1, Inp_ut1_is_urt1, 1, &
                                "MPI_INTEGER", "MPI_MAX", "grid", err )

      if (F_nka .gt. 0) then

         inp_read= 0
         if (F_nka .ge. 1) then
            if (Inp_iome .lt.0) allocate ( F_ip1(F_nka) )
            call rpn_comm_bcast ( F_ip1, F_nka, "MPI_INTEGER", &
                                  Inp_iobcast, "grid", err )
         endif

         allocate (zlist(nz)) ; zlist= -1
         do i=1, local_nk
            zlist(i)= i + kstart - 1
         end do

         allocate ( F_dest(l_minx:l_maxx,l_miny:l_maxy,lislon), &
                    zlist_o(lislon) )

         zlist_o= .FALSE.

         err = RPN_COMM_shuf_ezdist ( Inp_comm_setno, Inp_comm_id, &
                           wk2, nz, F_dest, lislon, zlist, zlist_o )

         deallocate (wk2,zlist,zlist_o)

         F_dest(1:l_ni,1:l_nj,:) = F_dest(1:l_ni,1:l_nj,:) * mult + add
         if (nomvar == 'ST1') &
         F_dest(1:l_ni,1:l_nj,:)= Inp_pref_a_8 * &
                                  exp(F_dest(1:l_ni,1:l_nj,:))

      else

         inp_read= -1
         if (Inp_iome .ge.0) write(6,'(7a)') ' FIELD: ',trim(F_var_S),&
                     ':',trim(nomvar),' valid: ',Inp_datev, 'NOT FOUND'

      endif

 1001 format (3a,i3,5a)
!
!---------------------------------------------------------------------
!
      return
      end
