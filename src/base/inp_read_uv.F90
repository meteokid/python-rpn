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

!**s/r inp_read_uv - Read horizontal winds UU,VV F valid at F_datev
!                    and perform vectorial horizontal interpolation 
!                    on proper Arakawa grid u and v respectively

      subroutine inp_read_uv ( F_u, F_v, F_ip1, F_nka )
      implicit none
#include <arch_specific.hf>

      integer                           , intent(OUT) :: F_nka
      integer, dimension(:    ), pointer, intent(OUT) :: F_ip1
      real   , dimension(:,:,:), pointer, intent(OUT) :: F_u, F_v

#include "glb_ld.cdk"
#include "dcst.cdk"
#include "geomn.cdk"
#include "hgc.cdk"
#include "inp.cdk"
#include <rmnlib_basics.hf>

      integer, external :: RPN_COMM_shuf_ezdist
      character*1 typ,grd
      character*4 var
      character*12 lab
      logical, dimension (:), allocatable :: zlist_o
      integer, parameter :: nlis = 1024
      integer i,err, nz, n1,n2,n3, nrec, liste(nlis),lislon,cnt
      integer mpx,local_nk,irest,kstart, src_gid, dst_gid, vcode, nkk
      integer dte, det, ipas, p1, p2, p3, g1, g2, g3, g4, bit, &
              dty, swa, lng, dlf, ubc, ex1, ex2, ex3
      integer, dimension(:  ), allocatable :: zlist
      real   , dimension(:,:), allocatable :: u,v,uhr,vhr,uv
      real   , dimension(:  ), pointer     :: posxu,posyu,posxv,posyv
      common /bcast_i / lislon,nz
!
!---------------------------------------------------------------------
!
      local_nk= 0 ; F_nka= -1
      if (associated(F_ip1)) deallocate (F_ip1)
      if (associated(F_u  )) deallocate (F_u  )
      if (associated(F_v  )) deallocate (F_v  )
      nullify (F_ip1, F_u, F_v)

      if (Inp_iome .ge.0) then
         vcode= -1 ; nz= -1
         nrec= fstinl (Inp_handle,n1,n2,n3,Inp_cmcdate,' ',-1,-1,-1,' ',&
                       'UU',liste,lislon,nlis)
         if (lislon == 0) goto 999

         nz= (lislon + Inp_npes - 1) / Inp_npes
         allocate ( F_ip1(lislon), uhr(G_ni*G_nj,nz), &
                    vhr(G_ni*G_nj,nz), uv(G_ni*G_nj,nz) )

         err= fstprm (liste(1), DTE, DET, IPAS, n1, n2, n3,&
                  BIT, DTY, P1, P2, P3, TYP, VAR, LAB, GRD,&
                  G1,G2,G3,G4,SWA,LNG,DLF,UBC,EX1,EX2,EX3)

         if (lislon.gt.1) then
            call sort_ip1 (liste,F_ip1,lislon)
         else
            F_ip1(1) = p1
         endif

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

         allocate (u(n1*n2,max(local_nk,1)), v(n1*n2,max(local_nk,1)))

         cnt= 0
         do i= kstart, kstart+local_nk-1
            cnt= cnt+1
            err= fstlir ( u(1,cnt), Inp_handle,n1,n2,n3,Inp_cmcdate,&
                          LAB, F_ip1(i), P2, P3,TYP, 'UU' )
            err= fstlir ( v(1,cnt), Inp_handle,n1,n2,n3,Inp_cmcdate,&
                          LAB, F_ip1(i), P2, P3,TYP, 'VV' )
         end do

         if (local_nk.gt.0) then

            posxu => Geomn_longu
            posyu => Geomn_latgs
            posxv => Geomn_longs
            posyv => Geomn_latgv

            err = ezsetopt ('INTERP_DEGREE', 'CUBIC')

            src_gid = ezqkdef (n1, n2, GRD, g1, g2, g3, g4, Inp_handle)

            write(6,'(3a)') 'Interpolating: UU valid: ',&
                             Inp_datev,' on U grid'
            dst_gid = ezgdef_fmem (G_ni, G_nj, 'Z', 'E', Hgc_ig1ro, &
                                   Hgc_ig2ro, Hgc_ig3ro, Hgc_ig4ro, &
                                   posxu, posyu)
            err = ezdefset ( dst_gid , src_gid )

            do i=1,local_nk
               err = ezuvint  ( uhr(1,i),uv(1,i), u(1,i),v(1,i))
            end do

            write(6,'(3a)') 'Interpolating: VV valid: ',&
                             Inp_datev,' on V grid'
            dst_gid = ezgdef_fmem (G_ni, G_nj, 'Z', 'E', Hgc_ig1ro, &
                                   Hgc_ig2ro, Hgc_ig3ro, Hgc_ig4ro, &
                                   posxv, posyv)
            err = ezdefset ( dst_gid , src_gid )
            do i=1,local_nk
               err = ezuvint  ( uv(1,i),vhr(1,i), u(1,i),v(1,i))
            end do
         endif
         deallocate (u,v,uv)
      else
         allocate (uhr(1,1), vhr(1,1))
      endif

 999  call rpn_comm_bcast (lislon, 2, "MPI_INTEGER", 0, "grid", err)
      F_nka= lislon

      if (F_nka .gt. 0) then

         if (F_nka .ge. 1) then
            if (Inp_iome .lt.0) allocate ( F_ip1(F_nka) )
            call rpn_comm_bcast ( F_ip1, F_nka, "MPI_INTEGER", &
                                  0, "grid", err )
         endif

         allocate (zlist(nz)) ; zlist= -1
         do i=1, local_nk
            zlist(i)= i + kstart - 1
         end do

         allocate ( F_u(l_minx:l_maxx,l_miny:l_maxy,lislon), &
                    F_v(l_minx:l_maxx,l_miny:l_maxy,lislon), &
                    zlist_o(lislon) )

         zlist_o= .FALSE.

         err = RPN_COMM_shuf_ezdist ( Inp_comm_setno, Inp_comm_id, &
                              uhr, nz, F_u, lislon, zlist, zlist_o )
         zlist_o= .FALSE.

         err = RPN_COMM_shuf_ezdist ( Inp_comm_setno, Inp_comm_id, &
                              vhr, nz, F_v, lislon, zlist, zlist_o )

         deallocate (uhr,vhr,zlist,zlist_o)

         F_u(1:l_ni,1:l_nj,:) = F_u(1:l_ni,1:l_nj,:) * Dcst_knams_8
         F_v(1:l_ni,1:l_nj,:) = F_v(1:l_ni,1:l_nj,:) * Dcst_knams_8

      else

         if (Inp_iome .ge.0) write(6,'(3a)') &
                  'Variable: UU,VV valid: ',Inp_datev, 'NOT FOUND'

      endif
!
!---------------------------------------------------------------------
!
      return
      end
