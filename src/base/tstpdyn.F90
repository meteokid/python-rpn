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

!**s/r tstpdyn -  Performs a dynamical timestep of the model

      subroutine tstpdyn ( F_fnitraj )
      use gmm_vt1
      use gmm_vt0
      use gmm_nest
      use gmm_rhsc
      use gmm_orh
      use gmm_geof
      use gem_options
      use grid_options
      use tdpack
      implicit none
#include <arch_specific.hf>

      integer, intent(IN) ::  F_fnitraj ! number of iterations to 
!                                         compute upstream positions

#include "gmm.hf"
#include "glb_ld.cdk"
#include "ldnh.cdk"
#include "lun.cdk"
#include "nl.cdk"

!TODO : remove the following when removing GU
#include "geomg.cdk"

      integer i0, in, j0, jn, k0, ni, nj, iln, gmmstat, j, icln
      real*8, dimension (:,:,:), allocatable :: rhs_sol, lhs_sol
      real, pointer, dimension(:,:,:)  :: hut1, hut0
!
!     ---------------------------------------------------------------
!
      gmmstat = gmm_get (gmmk_ut0_s, ut0)
      gmmstat = gmm_get (gmmk_vt0_s, vt0)
      gmmstat = gmm_get (gmmk_tt0_s, tt0)
      gmmstat = gmm_get (gmmk_st0_s, st0)
      gmmstat = gmm_get (gmmk_wt0_s, wt0)
      gmmstat = gmm_get (gmmk_qt0_s, qt0)
      gmmstat = gmm_get (gmmk_zdt0_s, zdt0)
      gmmstat = gmm_get (gmmk_fis0_s, fis0)

      gmmstat = gmm_get (gmmk_sls_s ,sls )

      gmmstat = gmm_get (gmmk_ut1_s, ut1)
      gmmstat = gmm_get (gmmk_vt1_s, vt1)
      gmmstat = gmm_get (gmmk_tt1_s, tt1)
      gmmstat = gmm_get (gmmk_st1_s, st1)
      gmmstat = gmm_get (gmmk_wt1_s, wt1)
      gmmstat = gmm_get (gmmk_qt1_s, qt1)
      gmmstat = gmm_get (gmmk_zdt1_s, zdt1)

      gmmstat = gmm_get (gmmk_orhsu_s, orhsu)
      gmmstat = gmm_get (gmmk_orhsv_s, orhsv)
      gmmstat = gmm_get (gmmk_orhst_s, orhst)
      gmmstat = gmm_get (gmmk_orhsc_s, orhsc)
      gmmstat = gmm_get (gmmk_orhsf_s, orhsf)
      gmmstat = gmm_get (gmmk_orhsw_s, orhsw)

      gmmstat = gmm_get (gmmk_rhsu_s, rhsu)
      gmmstat = gmm_get (gmmk_rhsv_s, rhsv)
      gmmstat = gmm_get (gmmk_rhst_s, rhst)
      gmmstat = gmm_get (gmmk_rhsc_s, rhsc)
      gmmstat = gmm_get (gmmk_rhsf_s, rhsf)
      gmmstat = gmm_get (gmmk_rhsw_s, rhsw)
      gmmstat = gmm_get (gmmk_rhsb_s, rhsb)

      gmmstat = gmm_get('TR/HU:M' ,hut1)
      gmmstat = gmm_get('TR/HU:P' ,hut0)

      if (.not. Grd_yinyang_L) then
         gmmstat = gmm_get (gmmk_nest_t_s, nest_t )
         gmmstat = gmm_get (gmmk_nest_q_s, nest_q )
         gmmstat = gmm_get (gmmk_nest_fullme_s,nest_fullme)
      else
         nest_t => ut1
         nest_q => ut1
      endif

      i0= 1   +pil_w
      in= l_ni-pil_e
      j0= 1   +pil_s
      jn= l_nj-pil_n
      k0= 1+Lam_gbpil_T

      if (Lun_debug_L) write(Lun_out,1000)

      if ( Orh_icn .eq. 1 ) then       ! Compute RHS

         call timing_start2 ( 20, 'RHS', 10 )

!        Compute the right-hand sides of the governing equations
         call rhs ( orhsu, orhsv, orhsc, orhst, orhsw, orhsf,&
                    ut1, vt1, wt1               ,&
                    tt1, st1, zdt1, qt1, hut1,sls, fis0     ,&
                    l_minx,l_maxx,l_miny,l_maxy, l_nk )

         call timing_stop (20)

         call frstgss ()

! Perform time interpolation of Lateral BCs for LAM configurations

         if ( .not. Grd_yinyang_L ) call nest_bcs ()

      endif

!     Perform Semi-Lagrangian advection
      
      call timing_start2 (21, 'ADV_MAIN', 10)
      call  adv_main ( F_fnitraj, Orh_icn               ,&
                       ut0, vt0 , zdt0, ut1, vt1 , zdt1 ,&
                       orhsu, rhsu, orhsv, rhsv, orhsc  ,&
                       rhsc, orhst,  rhst, orhsf, rhsf  ,&
                       orhsw, rhsw                      ,&
                        l_minx,l_maxx,l_miny,l_maxy,l_nk )        
      call timing_stop(21)

      call timing_start2 (22, 'PRE', 10)

      if ( Orh_icn.eq.1 ) then
         if ( .not. Grd_yinyang_L .and. .not. Lam_ctebcs_L) then
            fis0(1:l_ni,1:l_nj)= nest_fullme(1:l_ni,1:l_nj)
            call rpn_comm_xch_halo (fis0,l_minx,l_maxx,l_miny,l_maxy,&
               l_ni,l_nj,1,G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
         else
            if (Vtopo_L .and. (Lctl_step .ge. Vtopo_start)) then
               gmmstat = gmm_get(gmmk_fis0_s,fis0)
               call var_topo2 (fis0, real(Lctl_step),&
                               l_minx,l_maxx,l_miny,l_maxy)
               if (Grd_yinyang_L) &
                  call yyg_xchng (fis0, l_minx,l_maxx,l_miny,l_maxy, &
                                  1, .false., 'CUBIC')
               call rpn_comm_xch_halo (fis0,l_minx,l_maxx,l_miny,l_maxy,&
                  l_ni,l_nj,1,G_halox,G_haloy,G_periodx,G_periody,l_ni,0)
            endif
         endif
      endif

!     Combine some rhs to obtain the linear part
!     of the right-hand side of the elliptic problem
      call pre (rhsu, rhsv,  fis0,  rhsc, rhst, &
                rhsw, rhsf, orhsu, orhsv, rhsb, &
                nest_t, l_minx,l_maxx,l_miny,l_maxy,&
                i0, j0, in, jn, k0, l_ni, l_nj, l_nk)

      call timing_stop (22)

      if ( Lun_debug_L ) write (Lun_out,1005) Schm_itnlh

      ni = l_maxx-l_minx+1
      nj = l_maxy-l_miny+1
      allocate (nl_u(ni,nj,l_nk),nl_v(ni,nj,l_nk),nl_t(ni,nj,l_nk), &
                nl_c(ni,nj,l_nk),nl_f(ni,nj,l_nk),nl_w(ni,nj,l_nk), &
                nl_b(ni,nj))

      ni = ldnh_maxx-ldnh_minx+1
      nj = ldnh_maxy-ldnh_miny+1
      allocate ( rhs_sol(ni,nj,l_nk), lhs_sol(ni,nj,l_nk) )

      do iln=1,Schm_itnlh

         call timing_start2 ( 23, 'NLI', 10 )

!        Compute non-linear components and combine them
!        to obtain final right-hand side of the elliptic problem
         icln=Orh_icn*iln
         if ( .not. Grd_yinyang_L ) icln=icln+1
         call nli (nl_u, nl_v, nl_t, nl_c, nl_w, nl_f          ,&
                   ut0, vt0, tt0, st0, zdt0, qt0, rhs_sol, rhsc,&
                   sls, fis0, nl_b, hut0                       ,&
                   l_minx,l_maxx,l_miny,l_maxy                 ,&
                   l_nk, ni, nj, i0, j0, in, jn, k0, icln)

         call timing_stop (23)

         call timing_start2 ( 24, 'SOL', 10 )

!        Solve the elliptic problem
         call sol_main (rhs_sol,lhs_sol,ni,nj, l_nk, iln)

         call timing_stop (24)

         call timing_start2 ( 25, 'BAC', 10 )

!        Back subtitution
         call  bac (lhs_sol, sls, fis0                        ,&
                    ut0, vt0, wt0, tt0, st0, zdt0, qt0, nest_q,&
                    rhsu, rhsv, rhst, rhsw, rhsf, rhsb        ,&
                    nl_u, nl_v, nl_t, nl_w, nl_f, nl_b        ,&
                    l_minx, l_maxx, l_miny, l_maxy            ,&
                    ni,nj,l_nk,i0, j0, k0, in, jn)

         call timing_stop (25)

        if (Grd_yinyang_L) then
            call yyg_nestuv(ut0,vt0, l_minx,l_maxx,l_miny,l_maxy, G_nk)
            call yyg_xchng (tt0 , l_minx,l_maxx,l_miny,l_maxy, G_nk,&
                            .false., 'CUBIC')
            call yyg_xchng (zdt0, l_minx,l_maxx,l_miny,l_maxy, G_nk,&
                            .false., 'CUBIC')
            call yyg_xchng (st0 , l_minx,l_maxx,l_miny,l_maxy, 1   ,&
                            .false., 'CUBIC')
            if (.not.Schm_hydro_L) &
            call yyg_xchng (qt0 , l_minx,l_maxx,l_miny,l_maxy, G_nk+1,&
                            .false., 'CUBIC')
         endif

      end do

      if (Grd_yinyang_L) &
         call yyg_xchng (wt0, l_minx,l_maxx,l_miny,l_maxy, G_nk,&
                         .false., 'CUBIC')

      deallocate ( nl_u,nl_v,nl_t,nl_c,nl_f,nl_b,nl_w,&
                   rhs_sol,lhs_sol )
!
!     ---------------------------------------------------------------
!
 1000 format( &
       3X,'PERFORM A DYNAMICAL STEP: (S/R TSTPDYN)', &
      /3X,'========================================',/)
 1005 format( &
       3X,'ITERATING SCHM_ITNLH=',I3,' TIMES TO SOLVE NON-LINEAR ', &
          'HELMHOLTZ PROBLEM')

      return
      end
