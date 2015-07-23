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

!**s/r tstpdyn_gu -  Dynamics timestep for GU grids
!
      subroutine tstpdyn_gu ( F_fnitraj )
      implicit none
#include <arch_specific.hf>

      integer F_fnitraj

!author
!     Alain Patoine ( after version v1_93 of tstpdyn2 )
!
!revision
! v2_00 - Desgagne M.       - initial MPI version
! v2_10 - Tanguay M.        - store TRAJ for 4D-Var
! v2_30 - Edouard S.        - introduce Ncn
! v3_00 - Desgagne & Lee    - Lam configuration
! v3_03 - Tanguay M.        - Adjoint NoHyd configuration
! v3_10 - Corbeil & Desgagne & Lee - AIXport+Opti+OpenMP
! v3_20 - Tanguay M.        - Option of storing instead of redoing TRAJ
! v3_21 - Desgagne M.       - introduce new timing routines
! v4_04 - Tanguay M.        - Staggered version TL/AD
! v4_05 - Girard C.         - Added boundary condition for top piloting
! v4_40 - Tanguay M.        - Revision TL/AD
! v4_70 - Gaudreault S.     - Simplify interface for gmm pointer
!
!arguments
!  Name        I/O                 Description
!----------------------------------------------------------------
! F_fnitraj     I         number of iterations to compute upstream
!                         positions
!----------------------------------------------------------------

#include "gmm.hf"
#include "glb_ld.cdk"
#include "grd.cdk"
#include "lam.cdk"
#include "lctl.cdk"
#include "ldnh.cdk"
#include "lun.cdk"
#include "nest.cdk"
#include "nl.cdk"
#include "orh.cdk"
#include "p_geof.cdk"
#include "rhsc.cdk"
#include "schm.cdk"
#include "vtopo.cdk"
#include "vt0.cdk"
#include "vt1.cdk"

!TODO : remove the following when removing GU
#include "dcst.cdk"
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
      gmmstat = gmm_get (gmmk_xdt0_s, xdt0)
      gmmstat = gmm_get (gmmk_qdt0_s, qdt0)
      gmmstat = gmm_get (gmmk_fis0_s, fis0)

      gmmstat = gmm_get (gmmk_ut1_s, ut1)
      gmmstat = gmm_get (gmmk_vt1_s, vt1)
      gmmstat = gmm_get (gmmk_tt1_s, tt1)
      gmmstat = gmm_get (gmmk_st1_s, st1)
      gmmstat = gmm_get (gmmk_wt1_s, wt1)
      gmmstat = gmm_get (gmmk_qt1_s, qt1)
      gmmstat = gmm_get (gmmk_zdt1_s, zdt1)
      gmmstat = gmm_get (gmmk_xdt1_s, xdt1)
      gmmstat = gmm_get (gmmk_qdt1_s, qdt1)

      gmmstat = gmm_get (gmmk_orhsu_s, orhsu)
      gmmstat = gmm_get (gmmk_orhsv_s, orhsv)
      gmmstat = gmm_get (gmmk_orhst_s, orhst)
      gmmstat = gmm_get (gmmk_orhsc_s, orhsc)
      gmmstat = gmm_get (gmmk_orhsf_s, orhsf)
      gmmstat = gmm_get (gmmk_orhsw_s, orhsw)
      gmmstat = gmm_get (gmmk_orhsx_s, orhsx)
      gmmstat = gmm_get (gmmk_orhsq_s, orhsq)

      gmmstat = gmm_get (gmmk_rhsu_s, rhsu)
      gmmstat = gmm_get (gmmk_rhsv_s, rhsv)
      gmmstat = gmm_get (gmmk_rhst_s, rhst)
      gmmstat = gmm_get (gmmk_rhsc_s, rhsc)
      gmmstat = gmm_get (gmmk_rhsf_s, rhsf)
      gmmstat = gmm_get (gmmk_rhsw_s, rhsw)
      gmmstat = gmm_get (gmmk_rhsx_s, rhsx)
      gmmstat = gmm_get (gmmk_rhsq_s, rhsq)
      gmmstat = gmm_get (gmmk_rhsb_s, rhsb)

      gmmstat = gmm_get (gmmk_ruw1_s, ruw1)
      gmmstat = gmm_get (gmmk_rvw1_s, rvw1)
      gmmstat = gmm_get (gmmk_ruw2_s, ruw2)
      gmmstat = gmm_get (gmmk_rvw2_s, rvw2)
      gmmstat = gmm_get (gmmk_xct1_s, xct1)
      gmmstat = gmm_get (gmmk_yct1_s, yct1)
      gmmstat = gmm_get (gmmk_zct1_s, zct1)

      gmmstat = gmm_get('TR/HU:M' ,hut1)
      gmmstat = gmm_get('TR/HU:P' ,hut0)

      nest_t => ut1
      nest_q => ut1

      i0= 1   +pil_w
      in= l_ni-pil_e
      j0= 1   +pil_s
      jn= l_nj-pil_n
      k0= 1+Lam_gbpil_T

      do j = 1, l_nj
         ut1(:,j,:) = ut1(:,j,:) * geomg_cy_8(j)  / Dcst_rayt_8
         vt1(:,j,:) = vt1(:,j,:) * geomg_cyv_8(j) / Dcst_rayt_8
         ut0(:,j,:) = ut0(:,j,:) * geomg_cy_8(j)  / Dcst_rayt_8
         vt0(:,j,:) = vt0(:,j,:) * geomg_cyv_8(j) / Dcst_rayt_8
      end do

      if ( Orh_icn .eq. 1 ) then       ! Compute RHS

         call timing_start2 ( 20, 'RHS', 10 )

         call rhs_gu &
            (orhsu, orhsv, orhsc, orhst, orhsw, orhsf, orhsx, orhsq, &
             ruw1,rvw1,ut1,vt1,wt1,tt1,st1,zdt1,qt1,xdt1,qdt1,hut1 , &
             fis0, l_minx,l_maxx,l_miny,l_maxy,l_nk)

         call timing_stop (20)

         call frstgss ()

      endif

      call timing_start2 (21, 'ADW', 10)

      call itf_adx_main (F_fnitraj)  ! Semi-Lagrangian advection

      call timing_stop(21)

      call timing_start2 (22, 'PRE', 10)

!     Combine some rhs to obtain the linear part
!     of the right-hand side of the elliptic problem

      call pre (rhsu, rhsv, ruw1, ruw2, rvw1, rvw2, &
                xct1, yct1, zct1, fis0, rhsc, rhst, &
                rhsw, rhsf, rhsx, rhsq, orhsu, orhsv, rhsb, &
                nest_t, l_minx,l_maxx,l_miny,l_maxy,&
                i0, j0, in, jn, k0, l_ni, l_nj, l_nk)

      call timing_stop (22)

      ni = l_maxx-l_minx+1
      nj = l_maxy-l_miny+1
      allocate (nl_u(ni,nj,l_nk),nl_v(ni,nj,l_nk),nl_t(ni,nj,l_nk), &
                nl_c(ni,nj,l_nk),nl_f(ni,nj,l_nk),nl_w(ni,nj,l_nk), &
                nl_x(ni,nj,l_nk),nl_b(ni,nj))

      ni = ldnh_maxx-ldnh_minx+1
      nj = ldnh_maxy-ldnh_miny+1
      allocate ( rhs_sol(ni,nj,l_nk), lhs_sol(ni,nj,l_nk) )

      do iln=1,Schm_itnlh

         call timing_start2 ( 23, 'NLI', 10 )

!        Compute non-linear components and combine them
!        to obtain final right-hand side of the elliptic problem
         icln=Orh_icn*iln
         call nli (nl_u, nl_v, nl_t, nl_c, nl_w, nl_f, nl_x     ,&
                   ut0, vt0, tt0, st0, zdt0, qt0, rhs_sol, rhsc ,&
                   fis0, nl_b, xdt0, qdt0, hut0, l_minx,l_maxx,l_miny,l_maxy,&
                   l_nk, ni, nj, i0, j0, in, jn, k0, icln)

         call timing_stop (23)

         call timing_start2 ( 24, 'SOL', 10 )

!        Solve the elliptic problem
         call sol_main (rhs_sol,lhs_sol,ni,nj, l_nk, iln)

         call timing_stop (24)

         call timing_start2 ( 25, 'BAC', 10 )

!        Back subtitution: final solution in ut0, vt0, etc...
         call  bac (lhs_sol, fis0                             ,&
                    ut0, vt0, wt0, tt0, st0, zdt0, qt0, nest_q,&
                    rhsu, rhsv, rhst, rhsw, rhsf, rhsx, rhsb  ,&
                    nl_u, nl_v, nl_t, nl_w, nl_f, nl_x, nl_b  ,&
                    xdt0, qdt0, rhsq                          ,&
                    l_minx, l_maxx, l_miny, l_maxy            ,&
                    ni,nj,l_nk,i0, j0, k0, in, jn)

         call timing_stop (25)

      end do

      do j = 1, l_nj
         ut1(:,j,:) = ut1(:,j,:) * geomg_invcy_8(j)  * Dcst_rayt_8
         vt1(:,j,:) = vt1(:,j,:) * geomg_invcyv_8(j) * Dcst_rayt_8
         ut0(:,j,:) = ut0(:,j,:) * geomg_invcy_8(j)  * Dcst_rayt_8
         vt0(:,j,:) = vt0(:,j,:) * geomg_invcyv_8(j) * Dcst_rayt_8
      end do

      deallocate (nl_u,nl_v,nl_t,nl_c,nl_f,nl_b,nl_w,nl_x,&
                  rhs_sol,lhs_sol)
!
!     ---------------------------------------------------------------
!
      return
      end
