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
!**s/r sol_3d - Full 3D iterative elliptic solver based on fgmres.
!               Available only for LAM configurations

   subroutine sol_3d ( F_rhs_sol, F_lhs_sol, F_ni, F_nj, F_nk, &
                       F_print_L, F_offi, F_offj )

      use matvec, only: matvec_3d
      use krylov, only: krylov_fgmres, krylov_fbicgstab

      use grid_options
      use gem_options
      use glb_ld
      use lun
      use ldnh
      use sol
      use opr
      implicit none
#include <arch_specific.hf>

      logical, intent(in) :: F_print_L
      integer, intent(in) :: F_ni,F_nj,F_nk,F_offi, F_offj
      real*8, dimension(F_ni,F_nj,F_nk), intent(in) ::  F_rhs_sol
      real*8, dimension(F_ni,F_nj,F_nk), intent(inout) ::  F_lhs_sol

!author
!     Abdessamad Qaddouri -- January 2014
!
!revision
! v4-70 - Qaddouri A.      - initial version
! v4-70 - Gaudreault S.    - new Krylov solvers


      integer i,j,k,iter,its
      real    linfini
      real*8, dimension (ldnh_maxx,ldnh_maxy,l_nk) :: rhs_yy
      real*8  conv
      real*8, dimension(:,:,:), allocatable, save ::  saved_sol
      integer :: i0, il, j0, jl
!
!     ---------------------------------------------------------------
!
      i0 = 1    + pil_w
      il = l_ni - pil_e
      j0 = 1    + pil_s
      jl = l_nj - pil_n

      if (.not. allocated(saved_sol)) then
         allocate(saved_sol(F_ni,F_nj,F_nk))
         saved_sol = 0.d0
      end if

      if (Grd_yinyang_L) then
         rhs_yy = F_rhs_sol

         do iter=1, Sol_yyg_maxits
            select case(Sol3D_krylov_S)
               case ('FGMRES')
                  its = krylov_fgmres (F_lhs_sol, matvec_3d, rhs_yy, saved_sol,&
                                       l_ni,l_nj, F_nk, ldnh_minx, ldnh_maxx  ,&
                                       ldnh_miny, ldnh_maxy, i0, il, j0, jl   ,&
                                       sol_fgm_eps, sol_im, sol_fgm_maxits    ,&
                                       Sol3D_precond_S, conv)

               if ( F_print_L ) then
                  write(Lun_out, "(3x,'Final FGMRES 3D solver convergence criteria: ',1pe14.7,' at iteration ', i3)") conv, its
               end if

               case ('FBICGSTAB')
                  its = krylov_fbicgstab (F_lhs_sol, matvec_3d, rhs_yy, saved_sol,&
                                          l_ni,l_nj, F_nk, ldnh_minx, ldnh_maxx  ,&
                                          ldnh_miny, ldnh_maxy, i0, il, j0, jl   ,&
                                          sol_fgm_eps, sol_fgm_maxits            ,&
                                          Sol3D_precond_S, conv)

                  if ( F_print_L ) then
                     write(Lun_out, "(3x,'Final BiCGSTAB 3D solver convergence criteria: ',1pe14.7,' at iteration ', i3)") conv, its
                  end if
            end select

            rhs_yy = 0.d0
            call yyg_rhs_scalbc (rhs_yy, F_lhs_sol, ldnh_minx, ldnh_maxx,&
                                 ldnh_miny, ldnh_maxy, l_nk, iter, linfini)
            do k = 1, Schm_nith
               do j = 1+pil_s, ldnh_nj-pil_n
                  do i = 1+pil_w, ldnh_ni-pil_e
                     rhs_yy(i,j,k)= F_rhs_sol(i,j,k)+rhs_yy(i,j,k)*Opr_opszp0_8(G_nk+k)/&
                                 Opr_opsxp0_8(G_ni+F_offi+i)/ Opr_opsyp0_8(G_nj+F_offj+j)
                  end do
               end do
            end do

            if (Lun_debug_L.and.F_print_L) write(Lun_out,1001) linfini,iter
            if ((iter.gt.1).and.(linfini.lt.Sol_yyg_eps)) exit

         end do

         if (F_print_L) then
            write(Lun_out,1002) linfini,iter
            if (linfini.gt.Sol_yyg_eps) write(Lun_out,9001) Sol_yyg_eps
         endif

      else

         select case(Sol3D_krylov_S)
            case ('FGMRES')
               its = krylov_fgmres (F_lhs_sol, matvec_3d, F_rhs_sol, saved_sol,&
                                    l_ni,l_nj, F_nk, ldnh_minx, ldnh_maxx     ,&
                                    ldnh_miny, ldnh_maxy, i0, il, j0, jl      ,&
                                    sol_fgm_eps, sol_im, sol_fgm_maxits       ,&
                                    Sol3D_precond_S, conv)

               if ( F_print_L ) then
                  write(Lun_out, "(3x,'Final FGMRES 3D solver convergence criteria: ',1pe14.7,' at iteration ', i3)") conv, its
               end if

            case ('FBICGSTAB')
               its = krylov_fbicgstab (F_lhs_sol, matvec_3d, F_rhs_sol, saved_sol,&
                                       l_ni,l_nj, F_nk, ldnh_minx, ldnh_maxx     ,&
                                       ldnh_miny, ldnh_maxy, i0, il, j0, jl      ,&
                                       sol_fgm_eps, sol_fgm_maxits               ,&
                                       Sol3D_precond_S, conv)

               if ( F_print_L ) then
                  write(Lun_out, "(3x,'Final BiCGSTAB 3D solver convergence criteria: ',1pe14.7,' at iteration ', i3)") conv, its
               end if
         end select
      endif

      saved_sol = F_lhs_sol

 1001 format (3x,'Iterative YYG    solver convergence criteria: ',1pe14.7,' at iteration', i3)
 1002 format (3x,'Final YYG    solver convergence criteria: ',1pe14.7,' at iteration', i3)
 9001 format (3x,'WARNING: iterative YYG solver DID NOT converge to requested criteria:: ',1pe14.7)
!
!     ---------------------------------------------------------------
!
      return
   end subroutine sol_3d


