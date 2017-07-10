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
!
module krylov_mod
   ! Common Krylov methods.
   !
   ! References
   ! C. T. Kelley, Iterative Methods for Linear and Nonlinear Equations, SIAM, 1995
   ! (https://www.siam.org/books/textbooks/fr16_book.pdf)
   !
   ! Y. Saad, Iterative Methods for Sparse Linear Systems. SIAM, 2003.
   ! (http://www-users.cs.umn.edu/~saad/IterMethBook_2ndEd.pdf)
   !
   ! Author
   !     Stéphane Gaudreault -- June 2014
   !
   ! Revision
   !     v4-70 - Gaudreault S.      - initial version
   !
   implicit none

   private

#include "prec.cdk"
#include "lun.cdk"
#include "glb_ld.cdk"

   public :: krylov_fbicgstab, krylov_fgmres

contains

      integer function krylov_fbicgstab(x, matvec, b, x0, ni, nj, nk, &
                                        minx, maxx, miny, maxy, i0, il, j0, jl, &
                                        tolerance, maxiter, precond_S, conv) result(retval)
      ! Solves a linear system using a Bi-Conjugate Gradient STABilised
      !
      integer, intent(in) :: ni, nj, nk, minx, maxx, miny, maxy, i0, il, j0, jl
      !
      ! The converged solution.
      real*8, dimension (minx:maxx, miny:maxy, nk), intent(out) :: x
      !
      ! Initial guess for the solution
      real*8, dimension (minx:maxx, miny:maxy, nk), intent(in) :: x0
      !
      ! Right hand side of the linear system.
      real*8, dimension (minx:maxx, miny:maxy, nk), intent(in) :: b
      !
      ! A matrix-vector product routine (A.*v).
      interface
         subroutine matvec(v, prod)
#include "ldnh.cdk"
#include "glb_ld.cdk"
            real*8, dimension (ldnh_minx:ldnh_maxx, ldnh_miny:ldnh_maxy, l_nk), intent(in) :: v
            real*8, dimension (ldnh_minx:ldnh_maxx, ldnh_miny:ldnh_maxy, l_nk), intent(out) :: prod
         end subroutine
      end interface
      !
      ! Tolerance to achieve. The algorithm terminates when either the relative
      ! or the absolute residual is below tolerance.
      real*8, intent(in) :: tolerance
      !
      ! Maximum number of iterations. Iteration will stop after maxiter steps
      ! even if the specified tolerance has not been achieved.
      integer, intent(in) :: maxiter
      ! Preconditioner
      character(len=*), intent(in) :: precond_S
      !
      real*8, intent(out) :: conv

      integer :: iter, i, j, k
      real*8 :: relative_tolerance, alpha, beta, tau, omega
      real*8 :: rho_old, rho, norm_residual, r0
      logical :: force_restart

      real*8, dimension(minx:maxx, miny:maxy, nk) :: residual, hatr0 ,res_matvec
      real*8, dimension(minx:maxx, miny:maxy, nk) :: vv, pp, pp_prec, ss, ss_prec, tt

      logical almost_zero
      real*8 dist_dotproduct

      ! Here we go !

      retval = 0

      x(:, :, :) = x0(:, :, :)

      residual(:, :, :) = 0.0d0
      ss(:, :, :) = 0.0d0
      tt(:, :, :) = 0.0d0
      vv(:, :, :) = 0.0d0
      pp(:, :, :) = 0.0d0

      ! Residual of the initial iterate
      call matvec(x, res_matvec)

      do k=1,nk
         do j=j0,jl
            do i=i0,il
               residual(i, j, k) = b(i, j, k) - res_matvec(i, j, k)
            end do
         end do
      end do

      force_restart = .false.
      hatr0 = residual
      rho_old = 1.d0
      alpha = 1.d0
      omega = 1.d0
      rho = dist_dotproduct(hatr0, residual, minx, maxx, miny, maxy, i0, il, j0, jl, nk)
      norm_residual = sqrt(rho)
      r0 = norm_residual

      ! Scale tolerance acording to the values of the residual
      relative_tolerance = tolerance * norm_residual

      iter = 0
      do while ((norm_residual > relative_tolerance) .and. (iter < maxiter))
         iter = iter + 1

         if (almost_zero(omega)) then
            write(lun_out, *) 'WARNING : BiCGSTAB breakdown (omega=0)'
            force_restart = .true.

            beta = 0.0d0
         else
            beta = (rho / rho_old) * (alpha / omega)
         end if

!$omp parallel
!$omp do private (i,j,k)
         do k=1,nk
            do j=j0,jl
               do i=i0,il
                  pp(i, j, k) = residual(i, j, k) + beta * (pp(i, j, k) - omega * vv(i, j, k))
               end do
            end do
         end do
!$omp enddo
!$omp single

         select case(precond_S)
            case ('JACOBI')
               call pre_jacobi3D (pp_prec(i0:il,j0:jl,:), pp(i0:il,j0:jl,:), Prec_xevec_8, &
                                  (Ni-pil_e)-(1+pil_w)+1, (Nj-pil_n)-(1+pil_s)+1, nk, &
                                  Prec_ai_8,Prec_bi_8,Prec_ci_8)
            case default
               pp_prec(i0:il,j0:jl,:) = pp(i0:il,j0:jl,:)
         end select

         ! Compute search direction
         call matvec(pp_prec, vv)
         tau = dist_dotproduct(hatr0, vv, minx, maxx, miny, maxy, i0, il, j0, jl, nk)

         if (almost_zero(tau)) then
            write(lun_out, *) 'WARNING : BiCGSTAB breakdown (tau=0)'
            force_restart = .true.

            alpha = 0.0d0
         else
            alpha = rho / tau
         end if

!$omp end single
!$omp do private (i,j,k)
         do k=1,nk
            do j=j0,jl
               do i=i0,il
                  ss(i, j, k) = residual(i, j, k) - alpha * vv(i, j, k)
               end do
            end do
         end do
!$omp enddo
!$omp single

         select case(precond_S)
            case ('JACOBI')
                  call pre_jacobi3D (ss_prec(i0:il,j0:jl,:), ss(i0:il,j0:jl,:), Prec_xevec_8, &
                                     (Ni-pil_e)-(1+pil_w)+1, (Nj-pil_n)-(1+pil_s)+1, nk, &
                                     Prec_ai_8,Prec_bi_8,Prec_ci_8)
            case default
               ss_prec(i0:il,j0:jl,:) = ss(i0:il,j0:jl,:)
         end select

         call matvec(ss_prec, tt)
         tau = dist_dotproduct(tt, tt, minx, maxx, miny, maxy, i0, il, j0, jl, nk)

         if (almost_zero(tau)) then
            write(lun_out, *) 'WARNING : BiCGSTAB breakdown (tt=0)'
            force_restart = .true.

            omega = 0.0d0
         else
            omega = dist_dotproduct(tt, ss, minx, maxx, miny, maxy, i0, il, j0, jl, nk) / tau
         end if

         rho_old = rho
         rho = -omega * (dist_dotproduct(hatr0, tt, minx, maxx, miny, maxy, i0, il, j0, jl, nk))

!$omp end single
!$omp do private (i,j,k)
         do k=1,nk
            do j=j0,jl
               do i=i0,il
                  ! Update the solution and the residual vectors
                  x(i, j, k) = x(i, j, k) + alpha * pp_prec(i, j, k) + omega * ss_prec(i, j, k)
                  residual(i, j, k) = ss(i, j, k) - omega * tt(i, j, k)
               end do
            end do
         end do
!$omp enddo
!$omp end parallel
         norm_residual = sqrt(dist_dotproduct(residual, residual, minx, maxx, miny, maxy, &
                                               i0, il, j0, jl, nk))

         if (force_restart) then
            call matvec(x, res_matvec)

            do k=1,nk
               do j=j0,jl
                  do i=i0,il
                     residual(i, j, k) = b(i, j, k) - res_matvec(i, j, k)
                  end do
               end do
            end do

            hatr0 = residual
            rho_old = 1.d0
            alpha = 1.d0
            omega = 1.d0
            rho = dist_dotproduct(hatr0, residual, minx, maxx, miny, maxy, i0, il, j0, jl, nk)
            norm_residual = sqrt(rho)

            vv(:, :, :) = 0.d0
            pp(:, :, :) = 0.d0

            force_restart = .false.
         end if
      end do

       conv = norm_residual / r0
      retval = iter

   end function krylov_fbicgstab


   integer function krylov_fgmres(x, matvec, b, x0, ni, nj, nk, &
                                  minx, maxx, miny, maxy, i0, il, j0, jl, &
                                  tolerance, maxinner, maxouter, precond_S, conv) result(retval)
      ! Flexible generalized minimum residual method (with restarts)
      ! solve A x = b.
      !
      integer, intent(in) :: ni, nj, nk, minx, maxx, miny, maxy, i0, il, j0, jl
      !
      ! The converged solution.
      real*8, dimension (minx:maxx, miny:maxy, nk), intent(out) :: x
      !
      ! Initial guess for the solution
      real*8, dimension (minx:maxx, miny:maxy, nk), intent(in) :: x0
      !
      ! Right hand side of the linear system.
      real*8, dimension (minx:maxx, miny:maxy, nk), intent(in) :: b
      !
      ! A matrix-vector product routine (A.*vv).
      interface
         subroutine matvec(v, prod)
#include "ldnh.cdk"
#include "glb_ld.cdk"
            real*8, dimension (ldnh_minx:ldnh_maxx, ldnh_miny:ldnh_maxy, l_nk), intent(in) :: v
            real*8, dimension (ldnh_minx:ldnh_maxx, ldnh_miny:ldnh_maxy, l_nk), intent(out) :: prod
         end subroutine
      end interface
      !
      ! Tolerance to achieve. The algorithm terminates when either the relative
      ! or the absolute residual is below tolerance.
      real*8, intent(in) :: tolerance
      !
      ! Restarts the method every maxinner inner iterations.
      integer, intent(in) :: maxinner
      !
      ! Specifies the maximum number of outer iterations.
      ! Iteration will stop after maxinner*maxouter steps
      ! even if the specified tolerance has not been achieved.
      integer, intent(in) :: maxouter
      !
      real*8, intent(out) :: conv
      ! Preconditioner
      character(len=*), intent(in) :: precond_S

      integer :: initer, outiter, nextit, i, j, k, it, ierr
      real*8 :: relative_tolerance, norm_residual, nu, dotprod, r0

      real*8, dimension(maxinner+1, maxinner) :: hessenberg
      real*8, dimension(maxinner+1) :: rot_cos, rot_sin, gg, xh

      real*8, dimension(minx:maxx,miny:maxy, nk, maxinner+1) :: vv
      real*8, dimension(minx:maxx,miny:maxy, nk, maxinner) :: ww
      real*8, dimension(minx:maxx, miny:maxy, nk) :: residual
      real*8, dimension(minx:maxx, miny:maxy, nk) :: res_matvec

      real*8, dimension(maxinner) :: dotprod_local

      logical almost_zero
      real*8 dist_dotproduct

      ! Here we go !

      retval = 0
      x(:,:,:) = x0(:,:,:)

      conv = 1.0d0
      residual(:,:,:) = 0.0d0

      do outiter = 1, maxouter

         ! Residual of the initial iterate
         call matvec(x, res_matvec)
         do k=1,nk
            do j=j0,jl
               do i=i0,il
                  residual(i, j, k) = b(i, j, k) - res_matvec(i, j, k)
               end do
            end do
         end do

         norm_residual = sqrt(dist_dotproduct(residual, residual, minx, maxx, miny, maxy, &
                                               i0, il, j0, jl, nk))

         if (outiter == 1) then
            relative_tolerance = tolerance * norm_residual
            r0 = norm_residual
         end if

         !  Initial guess is a good enough solution
         if (norm_residual <= relative_tolerance) then
            exit
         end if

         hessenberg    = 0.d0
         rot_cos       = 0.d0
         rot_sin       = 0.d0
         dotprod_local = 0.d0
         vv            = 0.d0

         gg(1)  = norm_residual
         gg(2:) = 0.0d0

         nu = 1.d0 / norm_residual
         do k=1,nk
            do j=j0,jl
               do i=i0,il
                  vv(i, j, k, 1) = residual(i, j, k) * nu
               end do
            end do
         end do

         initer = 0
         do while ((norm_residual > relative_tolerance) .and. (initer < maxinner))
            initer = initer + 1
            nextit = initer + 1

            select case(precond_S)
               case ('JACOBI')
                  call pre_jacobi3D ( res_matvec(i0:il,j0:jl,:), vv(i0:il,j0:jl,:,initer), Prec_xevec_8, &
                                      (Ni-pil_e)-(1+pil_w)+1, (Nj-pil_n)-(1+pil_s)+1, nk, &
                                      Prec_ai_8,Prec_bi_8,Prec_ci_8 )
               case default
                  res_matvec(i0:il,j0:jl,:) = vv(i0:il,j0:jl,:,initer)
            end select

            ww(i0:il,j0:jl,:,initer) = res_matvec(i0:il,j0:jl,:)

            ! Matrix-vector multiplication : vv(:,:,:,nextit) = A * vv(:,:,:,initer)
            call matvec(res_matvec, vv(:,:,:,nextit))

            ! Modified Gram-Schmidt orthogonalisation

!$omp parallel
!$omp do private (it,i,j,k,dotprod)
            do it=1,initer
                dotprod = 0.0d0
                do k=1,nk
                   do j=j0,jl
                      do i=i0,il
                         dotprod = dotprod + (vv(i, j, k, it) * vv(i, j, k, nextit))
                      end do
                   end do
                end do
                dotprod_local(it) = dotprod
            end do
!$omp enddo

!$omp single
            call RPN_COMM_allreduce(dotprod_local(:), hessenberg(:,initer), initer, "MPI_double_precision", "MPI_sum", "grid", ierr)
!$omp end single

!$omp do private (i,j,k,it)
            do it=1,initer
               do k=1,nk
                  do j=j0,jl
                     do i=i0,il
                        vv(i, j, k, nextit) = vv(i, j, k, nextit) - hessenberg(it,initer) * vv(i, j, k, it)
                     end do
                  end do
               end do
            end do
!$omp enddo
!$omp end parallel

            hessenberg(nextit,initer) = sqrt(dist_dotproduct(vv(:,:,:,nextit), vv(:,:,:,nextit), minx, maxx, miny, maxy, i0, il, j0, jl, nk))

            ! Watch out for happy breakdown
            if (.not. almost_zero( hessenberg(nextit,initer) )) then
               nu = 1.d0 / hessenberg(nextit,initer)
!$omp parallel private (i,j,k)
!$omp do
               do k=1,nk
                  do j=j0,jl
                     do i=i0,il
                        vv(i, j, k, nextit) = vv(i, j, k, nextit) * nu
                     end do
                  end do
               end do
!$omp enddo
!$omp end parallel
            end if

            ! Form and store the information for the new Givens rotation
            if (initer > 1) then
               call givens_rotations(rot_cos(1:initer-1), rot_sin(1:initer-1), &
                                     hessenberg(1:initer,initer), initer-1,    &
                                     hessenberg(1:initer,initer))
            end if

            nu = sqrt(dot_product(hessenberg(initer:nextit,initer), hessenberg(initer:nextit,initer)))
            if (.not. almost_zero( nu ) ) then
               rot_cos(initer) =  hessenberg(initer,initer) / nu
               rot_sin(initer) = -hessenberg(nextit,initer) / nu
               hessenberg(initer,initer) = rot_cos(initer) * hessenberg(initer,initer) - rot_sin(initer) * hessenberg(nextit,initer)
               hessenberg(nextit,initer) = 0.d0
               call givens_rotations(rot_cos(initer:), rot_sin(initer:), gg(initer:nextit), 1, gg(initer:nextit))
            end if

            norm_residual = abs(gg(nextit))
         end do

         ! At this point either the maximum number of inner iterations
         ! was reached or the absolute residual is below the scaled tolerance.

         ! Solve "hessenberg * xh = gg" ("hessenberg" is upper triangular, xh is the unkonwn)
         do j = initer,1,-1
            xh(j) = gg(j) / hessenberg(j,j)
            do i = j-1,1,-1
               gg(i) = gg(i) - xh(j) * hessenberg(i,j)
            enddo
         enddo

         ! updating solution

         do it = initer,1,-1
            nu = xh(it)
            do k=1,nk
               do j=j0,jl
                  do i=i0,il
                     x(i, j, k) = x(i, j, k) + nu * ww(i, j, k, it)
                  end do
               end do
            end do
         end do

      end do

      conv = norm_residual / r0
      retval = outiter - 1

   end function krylov_fgmres

   subroutine givens_rotations (rot_cos, rot_sin, vin, k, vrot)
      ! Apply a sequence of k Givens rotations
      integer, intent(in) :: k
      real*8, dimension(:), intent(in) :: rot_cos, rot_sin
      real*8, dimension(:), intent(in) :: vin
      real*8, dimension(:), intent(out) :: vrot

      real*8 :: rot1, rot2
      integer :: i
      vrot = vin
      do i=1,k
         rot1 = rot_cos(i) * vrot(i) - rot_sin(i) * vrot(i+1)
         rot2 = rot_sin(i) * vrot(i) + rot_cos(i) * vrot(i+1)
         vrot(i)   = rot1
         vrot(i+1) = rot2
      end do
   end subroutine givens_rotations

end module krylov_mod
