!-------------------------------------- LICENCE BEGIN ------------------------------------
!Environment Canada - Atmospheric Science and Technology License/Disclaimer,
!                     version 3; Last Modified: May 7, 2008.
!This is free but copyrighted software; you can use/redistribute/modify it under the terms
!of the Environment Canada - Atmospheric Science and Technology License/Disclaimer
!version 3 or (at your option) any later version that should be found at:
!http://collaboration.cmc.ec.gc.ca/science/rpn.comm/license.html
!
!This software is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
!without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
!See the above mentioned License/Disclaimer for more details.
!You should have received a copy of the License/Disclaimer along with this software;
!if not, you can write to: EC-RPN COMM Group, 2121 TransCanada, suite 500, Dorval (Quebec),
!CANADA, H9P 1J3; or send e-mail to service.rpn@ec.gc.ca
!-------------------------------------- LICENCE END --------------------------------------

subroutine clsgs4(thl, tve, qw, qc, frac, fnn, c1, &
                  zn, ze, s, ps, mg, a, b, c, at2t, &
                  at2m, at2e, n, nk)
   use phy_options
   implicit none
#include <arch_specific.hf>

   integer n, nk
   real thl(n,nk), tve(n,nk), qw(n,nk), qc(n,nk)
   real frac(n,nk), fnn(n,nk)
   real c1(n,nk), zn(n,nk), ze(n,nk), s(n,nk)
   real a(n,nk), b(n,nk), c(n,nk)
   real at2t(n,nk), at2m(n,nk), at2e(n,nk), ps(n)
   real mg(n)

!@Author
!          J. Mailhot (Jun 2002)
!
!@Revision
! 001      J. Mailhot (Feb 2003) Clipping at upper levels
! 002      S. Belair  (Apr 2003) Minimum values of 50 m for ZE and ZN
!                                in calculation of sigmase.
! 003      A-M. Leduc (Jun 2003) Pass ps to blweight ---> blweight2
! 004      J. P. Toviessi ( Oct. 2003) - IBM conversion
!               - calls to vslog routine (from massvp4 library)
!               - unnecessary calculations removed
!               - etc.
! 005      L. Spacek (Dec 2007) - add "vertical staggering" option
!                                 change the name to clsgs3
! 006      A. Zadra (Oct 2015) -- add land-water mask (MG) to input and
!               add a new user-defined reduction parameter
!               (which may or may not depend on the land-water fraction) 
!               to control the flux enhancement factor (fnn)
!
!@Object Calculate the boundary layer sub-grid-scale cloud properties
!
!@Arguments
!
!          - Input -
! THL      cloud water potential temperature
! TVE      virtual temperature on 'E' levels
! QW       total water content
!
!          - Output -
! QC       cloud water content
! FRAC     cloud fraction
! FNN      flux enhancement factor (fn) times cloud fraction (N)
!
!          - Input -
! C1       constant C1 in second-order moment closure
! ZN       length scale for turbulent mixing (on 'E' levels)
! ZE       length scale for turbulent dissipationa (on 'E' levels)
! S        sigma levels
! PS       surface pressure (in Pa)
! MG       land-water mask (0 to 1)
! A        thermodynamic coefficient
! B        thermodynamic coefficient
! C        thermodynamic coefficient
! AT2T     coefficients for interpolation of T,Q to thermo levels
! AT2M     coefficients for interpolation of T,Q to momentum levels
! AT2E     coefficients for interpolation of T,Q to energy levels
! N        horizontal dimension
! NK       vertical dimension
!
!
!@Notes
!          Implicit (i.e. subgrid-scale) cloudiness scheme for unified
!             description of stratiform and shallow, nonprecipitating
!             cumulus convection appropriate for a low-order turbulence
!             model based on Bechtold et al.:
!            - Bechtold and Siebesma 1998, JAS 55, 888-895
!            - Cuijpers and Bechtold 1995, JAS 52, 2486-2490
!            - Bechtold et al. 1995, JAS 52, 455-463
!            - Bechtold et al. 1992, JAS 49, 1723-1744

   include "thermoconsts.inc"

   integer j, k, itotal
   real epsilon, qcmin, qcmax, a1
   real*8 gravinv
   real dz(n,nk), dqwdz(n,nk), dthldz(n,nk), sigmas(n,nk), &
        sigmase(n,nk), q1(n,nk), weight(n,nk), work(n,nk)

!------------------------------------------------------------------------

      epsilon = 1.0e-10
      qcmin   = 1.0e-6
      qcmax   = 1.0e-3
      gravinv = 1.0/dble(grav)

!AZ: The new parameters below (fnn_mask and fnn_reduc) are the two parameters 
!    that should be read from the gem_settings
!       fnn_mask = T/F means "use (do not use) the land-water fraction
!            to modulate the fnn reduction
!       fnn_reduc = is the reduction parameter that should vary within
!            the range 0. to 1.; 1 means "keep the original estimate"; 
!            any value smaller than 1 means " multiply the original fnn 
!            by a factor fnn_reduc"
!
!    The default values should be
!       fnn_mask = .false.
!       fnn_reduc = 1
!
!    The values tested and chosen for the RDPS-10km are
!       fnn_mask = .true.
!       fnn_reduc = 0.8

!MD: This initialization to 0. must be revisited on intel and pgi14
!    as it will hurt performance

      dz      = 0. ; dqwdz = 0. ; dthldz = 0. ; sigmas = 0.
      sigmase = 0. ; q1    = 0. ; weight = 0. ; work   = 0.
!
!       1.     Vertical derivative of THL and QW
!       ----------------------------------------
!
      call tothermo(thl, thl, at2m,at2m,n,nk+1,nk,.false.)
      call tothermo(qw,  qw,  at2m,at2m,n,nk+1,nk,.false.)

      do k=1,nk-1
      do j=1,n
        work (j,k) = s(j,k+1)/s(j,k)
      end do
      end do
      call vslog(work ,work ,n*(nk-1))
      do k=1,nk-1
      do j=1,n
        dz(j,k) = -rgasd*tve(j,k)*work (j,k)*gravinv
      end do
      end do
      dz(1:n,nk) = 0.0

      call dvrtdf ( dthldz, thl, dz, n, n, n, nk)
      call dvrtdf ( dqwdz , qw , dz, n, n, n, nk)
!
!       2.     Standard deviation of s and normalized saturation deficit Q1
!       -------------------------------------------------------------------
!
      do k=1,nk-1
      do j=1,n
        work (j,k) = c1(j,k)*max(zn(j,k),50.)*max(ze(j,k),50.)
      end do
      end do
      call vssqrt(work ,work ,n*(nk-1))
!
      call tothermo(a,  a,  at2t,at2t,n,nk+1,nk,.true.)
      call tothermo(b,  b,  at2t,at2t,n,nk+1,nk,.true.)
!
      do k=1,nk-1
      do j=1,n
!                                              sigmas (cf. bcmt 1995 eq. 10)
!                                        (computation on 'e' levels stored in sigmase)
        sigmase(j,k) = work (j,k) * &
                    abs(a(j,k)*dqwdz(j,k) - b(j,k)*dthldz(j,k) )
      end do
      end do
!
      call tothermo(sigmas,sigmase,  at2e,at2e,n,nk+1,nk,.false.)
!
      do k=2,nk-1
      do j=1,n
!                                              (back to full levels)
!                                              normalized saturation deficit
        q1(j,k) = c(j,k) / ( sigmas(j,k) + epsilon )
        q1(j,k) = max ( -6. , min ( 4. , q1(j,k) ) )
      end do
      end do
!
      sigmas(1:n,1) = 0. ; sigmas(1:n,nk) = 0.
      q1    (1:n,1) = 0. ; q1    (1:n,nk) = 0. ;
!
!       3.     Cloud properties
!       -----------------------
!                                              cloud fraction, cloud water content
!                                              and flux enhancement factor
!                                              (cf. BS 1998 Appendix B)
      do k=2,nk-1
      do j=1,n
!
        if( q1(j,k) .gt. -1.2 ) then
          frac(j,k) = max ( 0. , min ( 1. , &
                            0.5 + 0.36*atan(1.55*q1(j,k)) ) )
        elseif( q1(j,k) .ge. -6.0 ) then
          frac(j,k) = exp ( q1(j,k)-1.0 )
        else
          frac(j,k) = 0.0
        endif
!
        if( q1(j,k) .ge. 0.0 ) then
          qc(j,k) = exp( -1.0 ) + 0.66*q1(j,k) + 0.086*q1(j,k)**2
        elseif( q1(j,k) .ge. -6.0 ) then
          qc(j,k) = exp( 1.2*q1(j,k)-1.0 )
        else
          qc(j,k) = 0.0
        endif
!
        qc(j,k) = min ( qc(j,k)*( sigmas(j,k) + epsilon ) &
                        , qcmax )
!
        fnn(j,k) = 1.0
        if( q1(j,k).lt.1.0 .and. q1(j,k).ge.-1.68 ) then
          fnn(j,k) = exp( -0.3*(q1(j,k)-1.0) )
        elseif( q1(j,k).lt.-1.68 .and. q1(j,k).ge.-2.5 ) then
          fnn(j,k) = exp( -2.9*(q1(j,k)+1.4) )
        elseif( q1(j,k).lt.-2.5 ) then
          fnn(j,k) = 23.9 + exp( -1.6*(q1(j,k)+2.5) )
        endif
!                                              flux enhancement factor * cloud fraction
!                                              (parameterization formulation)
        fnn(j,k) = fnn(j,k)*frac(j,k)
        if( q1(j,k).le.-2.39 .and. q1(j,k).ge.-4.0 ) then
          fnn(j,k) = 0.60
        elseif( q1(j,k).lt.-4.0 .and. q1(j,k).ge.-6.0 ) then
          fnn(j,k) = 0.30*( q1(j,k)+6.0 )
        elseif( q1(j,k).lt.-6.0 ) then
          fnn(j,k) = 0.0
        endif

!AZ This is the where the original estimate of the fnn is
!   reduced depending on the user's choice (defined by the
!   input paremeters fnn_mask and fnn_reduc)
!
        a1 = 1.
        if (fnn_mask) then
          a1 = fnn_reduc + (1.-fnn_reduc)*max(0.,min(1.,mg(j)))
        else
          a1 = fnn_reduc
        endif
        fnn(j,k) = a1*fnn(j,k)

      end do
      end do

      frac(1:n,1) = 0. ; frac(1:n,nk) = 0.
      fnn (1:n,1) = 0. ; fnn (1:n,nk) = 0.
      qc  (1:n,1) = 0. ; qc  (1:n,nk) = 0.

      call blweight2 ( weight, s, ps, n, nk )

      do k=1,nk
      do j=1,n
        frac(j,k) = frac(j,k)*weight(j,k)
        fnn (j,k) = fnn (j,k)*weight(j,k)
        qc  (j,k) = qc  (j,k)*weight(j,k)
      end do
      end do

!------------------------------------------------------------------------
   return
end subroutine clsgs4
