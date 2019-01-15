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
!author
!     Lubos Spacek - January 2010
!
!revision
! v4_12 - Spacek, L.     - Single precision synthesis (see harmons, fourier2)
!
!
!*module harmonsphere - spherical harmonics transform based on
!                       Drake, Worley, D'Azevedo : Spherical Hramonics
!                       Transform Algorithms, ACM Transactions on Mathematical
!                       Software (TOMS), Volume 35 ,  Issue 3  (October 2008)
!
Module harmonsphere
  !
  ! fortran95 spherical harmonic module.
  !
  ! version 0.0  14.1.2010 (first version using fft8 and triangular truncation)
  ! Lubos Spacek <Lubos.Spacek@ec.gc.ca>

  Implicit None
#include <arch_specific.hf>


  ! everything private to module, unless otherwise specified.

  Private
  Public :: harmon, harmon_init, harmon_destroy, harmona, harmons


  Type harmon

     ! ni is number of longitudes
     Integer :: ni = 0

     ! nj is number of gaussian latitudes.
     Integer :: nj = 0

     ! n2=nj/2 is number of gaussian latitudes.
     Integer :: n2 = 0

     ! truncation parameters - the largest Fourier wavenumber, the highest
     ! degree of the associated Legendre function and the highest degree
     ! of the associated Legendre function of order zero.
     ! Initial version, only triangular truncation is supported (mm = kk = nn)

     Integer :: mm = 0, kk = 0, nn = 0

     ! roo is sin(gaussian lats), wei are gaussian weights,
     Real(Kind(1.d0)), Dimension(:), Pointer :: roo, wei

     ! pp is associated Legendre functions, , gg is -(1-x**2)d(pnm)/dx,
     ! where x = gaulats.
     Real(Kind(1.d0)), Dimension(:,:), Pointer :: pp, gg
     !
     Logical :: isinitialized=.False.

  End Type harmon

Contains

  Subroutine harmon_init(harmon_dsc,ni,nj,mm,kk,nn)

    ! initialize a harmon object.

    Integer, Intent(in) :: ni,nj,mm,kk,nn
    Type (harmon), Intent(inout) :: harmon_dsc
    Integer :: npol,i,j,k,m,n

    harmon_dsc%ni = ni
    harmon_dsc%nj = nj
    harmon_dsc%n2 = nj/2
    harmon_dsc%mm = mm
    harmon_dsc%kk = kk
    harmon_dsc%nn = nn

    npol = (kk+1)*(kk+2)/2

    Allocate(harmon_dsc%pp(harmon_dsc%n2,0:npol))
    Allocate(harmon_dsc%gg(harmon_dsc%n2,0:npol))
    Allocate(harmon_dsc%roo(harmon_dsc%nj))
    Allocate(harmon_dsc%wei(harmon_dsc%nj))

    Call gauleg(harmon_dsc%roo,harmon_dsc%wei,nj)

    Call assleg(harmon_dsc%pp,harmon_dsc%gg,harmon_dsc%roo,kk,harmon_dsc%n2,npol)

    harmon_dsc%isinitialized = .True.

  End Subroutine harmon_init

  Subroutine harmon_destroy(harmon_dsc)

    ! deallocate pointers in harmon object.

    Type (harmon), Intent(inout) :: harmon_dsc

    If (.Not. harmon_dsc%isinitialized) Return

    Deallocate(harmon_dsc%pp)
    Deallocate(harmon_dsc%gg)
    Deallocate(harmon_dsc%roo)
    Deallocate(harmon_dsc%wei)

  End Subroutine harmon_destroy

  Subroutine harmona(harmon_dsc,gp,sc,ni,nj,mm,kk,nn)

    Type (harmon),                            Intent(In)     :: harmon_dsc
    Real(Kind(1.d0)),    Dimension(ni,nj),    Intent(InOut)  :: gp
    Complex(Kind(1.d0)), Dimension(nn+1,mm+1),Intent(Out)    :: sc
    Integer,                                  Intent(In)     :: ni,nj,mm,kk,nn
    !work varibles and arrays
    Integer :: i,j,k,l,m,n
    Complex(Kind(1.d0)), Dimension(ni,nj)                    :: fc
    Complex(Kind(1.d0)), Dimension(harmon_dsc%n2)            :: fcp,fcm
    !
    Call fourier(ni, nj, mm, gp, fc, -1)

    Do j=1,nj
       fc(:,j)=harmon_dsc%wei(j)*fc(:,j);
    Enddo

    fc(mm+2:ni,:) = (0.0d0,0.0d0)
    k=0
    l=mm
    Do m=0,mm
       Do j=1,harmon_dsc%n2
          fcp(j)=fc(m+1,j)+fc(m+1,nj-j+1)
          fcm(j)=fc(m+1,j)-fc(m+1,nj-j+1)
       Enddo
       Do n=m+1,nn+1,2
          sc(n,m+1)       = Dot_product(harmon_dsc%PP(:,k),fcp)
          if(n+1<=nn)                                               &
          sc(n+1,m+1)     = Dot_product(harmon_dsc%PP(:,k+1),fcm)
          k=k+2
       Enddo
       k=l+1;l=k+mm-m-1
    Enddo
  End Subroutine harmona

  Subroutine harmons(harmon_dsc,gp,sc,ni,nj,mm,kk,nn)

    Type (harmon),                            Intent(In)     :: harmon_dsc
!   GEM needs single precision only
!    Real(Kind(1.d0)),    Dimension(ni,nj),    Intent(InOut)  :: gp
!    Complex(Kind(1.d0)), Dimension(nn+1,mm+1),Intent(In)     :: sc
    Real(Kind(1.0)),    Dimension(ni,nj),    Intent(InOut)  :: gp
    Complex(Kind(1.0)), Dimension(nn+1,mm+1),Intent(In)     :: sc
    Integer,                                  Intent(In)     :: ni,nj,mm,kk,nn
    !work varibles and arrays
    Integer :: i,j,k,l,m,n
    Complex(Kind(1.d0)),          Dimension(mm+1,nj)         :: sf
    Real(Kind(1.d0)),             Dimension(:), Allocatable  :: am

    n=2*Ceiling(Real((nn+1)/2))+1
    Allocate(am(n))
    am(1)=1;Do i=2,n; am(i)=-am(i-1);Enddo
    k=0
    l=mm
    Do m=0,mm
!$omp parallel
!$omp do
       Do j=1,harmon_dsc%n2
          sf(m+1,j)=0.
          sf(m+1,nj-j+1) =0.
          sf(m+1,j)      =  sf(m+1,j)                                   &
               +Dot_product(harmon_dsc%PP(j,k:l),sc(m+1:nn+1,m+1))
          sf(m+1,nj-j+1) =  sf(m+1,nj-j+1)                              &
               +Dot_product(harmon_dsc%PP(j,k:l),sc(m+1:nn+1,m+1)*am(1:nn-m+1))
       Enddo
!$omp enddo
!$omp end parallel
       k=l+1;l=k+mm-m-1
    Enddo

    Call fourier2(ni, nj, mm, gp, sf, 1)
    deallocate(am)
  End Subroutine harmons


  Subroutine assleg( p , g , x , ntrunc , nlat, npol)
    !
    !     Calculates associate Legendre functions and their derivatives
    !     based on allp2.f
    !
    !     p        - associate Legendre functions
    !     g        - derivatives of associate Legendre functions
    !     x        = sin( latitude )
    !     Ntrunc   = truncation
    !     npol     = (ntrunc+1)*(ntrunc+2)/2
    !
    Implicit None
    Integer ntrunc, nlat, npol
    Real*8 p(nlat,0:npol) , g(nlat,0:npol)
    Real*8 x(nlat)

    Real*8 xp , xp2,  p0, enm, fnm
    Integer il , m , l , n, k, i
    Integer :: s(0:ntrunc)
    !-------------------------------------------------------------------
    s(0)=0.;k=ntrunc+1
    Do i=1,ntrunc;s(i)=s(i-1)+k;k=k-1;Enddo

       !     p[m,m]

       Do il=1,nlat
          xp2 = Sqrt( 1.d0 - x(il) ** 2 )
          p(il,0) = Sqrt(.5d0)
          Do m=1,ntrunc
             xp = dble(m)
             p(il,s(m)) = Sqrt( (2.d0*xp+1.d0)/(2.d0*xp) )   &
                  * xp2 * p(il,s(m-1))
          Enddo
       Enddo

       !     g[m,m]

       Do il=1,nlat
          Do m=0,ntrunc
             xp = dble(m)
             g(il,s(m)) = - x(il)*xp * p(il,s(m))
          Enddo
       Enddo

       !     p[l,m] , g[l,m]  l > m

       Do n=1,ntrunc
          Do  m=0, ntrunc-n
             p0 = dble(m+n)
             xp = dble(m)
             enm = Sqrt( ((p0*p0-xp*xp)*(2.d0*p0+1.0))/(2.d0*p0-1.d0) )
             fnm = Sqrt( (2.d0*p0+1.d0)/((p0*p0-xp*xp)*(2.d0*p0-1.d0)) )

             Do il = 1, nlat
                p(il,s(m)+n) = ( x(il) * p0 * p(il,s(m)+n-1)          &
                     -  g(il,s(m)+n-1) ) * fnm
                g(il,s(m)+n) = enm * p(il,s(m)+n-1) - x(il) * p0 * p(il,s(m)+n)
             Enddo
          Enddo
       Enddo

       Return
     End Subroutine assleg

     Subroutine gauleg(x,w,n)
       ! This subroutine returns arrays x(1:n) and w(1:n) containing
       ! the points and weights for the Gauss-Legendre-points quadrature
       ! (Based on GAULEG from Numerical Recipes, Chap 4.5, p145)

       Implicit None


       Real(Kind(1.d0)), Parameter :: Eps = 3.0d-14
       Real(Kind(1.d0)), Parameter :: Pi = 3.141592653589793228d0
       Integer,          Parameter :: itermax = 10
       Integer, Intent(In) :: n
       Real(Kind(1.d0)), Dimension(n), Intent(InOut) :: x,w

       Integer :: i,j,m, iter
       Real(Kind(1.d0)) :: p1,p2,p3,pp,xl,xm,z,z1

       Do i=1,n
          z = -Cos(pi*(i - 0.25)/(n + 0.5))
          Do iter=1,itermax
             p1 = 1.0
             p2 = 0.0

             Do j=1,n
                p3 = p2
                p2 = p1
                p1 = ((2.0*j - 1.0)*z*p2 - (j - 1.0)*p3)/j
             End Do

             pp = n*(z*p1 - p2)/(z*z - 1.0E+00)
             z1 = z
             z  = z1 - p1/pp
             If(Abs(z - z1) < eps) go to 10
          End Do
          Stop "converge failure in gauleg"

10        Continue

          x (i)     = z
          w (i)     = 2.0/((1.0 - z*z)*pp*pp)

       End Do

     End Subroutine gauleg

     subroutine fourier(ni, nj, ntrunc, gpdata, coeff, iori)
       implicit none
       INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(12,307)
       integer, intent(in) :: ni, nj, ntrunc, iori
       real(dp), dimension(ni,nj), intent(inout) :: gpdata
       complex(dp), dimension(ni,nj), intent(inout) :: coeff

       real(dp)                       :: pri8
       real(dp), dimension((ni+2)*nj) :: wrk
       integer ::  nlons,nlats,i,j,m,n


       nlons = ni
       nlats = nj

       if (ntrunc .gt. nlons/2) then
          stop 'ntrunc <= nlons/2 in fourier'
       end if

       !==> forward transform.

       if (iori .eq. -1) then

          !==> copy the gpdata into the work array.
          !    fft is real
          n = 0
          wrk = 0.
          do j=1,nlats
             do i=1,nlons+2  
                n = n + 1
                wrk(n) = 0.0
                if (i .le. nlons) then
                   wrk(n) = gpdata(i,j)
                end if
             enddo
          enddo

          call itf_fft_set(nlons,'PERIODIC',pri8)
          call ffft8(wrk,1,nlons+2,nlats,-1)

          n = -1
          do j=1,nlats
             do m=1,(nlons/2)+1
                n = n + 2
                if (m .le. ntrunc+1) then
                   coeff(m,j) = cmplx(wrk(n),wrk(n+1)) 
                end if
             enddo
          enddo

          !==> inverse transform.

       else if (iori .eq. +1) then

          wrk = 0.  ! ; coeff(1,:)=cmplx(real(coeff(1,:)),0)

          n = -1
          do j=1,nj
             do i=1,ni/2+1
                n = n + 2
                wrk(n) = real(coeff(i,j))
                wrk(n+1) = aimag(coeff(i,j))
             enddo
          enddo

          call itf_fft_set(ni,'PERIODIC',pri8)
          call ffft8(wrk,1,ni+2,nj,1)

          n = 0
          do j=1,nj
             do i=1,ni+2  
                n = n + 1
                if (i .le. ni) then
                   gpdata(i,j) = wrk(n)
                end if
             enddo
          enddo

       else
          stop 'fft: iori must be +1 or -1'
       end if

     end subroutine fourier

     subroutine fourier2(ni, nj, ntrunc, gpdata, coeff, iori)
       implicit none
       INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(12,307)
       integer, intent(in) :: ni, nj, ntrunc, iori
! GEM needs only single precission
!       real(dp), dimension(ni,nj), intent(inout) :: gpdata
       real,        dimension(ni,nj),       intent(inout) :: gpdata
       complex(dp), dimension(ntrunc+1,nj), intent(inout) :: coeff

       real(dp)                       :: pri8
       real(dp), dimension((ni+2)*nj) :: wrk
       integer ::  nlons,nlats,i,j,m,n


       nlons = ni
       nlats = nj
       ! ntrunc = sphere_dat%ntrunc
       if (ntrunc .gt. nlons/2) then
          print *, 'ntrunc must be less than or equal to nlons in fourier'
          stop
       end if

       !==> forward transform.

       if (iori .eq. -1) then

          !==> copy the gpdata into the work array.
          !    fft is real
          n = 0
          wrk = 0.
          do j=1,nlats
             do i=1,nlons+2  
                n = n + 1
                wrk(n) = 0.0
                if (i .le. nlons) then
                   wrk(n) = gpdata(i,j)
                end if
             enddo
          enddo

          call itf_fft_set(nlons,'PERIODIC',pri8)
          call ffft8(wrk,1,nlons+2,nlats,-1)

          n = -1
          do j=1,nlats
             do m=1,(nlons/2)+1
                n = n + 2
                if (m .le. ntrunc+1) then
                   coeff(m,j) = cmplx(wrk(n),wrk(n+1)) 
                end if
             enddo
          enddo

          !==> inverse transform.

       else if (iori .eq. +1) then

          wrk = 0.  ! ; coeff(1,:)=cmplx(real(coeff(1,:)),0)

          n = -1
          do j=1,nj
             do i=1,ntrunc+1
                n = n + 2
                wrk(n) = real(coeff(i,j))
                wrk(n+1) = aimag(coeff(i,j))
             enddo
             n=n+ni-2*ntrunc
          enddo

          call itf_fft_set(ni,'PERIODIC',pri8)
          call ffft8(wrk,1,ni+2,nj,1)

          n = 0
          do j=1,nj
             do i=1,ni+2  
                n = n + 1
                if (i .le. ni) then
                   gpdata(i,j) = wrk(n)
                end if
             enddo
          enddo

       else
          stop 'fft: iori must be +1 or -1'
       end if

     end subroutine fourier2


   End Module harmonsphere
