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

      subroutine gwd8 ( d, f, vb, eb, sized, sizef, sizev, sizee, &
                        std_p_prof, tau, kount, trnch,    &
                        n, m, nk, itask )
      use phy_options
      use phybus
      implicit none
#include <arch_specific.hf>
      integer itask, sized, sizef, sizev, sizee, trnch, n, m, nk, kount
      real, target :: d(sized), f(sizef), vb(sizev), eb(sizee)
      real std_p_prof(nk)
      real tau
!
!author
!          J.Mailhot RPN(May1990)
!
!Revision
! 001      see version 5.5.0 for previous history
!
!Object
!          to model the gravity wave drag
!
!Arguments
!
!          - Input/Output -
! f        field of permanent physics variables
! sizef    dimension of F
! u        u component of wind as input
!          u component of wind modified by the gravity wave
!          drag as output
! v        v component of wind as input
!          v component of wind modified by the gravity wave
!          drag as output
!
!          - Input -
! t        virtual temperature
! s        local sigma levels
! std_p_prof  standard pressure profil
!          - Output -
! rug      gravity wave drag tendency on the u component of real
!          wind
! rvg      gravity wave drag tendency on the v component of real
!          wind
! run      non-oro. gravity wave drag tendency on the u component of real
!          wind
! rvn      non-oro. gravity wave drag tendency on the v component of real
!          wind
! rtn      non-oro. gravity wave drag tendency on temperature
!
!          - Input -
! tau      timestep times a factor: 1 for two time-level models
!                                   2 for three time-level models
! trnch    index of the vertical plane(ni*nk) for which the
!          calculations are done.
! n        horizontal dimension
! m        1st dimension of u,v,t
! nk       vertical dimension
! itask    number for multi-tasking
!
!notes
!          this routine needs at least:
!          ( 12*nk + 12 )*n + 3*nk words in dynamic allocation
!            +3*nk       -"-         for local sigma
!            +2*nk       -"-         for gather on s, sh
!                           - 3*nk   s1,s2,s3 change from 1d to 2d

include "thermoconsts.inc"

!***********************************************************************
!     Automatic arrays
!***********************************************************************
!
      real work(n,nk), tvirt(n,nk+1)
      real*8, dimension(n)    :: fcorio, land, launch, mtdir8, pp, slope8, &
                                 sxx8, syy8, sxy8, xcent8
      real*8, dimension(n,nk) :: tt, te, uu, vv, sigma, s1, s2, s3, &
                                 utendgw, vtendgw,utendno , vtendno, ttendno
!
!***********************************************************************
!
      integer i,j,k,is,nik
      logical envelop,blocking
!
      real, pointer, dimension(:)   :: p, zdhdxdy, zdhdx, zdhdy, zfcor, &
                                       zlhtg, zmg, zmtdir, zslope, zxcent
      real, pointer, dimension(:,:) :: u, v, t, s, rug, rvg, run, rvn, rtn
!
!--------------------------------------------------------------------
!
      if(gwdrag.eq.'NIL'.and..not.non_oro)return

      call init_gwd (eb, sizee, f, sizef, kount, trnch, n, nk)

        p      (1:m) =>  f(pmoins:)
        zdhdx  (1:m) =>  f(dhdx:)
        zdhdxdy(1:m) =>  f(dhdxdy:)
        zdhdy  (1:m) =>  f(dhdy:)
        zfcor  (1:m) => vb(fcor:)
        zlhtg  (1:m) =>  f(lhtg:)
        zmg    (1:m) =>  f(mg:)
        zmtdir (1:m) =>  f(mtdir:)
        zslope (1:m) =>  f(slope:)
        zxcent (1:m) =>  f(xcent:)

        u(1:m,1:nk) =>  d(uplus:)
        v(1:m,1:nk) =>  d(vplus:)
        t(1:m,1:nk) =>  d(tplus:)
        s(1:m,1:nk) =>  d(sigm:)
      rug(1:m,1:nk) => vb(ugwd:)
      rvg(1:m,1:nk) => vb(vgwd:)
      run(1:m,1:nk) => vb(ugno:)
      rvn(1:m,1:nk) => vb(vgno:)
      rtn(1:m,1:nk) => vb(tgno:)

      envelop = .true.
      blocking = .true.
      nik=n*nk-1

      call mfotvt(tvirt,d(tplus),d(huplus),n,nk+1,n)
      call tothermo(work,tvirt, vb(at2t),vb(at2m),n,nk+1,nk,.false.)
!
!
!     tt   - temperature aux niveaux pleins
!     uu   - composante u du vent  (vent reel)
!     vv   - composante v du vent  (vent reel)
      do 100 k=1,nk
         do 100 j=1,n
            tt(j,k) = work(j,k)
            uu(j,k) = u(j,k)
            vv(j,k) = v(j,k)
  100    continue
         do 105 j=1,n
            pp(j) = p(j)
  105    continue

!     POINTEUR POUR LA ROUTINE DE GWD : -1 = CONTINENT
!                                        0 = OCEAN

      do 110 j=1,n
         land(j) = - abs( nint( zmg(j) ) )
  110    continue

!     s1, s2, s3 => 2d
!
!
!     s1    - demi-niveaux
!     s2    - niveaux pleins
!     s3    - demi-niveaux

      call tothermo(tvirt,work, vb(at2t),vb(at2m),n,nk+1,nk,.true.)
      do k=1 ,nk-1
         do j=1,n
            s1(j,k)=0.5*(s(j,k)+s(j,k+1))
            s2(j,k)=dble(s(j,k))

!     te   - temperature aux demi-niveaux
            te(j,k) = work(j,k)
         enddo
         call vlog(s3(1,k),s1(1,k),n)
         call vlog(s2(1,k),s2(1,k),n)
         do j=1,n
           s3(j,k) = cappa*s3(j,k)
           s2(j,k) = cappa*s2(j,k)
         enddo
         call vexp(s3(1,k),s3(1,k),n)
         call vexp(s2(1,k),s2(1,k),n)
      enddo

      do j=1,n
         s1(j,nk)=0.5*(s(j,nk)+1.)
         s2(j,nk)=dble(s(j,nk))

!     te   - temperature aux demi-niveaux

         te(j,nk) = work(j,nk)+vb(at2t+nik+j)*(t(j,nk)-t(j,nk-1))
       enddo 

       call vlog(s3(1,nk),s1(1,nk),n)
       call vlog(s2(1,nk),s2(1,nk),n)
       do j=1,n
          s3(j,nk) = cappa*s3(j,nk)
          s2(j,nk) = cappa*s2(j,nk)
       enddo
       call vexp(s3(1,nk),s3(1,nk),n)
       call vexp(s2(1,nk),s2(1,nk),n)

      do k=1,nk
         do j=1,n
            sigma(j,k) = s(j,k)
         end do
      end do

      if(gwdrag.eq.'GWD86') then

      do i=1,n
         launch(i)  = zlhtg(i)
         sxx8  (i)  = zdhdx(i)
         syy8  (i)  = zdhdy(i)
         sxy8  (i)  = zdhdxdy(i)
         mtdir8(i)  = zmtdir(i)
         slope8(i)  = zslope(i)
         xcent8(i)  = zxcent(i)
         fcorio(i)  = zfcor(i)
      end do

        call sgoflx5 (uu, vv, utendgw, vtendgw                    , &
                      te, tt, sigma, s1                           , &
                      nk, n, 1, n                                 , &
                      grav, rgasd, cappa, tau, taufac             , &
                      land, launch, slope8, xcent8, mtdir8        , &
                      pp, fcorio, .true., .true., .false., .false., &
                      .false., sgo_stabfac, sgo_nldirfac, sgo_cdmin)

      do 200 k=1,nk
         do 200 j=1,n
            rug(j,k) = utendgw(j,k)
            rvg(j,k) = vtendgw(j,k)
  200    continue
     call apply_tendencies1 (d,sized,vb,sizev,f,sizef,uplus,ugwd,n,nk)
     call apply_tendencies1 (d,sized,vb,sizev,f,sizef,vplus,vgwd,n,nk)

      call serxst2(rug, 'GU', trnch, n, nk, 0.0, 1.0, -1)
      call serxst2(rvg, 'GV', trnch, n, nk, 0.0, 1.0, -1)

      endif

      if (non_oro) then

        call gwspectrum4( n      , n      , nk, &
                          sigma  , s1     , s2, &
                          s3     , pp     , te, &
                          tt     , uu     , vv, &
                          ttendno, vtendno, utendno,        &
                          hines_flux_filter, grav  , rgasd, &
                          tau    , rmscon , iheatcal,       &
                          kount  , trnch  ,                 &
                          std_p_prof, non_oro_pbot)

        do 201 k=1,nk
           do 201 j=1,n
              run(j,k) = utendno(j,k)
              rvn(j,k) = vtendno(j,k)
              rtn(j,k) = ttendno(j,k)
  201      continue
     call apply_tendencies1 (d,sized,vb,sizev,f,sizef,uplus,ugno,n,nk)
     call apply_tendencies1 (d,sized,vb,sizev,f,sizef,vplus,vgno,n,nk)
      endif

      return
      end
