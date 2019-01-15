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
!** s/p water_integrated
  Subroutine water_integrated(fbus,fsiz,vbus,vsiz, &
       tt,qq,qc,qi,qr,qgp,qnp,sigma,ps, &
       ni,nk)
    use phy_options
    use phybus
    Implicit None
#include <arch_specific.hf>

    Integer               :: fsiz,vsiz,ni,nk
    Real,Dimension(fsiz)  :: fbus(fsiz)
    Real,Dimension(vsiz)  :: vbus(vsiz)
    Real,Dimension(ni)    :: ps
    Real,Dimension(ni,nk) :: tt,qq,qc,qi,qr,qgp,qnp,sigma
!
!Author
!          L.Spacek, November 2011
!
!Revisions
! 001
!
!Object
!          Calculate integrated quantities of some variables
!
!Arguments
!
!          - Input -
! fbus     historic variables for the physics
! fsiz     dimension of fbus
! vsiz     dimension of vbus
! tt       temperature
! qq       humidity
! qc       total condensate mixing ratio at t+dT
! qi       ice mixing ratio (M-Y, K-Y) at t+dT
! qr       rain mixing ratio (microphy) at t+dT
! qgp      graupel mixing ratio (M-Y, K-Y) at t+dT
! qnp      snow    mixing ratio (M-Y) at t+dT
! sigma    vertical coordinate
! ni       horizontal running length
! nk       vertical dimension
!
!          - Input/Output -
! vbus     physics tendencies and other output fields from the physics
!          icw    - integrated cloud water/ice
!          iwv    - integrated water vapor
!          iwv700 - integrated water vapor (0-700 mb)
!          iwp    - integrated ice water
!          lwp2   - liquid water path (Sundqvist)
!          slwp   - integrated SLW (supercooled liquid water)
!          slwp2  - integrated SLW (bottom to s2)
!          slwp3  - integrated SLW (s2 to s3)
!          slwp4  - integrated SLW (s3 to s4)
!
    Logical               :: integrate=.False.
    Integer               :: i,k,ik
    Real                  :: tcel,frac,airdenm1
    Real,Dimension(ni)    :: temp1,temp2
    Real,Dimension(ni,nk) :: liquid, solid
!
!Implicites
!
    Include "thermoconsts.inc"
!
!
!Modules
! None
!
! Arrays 'liquid' and 'solid' are passed to intwat3  and used
! for diagnostic calculations only.
! Computaion of lwc and iwc used by radiation code is done in prep_cw.
!
    If (stcond.Eq.'NEWSUND'.Or. stcond.Eq.'CONSUN') Then
       integrate=.True.
       Do k=1,nk
          Do i=1,ni
             tcel = Min(0.,tt(i,k) - tcdk)
             temp1(i) = -.003102 * tcel*tcel
          End Do
          Call vsexp(temp2,temp1,ni )
          Do i=1,ni
             If (tt(i,k) .Ge. tcdk) Then
                liquid(i,k) =  qc(i,k)
                solid(i,k)  =  0.

             Else
                frac = .0059 + .9941 * temp2(i)
                liquid(i,k) = frac*qc(i,k)
                solid(i,k)  = (1.-frac)*qc(i,k)
             End If
          End Do
       End Do
    End If
!
    If (stcond(1:2)=='MP') Then
       integrate=.True.
       Do k=1,nk
          Do i=1,ni
             ik = (k-1)*ni+i-1
             airdenm1    = rgasd *tt(i,k)/(sigma(i,k)*ps(i))
             If ((qc(i,k)+qi(i,k)+qnp(i,k)) .Gt. airdenm1*1.e-5) Then
                fbus(fxp+ik)= 1.
             Else
                fbus(fxp+ik)= 0.
             Endif
             liquid(i,k) = qc(i,k)
             !note: for stcond=mp_p3, qnp is passed in zero and qi is the sum of all qitot for all ice categories
             solid(i,k)  = qi(i,k)+qnp(i,k)
          End Do
       End Do
    Endif
!
!
!     calcul de quantites integrees
!
    If (integrate) &
    Call intwat3(vbus(icw),vbus(iwv),vbus(iwv700),vbus(iwp),vbus(lwp2), &
         vbus(slwp),vbus(slwp2),vbus(slwp3),vbus(slwp4), &
         tt,qq,liquid,solid,sigma,ps,ni,nk)

  End Subroutine water_integrated
