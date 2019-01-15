      function mapmod2phy(F_imod,F_jmod,F_lcl_ni,F_dim_ni)
      implicit none

      integer, dimension(2) :: mapmod2phy
      integer :: F_imod, F_jmod, F_lcl_ni, F_dim_ni

      integer :: nmod,ijm,iphy,jphy
!
!     ---------------------------------------------------------------
!

      nmod = (F_jmod-1) * F_lcl_ni
      ijm = nmod+F_imod
      jphy= ijm/F_dim_ni + min(1,mod(ijm,F_dim_ni))
      iphy= ijm - (jphy-1)*F_dim_ni
      mapmod2phy(1) = iphy
      mapmod2phy(2) = jphy
!
      end function mapmod2phy
