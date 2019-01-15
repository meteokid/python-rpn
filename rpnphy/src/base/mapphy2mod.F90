      function mapphy2mod(F_iphy,F_jphy,F_offi,F_offj,F_lcl_ni,F_lcl_nj,F_dim_ni)
      implicit none

      integer, dimension(2) :: mapphy2mod
      integer, intent(in)  :: F_iphy, F_jphy,F_offi,F_offj,F_lcl_ni,F_lcl_nj,F_dim_ni

      integer :: nphy,ijp,jmod
!
!     ---------------------------------------------------------------
!
      nphy = (F_jphy-1) * F_dim_ni
      ijp = min (nphy+F_iphy,F_lcl_ni*F_lcl_nj)
      jmod= ijp/F_lcl_ni + min(1,mod(ijp,F_lcl_ni))
      mapphy2mod(1)= ijp - (jmod-1)*F_lcl_ni + F_offi
      mapphy2mod(2)= jmod + F_offj

!
      end function mapphy2mod
