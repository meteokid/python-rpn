!**s/p adx_set_flux_in - Set F_in_o/F_in_i for Flux calculations based on Aranami et al. (2015)  

subroutine adx_set_flux_in (F_in_o,F_in_i,F_rho,Minx,Maxx,Miny,Maxy,F_nk,k0)

   implicit none

   !Arguments
   !---------
   integer,           intent(in) :: Minx,Maxx,Miny,Maxy                !I, Dimension H
   integer,           intent(in) :: k0                                 !I, Scope of operator
   integer,           intent(in) :: F_nk                               !I, Number of vertical levels
   real, dimension(Minx:Maxx,Miny:Maxy,F_nk), intent(out)   :: F_in_o  !I: Field to interpolate FLUX_out 
   real, dimension(Minx:Maxx,Miny:Maxy,F_nk), intent(out)   :: F_in_i  !I: Field to interpolate FLUX_in 
   real, dimension(Minx:Maxx,Miny:Maxy,F_nk), intent(in)    :: F_rho   !I: Field to interpolate (=1)

   !Author Monique Tanguay

   !revision
   ! v4_80 - Tanguay M.        - initial MPI version

!**/
#include "glb_ld.cdk"

   integer :: i,j,k
   real, dimension(Minx:Maxx,Miny:Maxy,F_nk) :: work 

   !---------------------------------------------------------------------

   !For Flux_out: Set mixing ratio = 0 on NEST at TIME T1
   !-----------------------------------------------------
   work = 0.

   do k=1,F_nk
      do j=1+pil_s,l_nj-pil_n
         do i=1+pil_w,l_ni-pil_e
            work(i,j,k) = F_rho(i,j,k)
         enddo
      enddo
   enddo

   F_in_o = work

   !For Flux_in: Set mixing ratio = 0 on CORE at TIME T1
   !----------------------------------------------------
   work = F_rho

   do k=1,F_nk
      do j=1+pil_s,l_nj-pil_n
         do i=1+pil_w,l_ni-pil_e
            work(i,j,k) = 0.
         enddo
      enddo
   enddo

   F_in_i = work

   return

end subroutine
