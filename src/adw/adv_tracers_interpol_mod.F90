module adv_tracers_interpol_mod
   ! 
   !
   ! Author
   !     Rabah Aider, Stéphane Gaudreault, Stéphane Chamberland -- March 2015
   !
   ! Revision
   !

   use adv_interpolation_mod , only :  adv_interpol_cubic
   
   implicit none

   private
#include "grd.cdk"
#include "glb_ld.cdk"
#include "adv_dims.cdk"
#include "adv_pos.cdk"
#include "lam.cdk"
#include "tr3d.cdk" 
#include "gmm.hf"

  public ::  adv_tracers_interp_lam

contains

   subroutine adv_tracers_interp_lam (F_water_tracers_only_L)
                                
   implicit none
#include <arch_specific.hf>

   !@objective Perform advection of all tracers
   !@arguments
   logical, intent(IN) :: F_water_tracers_only_L

   !@author  Michel Desgagne
   !@revisions
   ! v4_70 - Desgagne          - Initial Version
   ! v4_XX - Tanguay M.        - GEM4 Mass-Conservation


   logical qw_L
   integer  n,i0,j0,in,jn ,jext
   type(gmm_metadata) :: mymeta
   real, pointer, dimension (:,:,:) :: fld_in, fld_out
   integer  err
  
     
      jext=1
      if (Grd_yinyang_L) jext=2
      if (l_west)  i0 =        pil_w - jext
      if (l_east)  in = l_ni - pil_e + jext
      if (l_south) j0 =        pil_s - jext
      if (l_north) jn = l_nj - pil_n + jext


    do n=1,Tr3d_ntr
      qw_L= Tr3d_wload(n) .or. Tr3d_name_S(n)(1:2).eq.'HU'
      if (F_water_tracers_only_L) then
         if (.not. qw_L) cycle
      else
         if (qw_L) cycle
      endif
          
   err =     gmm_get('TR/'//trim(Tr3d_name_S(n))//':P' ,fld_in ,mymeta)
   err = min(gmm_get('TR/'//trim(Tr3d_name_S(n))//':M' ,fld_out,mymeta),err)

  
      
        call adv_interpol_cubic ('TR/'//trim(Tr3d_name_S(n))//':M', fld_out , fld_in,  &
		                 pxt, pyt, pzt,i0, in, j0, jn, &
 	                         Lam_gbpil_t+1, 't', Tr3d_mono(n), Tr3d_mass(n)) 
        
 
      end do
          
    
      end subroutine adv_tracers_interp_lam 

   end module adv_tracers_interpol_mod

   
  
