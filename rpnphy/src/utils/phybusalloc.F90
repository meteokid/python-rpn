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

module phybusalloc_mod
   private
   public :: phybusalloc
   
contains

   subroutine phybusalloc(p_nj, entbus, perbus, dynbus, volbus)
      implicit none
#include <arch_specific.hf>

      integer, intent(in) :: p_nj
      real, pointer, dimension(:,:) :: entbus, dynbus, perbus, volbus

#include <gmm.hf>
      include "buses.cdk"

      integer :: gmmstat
      type(gmm_metadata) :: meta_busent, meta_busper, meta_busdyn, meta_busvol
      !---------------------------------------------------------------
      buslck = .true.

      esp_busent = entpar(enttop,1)+entpar(enttop,2)-1
      esp_busper = perpar(pertop,1)+perpar(pertop,2)-1
      esp_busdyn = dynpar(dyntop,1)+dynpar(dyntop,2)-1
      esp_busvol = volpar(voltop,1)+volpar(voltop,2)-1

      call gmm_build_meta2D(meta_busent, 1,esp_busent,0,0,esp_busent, 1,p_nj,0,0,p_nj, 0,GMM_NULL_FLAGS )
      call gmm_build_meta2D(meta_busper, 1,esp_busper,0,0,esp_busper, 1,p_nj,0,0,p_nj, 0,GMM_NULL_FLAGS )
      call gmm_build_meta2D(meta_busdyn, 1,esp_busdyn,0,0,esp_busdyn, 1,p_nj,0,0,p_nj, 0,GMM_NULL_FLAGS )
      call gmm_build_meta2D(meta_busvol, 1,esp_busvol,0,0,esp_busvol, 1,p_nj,0,0,p_nj, 0,GMM_NULL_FLAGS )

      nullify(entbus, perbus, dynbus, volbus)

      gmmstat = gmm_create('BUSENT_3d', entbus, meta_busent, GMM_FLAG_IZER)
      gmmstat = gmm_create('BUSPER_3d', perbus, meta_busper, GMM_FLAG_IZER+GMM_FLAG_RSTR)
      gmmstat = gmm_create('BUSDYN_3d', dynbus, meta_busdyn, GMM_FLAG_IZER)
      gmmstat = gmm_create('BUSVOL_3d', volbus, meta_busvol, GMM_FLAG_IZER)

      gmmstat = gmm_get('BUSENT_3d',entbus)
      gmmstat = gmm_get('BUSPER_3d',perbus)
      gmmstat = gmm_get('BUSDYN_3d',dynbus)
      gmmstat = gmm_get('BUSVOL_3d',volbus)

      !---------------------------------------------------------------
      return
   end subroutine phybusalloc

end module phybusalloc_mod
