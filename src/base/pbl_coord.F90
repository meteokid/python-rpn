subroutine pbl_coord(pbl_s,pbl_se,s,se,pbl_ktop,pbl_zsplit,n,nk,pbl_nk)
  ! Compute sigma values for the PBL vertical coordinate
  implicit none
#include <arch_specific.hf>

  ! Input arguments
  integer, intent(in) :: pbl_ktop                 !top physicsl level index for PBL scheme
  integer, intent(in) :: pbl_zsplit               !number of sub-levels for each physics level
  integer, intent(in) :: n                        !horizontal dimension of slice
  integer, intent(in) :: nk                       !number of physics vertical levels
  integer, intent(in) :: pbl_nk                   !number of levels for the PBL scheme
  real, dimension(n,nk), intent(in) :: s          !momentum-level sigma values in phyiscs
  real, dimension(n,nk), intent(in) :: se         !energy-level sigma values in phyiscs

  ! Output arguments
  real, dimension(n,pbl_nk), intent(out) :: pbl_s !momentum-level sigma values for PBL
  real, dimension(n,pbl_nk), intent(out) :: pbl_se!energy-level sigma values for PBL

  ! Internal variables
  integer :: i,k,split,cnt,nk_all
  real, dimension(n,2*nk) :: s_all

  ! Construct a double-resolution column for profile interpolation
  nk_all = size(s_all,dim=2)
  do i=1,n
     s_all(i,2:nk_all:2) = se(i,:)
     s_all(i,1:nk_all-1:2) = s(i,:)
  enddo

  ! Coordinate setup for high vertical resolution energy levels
  cnt=1
  do k=2*pbl_ktop-1,nk_all-2
     do split=1,pbl_zsplit
        pbl_s(:,cnt) = s_all(:,k) + (split-1)*(s_all(:,k+1)-s_all(:,k))/real(pbl_zsplit)
        cnt = cnt+1
     enddo
  enddo
  pbl_s(:,pbl_nk) = s(:,nk)

  ! Coordinate setup for high resolution momentum levels
  do k=1,pbl_nk-1
     pbl_se(:,k) = .5*(pbl_s(:,k)+pbl_s(:,k+1))
  enddo
  pbl_se(:,pbl_nk) = 1.

end subroutine pbl_coord
