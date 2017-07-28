module type_mod
   implicit none
   public
   save
   type :: vertical
      sequence
      real  , dimension(:),pointer :: t,m
   end type vertical

   type :: vertical_8
      sequence
      real*8, dimension(:),pointer :: t,m
   end type vertical_8

   type :: vertical_i
      sequence
      integer, dimension(:),pointer :: t,m
   end type vertical_i

end module type_mod
