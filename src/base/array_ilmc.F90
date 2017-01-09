module array_ilmc 
save
type a_ilmc 
sequence
real, pointer, dimension(:):: i_rd ! I indice of neighbor in a given sweep
real, pointer, dimension(:):: j_rd ! J indice of neighbor in a given sweep
real, pointer, dimension(:):: k_rd ! J indice of neighbor in a given sweep
integer                    :: cell ! # neighbors in a given sweep 
end type a_ilmc 
type (a_ilmc), dimension(:,:,:,:), pointer :: sweep_rd
end module array_ilmc 
