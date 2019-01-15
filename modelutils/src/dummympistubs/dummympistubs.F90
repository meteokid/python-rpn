 
integer function mpi_comm_free()
   mpi_comm_free = -1
   print *,"ERROR: called a stub for mpi_comm_free"
   call flush(6)
   stop
end function mpi_comm_free
