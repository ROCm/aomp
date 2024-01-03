program sizeof_double
  use iso_c_binding
  implicit none
  integer, parameter :: integer_size = c_sizeof(1)
  integer, parameter :: real_size = c_sizeof(1.0e-1)
  integer, parameter :: double_size = c_sizeof(1.0d-1)
  write(*,*) "sizeof integer: ", integer_size
  write(*,*) "sizeof real:    ", real_size
  write(*,*) "sizeof double:  ", double_size
end program
