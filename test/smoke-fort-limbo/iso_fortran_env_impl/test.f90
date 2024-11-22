program kind_arrays
   use iso_fortran_env
   implicit none
   integer :: i
   real(kind=real_kinds(2)) :: dummy
   do i = 1, size(real_kinds)
       print *, real_kinds(i)
   end do
end program kind_arrays
