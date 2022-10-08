subroutine test(arr_size)
   implicit none
   integer, intent(in) :: arr_size
   integer, dimension(arr_size) :: arr

   !$omp target teams private(arr)
      arr(:) = 0
   !$omp end target teams
end subroutine test

program main
   implicit none
   call test(10)
end program main
