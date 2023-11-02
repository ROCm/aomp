subroutine test(arr_size)
   implicit none
   integer, intent(in) :: arr_size
   integer, dimension(arr_size) :: arr
   integer i

   !$omp target teams private(arr)
      arr(:) = 0
   !$omp end target teams
   do i=1, arr_size
     if (arr(i) .ne. 0) then
       write(*,*)"ERROR: wrong answer"
       stop 2
     endif
   end do
end subroutine test

program main
   implicit none
   call test(10)
   print *, "PASS"
   return
end program main
