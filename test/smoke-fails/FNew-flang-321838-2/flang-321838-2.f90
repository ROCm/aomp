subroutine test(arr_size)
   implicit none
   integer, intent(in) :: arr_size
   integer, dimension(arr_size) :: arr
   integer i
   integer fail 
   fail = 0
   !$omp target teams map(tofrom: fail)  private(arr)
      arr(:) = 0
      do i=1, arr_size
        if (arr(i) .ne. 0) then
           fail = 1
        endif
      enddo
   !$omp end target teams
   if (fail .eq. 1) then
     write(*,*)"ERROR: wrong answer"
     stop 2
   endif
end subroutine test

program main
   implicit none
   call test(10)
   print *, "PASS"
   return
end program main
