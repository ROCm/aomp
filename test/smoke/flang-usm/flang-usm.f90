program test
      integer :: i(5)
      i(2) = 0
  !$omp requires unified_shared_memory
     !$omp target
      i(2) = 1
     !$omp end target
      if (i(2) .ne. 1) then
        print *,'failed'
        call exit(1)
      endif
end program test
