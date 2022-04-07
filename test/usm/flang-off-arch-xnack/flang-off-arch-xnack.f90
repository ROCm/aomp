program test
      integer :: i, getpid
      i = 0
     !$omp target map(from:i)
      i = 1
     !$omp end target
      if (i .ne. 1) then
        print *,'failed'
        call kill(getpid(),4)
      endif
end program test
