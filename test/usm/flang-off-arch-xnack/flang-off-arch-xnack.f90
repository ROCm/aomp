program test
      integer :: i
      i = 0
     !$omp target map(from:i)
      i = 1
     !$omp end target
      if (i .ne. 1) then
        print *,'failed'
        call kill(0,4)
      endif
end program test
