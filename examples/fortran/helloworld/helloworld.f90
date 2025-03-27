program helloworld
  implicit none
  write(*,*) 'Hello CPU world!'
  !$omp target
  write(*,*) 'Hello GPU world!'
  !$omp end target
end program helloworld
