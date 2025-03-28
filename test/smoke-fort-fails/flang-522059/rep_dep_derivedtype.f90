!****************************
program depend
   type :: dt
       integer :: i
   end type

   type(dt) :: var

   !$omp parallel
       !$omp single
           !$omp task depend(out:var%i)
               call sleep(1)
               print *, 'A'
           !$omp end task

           !$omp task depend(in:var%i)
               print *, 'B'
           !$omp end task
       !$omp end single
   !$omp end parallel
end program depend
!****************************
