program loop_test

      implicit none
      integer   :: i
      ! cange type from complex to doub le and it passes
      double complex   :: D
      complex :: C
      C=(0,0)
      D=(0,0)

      !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO MAP(TOFROM: C) NUM_TEAMS(1)
      do i=1, 10
       !$omp atomic update
         C=C+(1,1)
       !$omp end atomic
      end do
      !$OMP END TARGET TEAMS DISTRIBUTE PARALLEL DO
      write(*, *) "C= ", C

      !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO MAP(TOFROM: D) NUM_TEAMS(1)
      do i=1, 10
       !$omp critical
         D=D+(1,2)
       !$omp end critical
      end do
      !$OMP END TARGET TEAMS DISTRIBUTE PARALLEL DO
      write(*, *) "D= ", D

end program loop_test
