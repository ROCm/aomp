program loop_test

      implicit none
      integer   :: i
      ! cange type from complex to doub le and it passes
      !double precision   :: C
      complex :: C
      DOUBLE complex :: D
      C=(0,0)
      D=(0,0)

      !$OMP TARGET MAP(TOFROM: C) 
         C=C+(1,2)
      !$OMP END TARGET 
      write(*, *) "C= ", C
      !$OMP TARGET MAP(TOFROM: D)
         D=D+(1,2)
      !$OMP END TARGET 

      write(*, *) "D= ", D

end program loop_test
