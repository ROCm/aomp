program loop_test

      implicit none
      integer   :: i
      complex(8) :: D,R

      D=(1,2)

      !$OMP TARGET PARALLEL DO MAP(FROM:R) MAP(TO:D)
      do i=1, 2
         R=EXP(D)
      end do
      !$OMP END TARGET PARALLEL DO

end program loop_test
