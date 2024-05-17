program loop_test

      implicit none
      integer   :: i
      complex   :: C,D
      complex   :: C_ref,D_ref
      C=(0,0)
      D=(1,2)
      C_ref=(0,0)
      D_ref=(1,2)

      !$OMP TARGET PARALLEL DO REDUCTION(+:C)
      do i=1, 10
         C=C+EXP(D)
      end do
      !$OMP  END TARGET PARALLEL DO

      do i=1, 10
         C_ref=C_ref+EXP(D_ref)
      end do

      write(*, *) "C= ", C
      write(*, *) "C_ref= ", C_ref

      if (C .ne. C_ref) then
             write(*,*) "Failed"
             stop 2
      endif
end program loop_test

