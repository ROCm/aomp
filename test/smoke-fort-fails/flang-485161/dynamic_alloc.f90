module dynamic_alloc
  implicit none

  contains
    subroutine sub_dynamic_alloc(Ain, Aout)
      !$omp declare target
      double precision, intent(in)  :: Ain(:)
      double precision, intent(out) :: Aout(size(Ain))

      double precision :: Atmp(size(Ain))
      Atmp = Ain + 1
      Aout = Atmp + 1
    end subroutine sub_dynamic_alloc

    function func_dynamic_alloc(Ain) result(Aout)
      !$omp declare target
      double precision, intent(in) :: Ain(:)
      double precision :: Aout(size(Ain))

      double precision :: Atmp(size(Ain))

      Atmp = Ain + 1.0
      Aout = Atmp + 1.0

      return
    end function func_dynamic_alloc

end module dynamic_alloc

program main
  use dynamic_alloc
  implicit none
  double precision :: A(50), B(50), C(50)
  integer i
  do i = 1, size(A)
      A(i) = i
  enddo
  B = 0
  C = 0
!$omp target
  call sub_dynamic_alloc(A, B)
  C = func_dynamic_alloc(A)
!$omp end target
  print *, "A = "
  print *, A
  print *, "B = "
  print *, B
  print *, "C = "
  print *, C
  do i = 1, size(A)
      if ((A(i)+2) /= B(i)) then
          print *, "Error B @", i
          stop(1)
      endif
      if ((A(i)+2) /= C(i)) then
          print *, "Error C @", i
          stop(2)
      endif
  enddo
  print *, "Success"
end program
