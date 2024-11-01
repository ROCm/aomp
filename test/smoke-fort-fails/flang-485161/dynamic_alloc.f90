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
  implicit none
end program
