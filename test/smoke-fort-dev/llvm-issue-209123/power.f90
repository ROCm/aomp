module m
contains
    subroutine s()
      integer :: n1
      real :: tmp1
      !$omp declare target
      tmp1 = 2**n1
    end subroutine s
end module
