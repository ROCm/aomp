program testFortranRuntime
use, intrinsic :: iso_fortran_env
implicit none
    integer, parameter :: n = 1024
    integer(int32) :: x(n), y(n)
    integer :: i
    x = 1
    y = 0
    y = test_offload(x)
    do i = 1,n
      if ( x(i) .ne. y(i)) then
        stop 1
      endif
    enddo
    print *, "Test passed"

    contains
        function test_offload(xin) result(xout)
            integer(int32), intent(in) :: xin(1024)
            integer(int32) :: xout(1024)
            xout = 0.0
!$omp target map(to:xin) map(from: xout)
                xout = xin
!$omp end target
        end function test_offload
end program testFortranRuntime
