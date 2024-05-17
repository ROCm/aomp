! How to reproduce
! make
! ./main
! Error:
! "PluginInterface" error: Failure to init kernel: Error in hsa_executable_get_symbol_by_name(
!   (__omp_offloading_fd00_6f803fb__QFPtest_offload_l23.kd):
! HSA_STATUS_ERROR_INVALID_SYMBOL_NAME: There is no symbol with the given name.

program sumbench
use, intrinsic :: iso_fortran_env
implicit none
    integer, parameter :: n = 1024
    integer(int32) :: x(n), sum_x, y(n)
    integer :: i, i_cutoff, j
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
            integer :: ii
            xout = 0.0
!$omp target parallel do map(to:xin) map(from: xout) 
            do ii=1,1024
                xout(ii) = xin(ii)
            end do
!$omp end target parallel do 
        end function test_offload
end program sumbench
