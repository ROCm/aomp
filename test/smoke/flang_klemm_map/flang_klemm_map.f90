! compile with
! flang -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 -o map map.F90
!
! Expected output
! 0.000000000000000         0.000000000000000         0.000000000000000
! 0.000000000000000         0.000000000000000         0.000000000000000
! 0.000000000000000         0.000000000000000         0.000000000000000
! 0.000000000000000

! 42.00000000000000         42.00000000000000         42.00000000000000
! 42.00000000000000         42.00000000000000         42.00000000000000
! 42.00000000000000         42.00000000000000         42.00000000000000
! 42.00000000000000

subroutine proc(arr, n)
    use iso_fortran_env

    implicit none

    integer                         :: n
    real(kind=real64), dimension(*) :: arr

    integer :: i

!$omp target map(tofrom:arr(1:n))
    do i = 1,n
        arr(i) = 42.0
    end do
!$omp end target

end subroutine

program map
    use iso_fortran_env

    implicit none

    integer, parameter :: N = 1000
    integer            :: i

    real(kind=real64), dimension(N) :: array

    array(:) = 0.0

    write (*,*) array(1:N/100)
    write (*,*)
    call proc(array, N)
    write (*,*) array(1:N/100)

end program
