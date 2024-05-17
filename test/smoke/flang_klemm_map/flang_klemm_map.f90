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

subroutine proc(arr, brr, n)
    use iso_fortran_env

    implicit none

    integer                         :: n
    real(kind=real64), dimension(n) :: arr
    real(kind=real64), dimension(*) :: brr

    integer :: i

!$omp target map(tofrom:arr(:n), brr(:n))
    do i = 1,n
        arr(i) = 42.0
        brr(i) = 43.0
    end do
!$omp end target

end subroutine

program map
    use iso_fortran_env

    implicit none

    integer, parameter :: N = 1000
    integer            :: i

    real(kind=real64), dimension(N) :: array
    real(kind=real64), dimension(N) :: array1

    array(:) = 0.0
    array1(:) = 0.0

    write (*,*) array(1:N/100)
    write (*,*) array1(1:N/100)
    write (*,*)
    call proc(array, array1, N)
    write (*,*) array(1:N/100)
    write (*,*) array1(1:N/100)
    if (array1(2) .ne. 43.0) write (*,*)'Failed assumed dim'
end program
