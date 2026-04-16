module kernels
    implicit none
contains
    subroutine kernel(array, n)
        implicit none
        integer, allocatable, dimension(:) :: array
        integer :: n
        integer :: i

        !$omp target teams distribute parallel do has_device_addr(array)
            do i = 1, n
                array(i) = array(i) / 2
            end do
        !$omp end target teams distribute parallel do
    end subroutine kernel

end module kernels


program test
    use kernels
    implicit none

    integer, parameter :: n = 2*1024*1028

    integer, allocatable, dimension(:) :: array
    integer, dimension(n) :: chk
    integer :: i

    allocate(array(n))

    array = 42
    chk = 0

    call kernel(array, n)

    !$omp target teams distribute parallel do map(from:chk) has_device_addr(array)
        do i = 1, n
            chk(i) = array(i)
        end do
    !$omp end target teams distribute parallel do

    if (.not.all(array.eq.21)) then
        print '(A)', 'FAIL'
        stop 1
    else
        print '(A)', 'PASS'
    end if
end program test