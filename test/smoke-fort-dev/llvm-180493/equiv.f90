program equiv
    use omp_lib
    implicit none
    common/ba/ a,b,c
    integer :: a,b,c
    integer :: x,y,z

    integer, parameter :: nthreads = 4
    integer :: i
    integer :: chk

    logical :: failed

    !$omp threadprivate(/ba/)

    equivalence (x,a)
    equivalence (y,a)

    failed = .false.

    !$omp parallel num_threads(nthreads) shared(failed) private(chk)
        x = 21
        chk = 21
        !$omp masked filter(1)
            x = 42
            chk = 42
        !$omp end masked

        !$omp barrier

        a = a + omp_get_thread_num()
        chk = chk + omp_get_thread_num()

        !$omp barrier
        do i = 0, nthreads-1
            if (omp_get_thread_num() == i) then
                print '(I3,I5)', omp_get_thread_num(), a
                if (chk /= a) then
                    failed = .true.
                end if
            end if
            !$omp barrier
        end do
    !$omp end parallel

    if (failed) then
        stop 1
    end if
end program equiv
