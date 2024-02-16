program label_and_default
    implicit none
    integer :: i

    !$omp parallel default(none)
        loop: do i = 1, 10
        end do loop
    !$omp end parallel
end program
