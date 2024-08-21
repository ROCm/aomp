program atomic_capture
    implicit none

    integer :: some_value
    integer :: other_value
    integer :: v, sz

    logical :: some_cond

    some_value = 1
    other_value = 1
    !$omp parallel
        !$omp atomic capture
            some_value = other_value
            other_value = other_value + (v-1)/sz + 1
        !$omp end atomic
    !$omp end parallel
end program atomic_capture
