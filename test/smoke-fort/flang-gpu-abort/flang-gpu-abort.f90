module gpu_abort_impl
    implicit none
contains
    subroutine abort_it()
        implicit none
        print '(A)', 'Launching kernel'
        !$omp target
            call abort()
        !$omp end target
        print '(A)', 'This should not be printed!'
    end subroutine abort_it
end module gpu_abort_impl

program gpu_stop
    use gpu_abort_impl
    implicit none
    call abort_it()
end program gpu_stop
