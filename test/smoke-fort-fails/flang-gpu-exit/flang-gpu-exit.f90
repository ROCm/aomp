module gpu_exit_impl
    implicit none
contains
    subroutine exit_with_integer()
        implicit none
        print '(A)', 'Launching kernel'
        !$omp target
            call exit(1)
        !$omp end target
        print '(A)', 'This should not be printed!'
    end subroutine exit_with_integer
end module gpu_exit_impl

program gpu_stop
    use gpu_exit_impl
    implicit none
    call exit_with_integer()
end program gpu_stop
