module reproducer
    implicit none
    type::inner
        sequence
        integer :: intgr
        type(inner), pointer :: ptr
    end type
    type::outer
        sequence
        real, allocatable :: arr
        type(inner), pointer :: ptr
    end type
contains
    subroutine routine(arg)
        implicit none
        type(outer) :: arg
        !$omp target enter data map(to:arg)
    end subroutine routine
end module reproducer