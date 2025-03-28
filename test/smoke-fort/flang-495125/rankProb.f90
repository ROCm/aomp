module assumed_rank_for_c_interfacing

    use iso_c_binding, only: c_loc, c_ptr, c_associated

    implicit none

    private
    public :: get_pointer

interface get_pointer

    module type(c_ptr) function get_host_pointer(host_data)
        type(*), intent(in), target :: host_data(..)
    end function get_host_pointer

end interface get_pointer

end module assumed_rank_for_c_interfacing

submodule(assumed_rank_for_c_interfacing) submodule_host

contains

    module type(c_ptr) function get_host_pointer(host_data)
        type(*), intent(in), target :: host_data(..)
        get_host_pointer = c_loc(host_data)
        if (.not. c_associated(get_host_pointer) ) error stop "Error(get_host_pointer): host_data is not allocated"
    end function get_host_pointer

end submodule submodule_host

program testing_assumed_rank_for_c_interfacing

    use iso_c_binding, only: c_loc, c_ptr, c_associated, c_intptr_t
    use iso_fortran_env, only: i32=>int32, r64=>real64
    use assumed_rank_for_c_interfacing, only: get_pointer

    implicit none

    real(r64), allocatable, target :: a(:)
    integer(i32), target :: b
    character(4), parameter :: c = 'test'

    integer(c_intptr_t) :: p, p_ref, p_ref_2

    ! Test the function
    if (.not. c_associated(get_pointer(c))) error stop "TEST FAILURE"

    p     = transfer(get_pointer(b), p)
    p_ref = transfer(c_loc(b), p_ref)

    if (.not. c_associated(get_pointer(b)) .or. p /= p_ref) error stop "TEST FAILURE"

    allocate(a(1000),source=1.0_r64)

    p     = transfer(get_pointer(a), p)
    p_ref = transfer(c_loc(a), p_ref)

    if (.not. c_associated(get_pointer(a)) .or. p /= p_ref) error stop "TEST FAILURE"

    p       = transfer(get_pointer(a(500)), p)
    p_ref   = transfer(c_loc(a(500)), p_ref)
    p_ref_2 = transfer(c_loc(a), p_ref)

    if (.not. c_associated(get_pointer(a)) .or. p /= p_ref .or. p == p_ref_2) error stop "TEST FAILURE"

    deallocate(a)

    write(*,*) 'TEST SUCCESS'

end program testing_assumed_rank_for_c_interfacing
