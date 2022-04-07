module test_aomp

implicit none
integer :: nsize
#define STR_LEN_39 39
#if defined(STR_LEN_39)
integer, parameter :: StrKIND = 39
#elif defined(STR_LEN_64)
integer, parameter :: StrKIND = 64
#elif defined(STR_LEN_512)
integer, parameter :: StrKIND = 512
#else
integer, parameter :: StrKIND = 38
#endif
character(len=strKIND), pointer :: in_str1
character(len=strKIND), pointer :: in_str2

contains
    subroutine assign_str(in_str)
        integer i,j,k,l,m
        character(len=strKIND), intent(inout), pointer :: in_str

        if (.not. associated(in_str1)) then
            allocate(in_str1)
!            !$omp target enter data map(alloc:in_str1)
            !$omp target
            in_str1 = "Luise"
            !$omp end target
        end if
        if (.not. associated(in_str2)) then
            allocate(in_str2)
!            !$omp target enter data map(alloc:in_str1)
            !$omp target
            in_str2 = "Shadow"
            !$omp end target
        end if
        if( command_argument_count() .gt. 0 )then
            in_str => in_str1
        else
            in_str => in_str2
        endif
    end subroutine assign_str

end module test_aomp

program test
    use test_aomp
    character(len=strKIND), target :: in_str
    character(len=strKIND), target :: out_str
    character(len=strKIND), pointer :: src
    character(len=strKIND), pointer :: dst

    call assign_str(src)
    dst => out_str

    do i=1,15
        dst(i:i) = src(i:i)
    end do

    print *, dst
end program test
