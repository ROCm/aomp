program test
use iso_c_binding
implicit none

    integer, target :: array0(7), array1(7)
    call test_subroutine([c_loc(array0), c_loc(array1)]);
    print *, "Test passed"

contains
    subroutine test_subroutine ( C_Pointer )
       type ( c_ptr ), dimension ( : ), intent ( in ) :: &
         C_Pointer
    end subroutine test_subroutine

end program test
