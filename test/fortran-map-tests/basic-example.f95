! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    implicit none
    INTEGER :: var_a(10) = (/0,0,0,0,0,0,0,0,0,0/)
    !$omp declare target link(var_a)
end module mod_a

subroutine sub_a(var_a)
integer, intent(in), dimension(10) :: var_a
integer var_b
do var_b = 1, 10
     PRINT *, var_a(var_b)
end do
end subroutine

program prog_a
    use mod_a
    implicit none
    integer :: var_b

!$omp target map(tofrom:var_a)
    do var_b = 1, 10
        var_a(var_b) = var_b;
    end do
!$omp end target

call sub_a(var_a)

do var_b = 1, 10
    if (var_a(var_b) /= var_b) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
end do

print*, "======= FORTRAN Test passed! ======="
end program
