! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

subroutine sub_a(var_a)
implicit none
real, intent(in), dimension(10) :: var_a
integer var_b
do var_b = 1, 10
     PRINT *, var_a(var_b)
end do
end subroutine

program prog_a
implicit none
REAL :: var_a(10) = (/0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5/)
integer :: var_b 

!$omp target map(tofrom:var_a)
    do var_b = 1, 10
        var_a(var_b) = var_a(var_b) + var_b
    end do
!$omp end target

call sub_a(var_a)

do var_b = 1, 10
    if (var_a(var_b) /= var_b + 0.5) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
end do

print*, "======= FORTRAN Test passed! ======="
end program
