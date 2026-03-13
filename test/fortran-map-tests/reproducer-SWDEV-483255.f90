! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

subroutine sub_a(var_a, var_b)
    !$omp declare target
    real :: var_b
    integer :: var_a
    var_b = var_b - var_a
end subroutine

function func_a(var_a)
    !$omp declare target
    real, intent(in) :: var_a
    func_a = var_a
end function func_a

program prog_a
real var_a
!$omp declare target(sub_a)
!$omp declare target(func_a)
var_a = func_a(var_a)
call sub_a(2,var_a)
call sub_b(var_a)
return
end

subroutine sub_b(var_a)
real var_a
integer var_b
!$omp declare target(sub_a)
!$omp declare target(func_a)
!$omp target
    var_a = func_a(var_a)
    call sub_a(var_b,var_a)
!$omp end target
return
end
