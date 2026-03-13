! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

subroutine sub_a(var_a, var_b)
    implicit none
    integer(8) :: var_a,var_b
    integer :: var_c(var_a:var_b)
   !$omp target
       var_c(4) = 22
       var_c(15) = 2222
   !$omp end target
    print *, var_c

    if (var_c(4) /= 22) then
        print *, "======= FORTRAN Test Failed! ======="
        stop 1
    end if

    if (var_c(15) /= 2222) then
        print *, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
end subroutine sub_a

subroutine sub_b(var_a, var_b, var_c)
    implicit none
    integer(8) :: var_a,var_b
    integer :: var_c(var_a:var_b)
   !$omp target
       var_c(4) = 22
       var_c(15) = 2222
   !$omp end target
end subroutine sub_b

program prog_a
    implicit none
    integer(8) :: var_a,var_b
    integer :: var_c(20)
    var_a = 4
    var_b = 15
    call sub_a(var_a,var_b)
    call sub_b(var_a, var_b, var_c)
    print *, var_c

    if (var_c(1) /= 22) then
        print *, "======= FORTRAN Test Failed! ======="
        stop 1
    end if

    if (var_c(12) /= 2222) then
        print *, "======= FORTRAN Test Failed! ======="
        stop 1
    end if

     print *, "======= FORTRAN Test Passed! ======="
end program
